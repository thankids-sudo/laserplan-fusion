package app.laserplan.survey;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;

final class Fusion {
    static final int UNKNOWN=0, OBSERVED=1, ESTIMATED=2;
    static final int NEUTRAL=0xff90a5b6;
    final float[] xyz, smooth, center;
    final int[] colors, source, previewColors, previewSource;
    int photoFrames, observed, estimated;
    /** Depth frames kept for the surface preview. */
    final List<SurfaceBuilder.Frame> depthFrames=new ArrayList<>();
    /** True when any photo had to use the approximate (crop) alignment of older ZIPs. */
    boolean approximateAlignment;
    int minConfidence=128;
    /** Calibrated photos (own intrinsics and pose) kept for the optional AI depth fill. */
    final List<Photo> photos=new ArrayList<>();
    static final class Photo{byte[] jpeg;int width,height;double fx,fy,cx,cy;double[] pose;}
    float maxRange=5;
    float span=.1f;
    private final HashMap<Long,Integer> cells=new HashMap<>();
    private final float voxel;

    Fusion(float[] xyz,float voxel){
        this.xyz=xyz;
        this.voxel=voxel;
        smooth=xyz.clone();
        colors=new int[xyz.length/3];
        source=new int[colors.length];
        previewColors=new int[colors.length];
        previewSource=new int[colors.length];
        Arrays.fill(colors,NEUTRAL);
        float[] min={Float.MAX_VALUE,Float.MAX_VALUE,Float.MAX_VALUE};
        float[] max={-Float.MAX_VALUE,-Float.MAX_VALUE,-Float.MAX_VALUE};
        for(int i=0;i<colors.length;i++){
            for(int j=0;j<3;j++){
                min[j]=Math.min(min[j],xyz[3*i+j]);
                max[j]=Math.max(max[j],xyz[3*i+j]);
            }
            cells.putIfAbsent(cell(i,0,0,0),i);
        }
        center=new float[3];
        for(int j=0;j<3;j++){center[j]=(min[j]+max[j])*.5f;span=Math.max(span,max[j]-min[j]);}
    }

    static Fusion read(File file) throws Exception{
        try(ZipFile zip=new ZipFile(file)){
            JSONObject manifest=json(zip,"manifest.json");
            if(!"laserplan.ar-survey".equals(manifest.optString("schema"))||manifest.optInt("version")!=1)
                throw new IOException("Unsupported LaserPlan ZIP.");
            JSONObject settings=manifest.optJSONObject("settings");
            if(settings==null)settings=new JSONObject();
            float voxel=(float)settings.optDouble("voxelM",.02);
            if(!Float.isFinite(voxel)||voxel<.005||voxel>.2)voxel=.02f;
            JSONObject files=manifest.optJSONObject("files");
            String cloud=files==null?"cloud.xyzc":files.optString("cloud","cloud.xyzc");
            float[] points=new float[1800000];
            int size=0;
            try(BufferedReader reader=new BufferedReader(new InputStreamReader(zip.getInputStream(entry(zip,cloud,64<<20)),StandardCharsets.UTF_8))){
                for(String line;(line=reader.readLine())!=null;){
                    line=line.trim();
                    if(line.isEmpty()||line.startsWith("#"))continue;
                    String[] row=line.split("\\s+");
                    if(row.length!=4||size==points.length)throw new IOException("Invalid or oversized point cloud.");
                    for(int j=0;j<3;j++){
                        float value=Float.parseFloat(row[j]);
                        if(!Float.isFinite(value)||Math.abs(value)>10000)throw new IOException("Invalid scan coordinates.");
                        points[size++]=value;
                    }
                }
            }
            if(size==0)throw new IOException("No points in this scan.");
            Fusion result=new Fusion(Arrays.copyOf(points,size),voxel);
            result.minConfidence=Math.max(0,Math.min(255,settings.optInt("minConfidence",128)));
            result.maxRange=(float)Math.max(.5,Math.min(20,settings.optDouble("maxRangeM",5)));
            float[][] votes=new float[4][size/3];
            float[] depthScratch=new float[size/3];
            int[] pixelScratch=new int[size/3];
            int n=0;
            List<String> names=new ArrayList<>();
            for(Enumeration<? extends ZipEntry> entries=zip.entries();entries.hasMoreElements();){
                String name=entries.nextElement().getName();
                if(name.matches("frames/[0-9]+[.]json"))names.add(name);
            }
            Collections.sort(names);
            for(String name:names){
                if(++n>300)throw new IOException("Too many scan frames.");
                JSONObject frame=json(zip,name);
                if(!frame.optBoolean("valid"))continue;
                // A depth frame whose pose was matched to its own camera frame is usable at any lag;
                // otherwise (older or approximate captures) only within 50 ms of the live pose.
                double lag=Math.abs(frame.optDouble("depthToCameraDeltaNs",0));
                if(!poseMatched(frame)&&lag>50_000_000)continue;
                try{result.depthFrames.add(depthFrame(zip,frame));}catch(Exception skipped){continue;}
                JSONObject rgb=frame.optJSONObject("rgb");
                if(rgb==null||!rgb.has("file")||zip.getEntry(rgb.optString("file"))==null)continue;
                if(result.project(zip,frame,rgb,votes,depthScratch,pixelScratch))result.photoFrames++;
            }
            for(int i=0;i<result.colors.length;i++){
                float w=votes[3][i];
                if(w<=0)continue;
                result.colors[i]=Color.rgb(Math.round(votes[0][i]/w),Math.round(votes[1][i]/w),Math.round(votes[2][i]/w));
                result.source[i]=OBSERVED;result.observed++;
            }
            result.preparePreview();
            return result;
        }
    }

    static boolean poseMatched(JSONObject frame){
        JSONObject pose=frame.optJSONObject("pose");
        String source=pose==null?"":pose.optString("source");
        return source.equals("matched-frame")||source.contains("the depth image belongs to");
    }

    private static double[] matrix(JSONArray values) throws Exception{
        if(values==null||values.length()!=16)return null;
        double[] m=new double[16];
        for(int i=0;i<16;i++){m[i]=values.getDouble(i);if(!Double.isFinite(m[i]))return null;}
        return m;
    }

    private static SurfaceBuilder.Frame depthFrame(ZipFile zip,JSONObject frame) throws Exception{
        JSONObject d=frame.getJSONObject("intrinsics").getJSONObject("depth");
        int w=d.getInt("width"),h=d.getInt("height");
        if(w<2||h<2||(long)w*h>500000)throw new IOException("Invalid depth dimensions.");
        byte[] depth=bytes(zip,frame.getJSONObject("depth").getString("file"),2*w*h);
        if(depth.length!=2*w*h)throw new IOException("Truncated depth frame.");
        SurfaceBuilder.Frame f=new SurfaceBuilder.Frame();
        f.width=w;f.height=h;
        f.depthMm=new short[w*h];
        java.nio.ByteBuffer.wrap(depth).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(f.depthMm);
        JSONObject conf=frame.optJSONObject("confidence");
        if(conf!=null&&conf.has("file")){
            byte[] c=bytes(zip,conf.getString("file"),w*h);
            if(c.length==w*h)f.confidence=c;
        }
        f.pose=matrix(frame.getJSONObject("pose").getJSONArray("matrixColumnMajor"));
        if(f.pose==null)throw new IOException("Invalid pose.");
        f.fx=d.getDouble("fx");f.fy=d.getDouble("fy");f.cx=d.getDouble("cx");f.cy=d.getDouble("cy");
        if(f.fx<=0||f.fy<=0)throw new IOException("Invalid camera focal length.");
        JSONObject pixels=frame.optJSONObject("pixels");
        f.measured=pixels==null?0:pixels.optInt("accepted",0);
        return f;
    }

    /**
     * Colours the cloud from one photo. A point takes this photo's colour only when it is the visible
     * surface in that photo: a depth buffer of the cloud itself (8 px cells) rejects points hidden behind
     * nearer ones. Samples are weighted by 1/distance² so close, sharp views dominate the blend.
     */
    private boolean project(ZipFile zip,JSONObject frame,JSONObject rgb,float[][] votes,float[] pz,int[] pp) throws Exception {
        byte[] photo=bytes(zip,rgb.getString("file"),12<<20);
        BitmapFactory.Options bounds=new BitmapFactory.Options();
        bounds.inJustDecodeBounds=true;
        BitmapFactory.decodeByteArray(photo,0,photo.length,bounds);
        int rw=bounds.outWidth,rh=bounds.outHeight;
        if(rw<1||rh<1||(long)rw*rh>12_000_000||rw!=rgb.optInt("width")||rh!=rgb.optInt("height"))return false;
        JSONObject d=frame.getJSONObject("intrinsics").getJSONObject("depth");
        int w=d.getInt("width"),h=d.getInt("height");
        double fx=d.getDouble("fx"),fy=d.getDouble("fy"),cx=d.getDouble("cx"),cy=d.getDouble("cy");
        JSONObject k=rgb.optJSONObject("intrinsics");
        double rfx,rfy,rcx,rcy;
        if(k!=null){rfx=k.getDouble("fx");rfy=k.getDouble("fy");rcx=k.getDouble("cx");rcy=k.getDouble("cy");}
        else{
            // Older LaserPlan ZIPs have no photo calibration: centred-crop estimate, labelled approximate.
            double crop=Math.max((double)rw/w,(double)rh/h);
            rfx=fx*crop;rfy=fy*crop;rcx=(cx+.5-w*.5)*crop+rw*.5-.5;rcy=(cy+.5-h*.5)*crop+rh*.5-.5;
            approximateAlignment=true;
        }
        if(rfx<=0||rfy<=0)return false;
        // Photo pose: Fusion stores rgb.pose.matrixColumnMajor, LaserPlan 13.2 rgb.poseMatrixColumnMajor.
        JSONObject rp=rgb.optJSONObject("pose");
        double[] r=matrix(rp!=null?rp.optJSONArray("matrixColumnMajor"):rgb.optJSONArray("poseMatrixColumnMajor"));
        if(r==null){r=matrix(frame.getJSONObject("pose").getJSONArray("matrixColumnMajor"));approximateAlignment=true;}
        if(r==null)return false;
        Bitmap image=BitmapFactory.decodeByteArray(photo,0,photo.length);
        if(image==null)return false;
        if(k!=null&&(rp!=null||rgb.has("poseMatrixColumnMajor"))){
            Photo kept=new Photo();
            kept.jpeg=photo;kept.width=rw;kept.height=rh;kept.fx=rfx;kept.fy=rfy;kept.cx=rcx;kept.cy=rcy;kept.pose=r;
            photos.add(kept);
        }
        try{
            int[] pixels=new int[rw*rh];
            image.getPixels(pixels,0,rw,0,0,rw,rh);
            int gw=(rw+7)/8,gh=(rh+7)/8;
            float[] nearest=new float[gw*gh];
            Arrays.fill(nearest,Float.MAX_VALUE);
            int count=colors.length;
            for(int i=0;i<count;i++){
                pp[i]=-1;
                double x=xyz[3*i]-r[12],y=xyz[3*i+1]-r[13],z=xyz[3*i+2]-r[14];
                double a=r[0]*x+r[1]*y+r[2]*z,b=r[4]*x+r[5]*y+r[6]*z,c=-(r[8]*x+r[9]*y+r[10]*z);
                if(c<=.1)continue;
                double u=rfx*a/c+rcx,v=rcy-rfy*b/c;
                if(u<0||v<0||u>=rw-1||v>=rh-1)continue;
                int px=(int)u,py=(int)v,g=(py>>3)*gw+(px>>3);
                pp[i]=py*rw+px;pz[i]=(float)c;
                if(c<nearest[g])nearest[g]=(float)c;
            }
            boolean contributed=false;
            for(int i=0;i<count;i++){
                if(pp[i]<0)continue;
                int px=pp[i]%rw,py=pp[i]/rw,g=(py>>3)*gw+(px>>3);
                float c=pz[i];
                if(c>nearest[g]+.05f+.02f*c)continue;   // hidden behind a nearer surface in this photo
                int color=pixels[pp[i]];
                float weight=1f/(Math.max(c,.3f)*Math.max(c,.3f));
                votes[0][i]+=weight*Color.red(color);
                votes[1][i]+=weight*Color.green(color);
                votes[2][i]+=weight*Color.blue(color);
                votes[3][i]+=weight;
                contributed=true;
            }
            return contributed;
        }finally{image.recycle();}
    }

    /** Fused colour near a position for the surface (photo or estimated); 0 when nothing within 3 cm. */
    int colorNear(float x,float y,float z,int[] sourceOut){
        long bx=(long)Math.floor(x/voxel),by=(long)Math.floor(y/voxel),bz=(long)Math.floor(z/voxel);
        int best=-1;float bestDistance=.03f*.03f;
        for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++)for(int dz=-1;dz<=1;dz++){
            Integer j=cells.get(((bx+dx)&0x1fffffL)<<42|((by+dy)&0x1fffffL)<<21|((bz+dz)&0x1fffffL));
            if(j==null||previewSource[j]==UNKNOWN)continue;
            float a=xyz[3*j]-x,b=xyz[3*j+1]-y,c=xyz[3*j+2]-z,dd=a*a+b*b+c*c;
            if(dd<bestDistance){bestDistance=dd;best=j;}
        }
        if(best<0)return 0;
        sourceOut[0]=previewSource[best]==OBSERVED?SurfaceBuilder.PHOTO:SurfaceBuilder.ESTIMATED;
        return previewColors[best];
    }

    SurfaceBuilder.Mesh surface(int smoothing,int fill){return surface(smoothing,fill,null);}

    /** Surface from measured depth frames; optional AI frames come last, so they only add what is still open. */
    SurfaceBuilder.Mesh surface(int smoothing,int fill,List<SurfaceBuilder.Frame> ai){
        List<SurfaceBuilder.Frame> frames=depthFrames;
        if(ai!=null&&!ai.isEmpty()){frames=new ArrayList<>(depthFrames);frames.addAll(ai);}
        return SurfaceBuilder.build(frames,this::colorNear,smoothing,fill,minConfidence,maxRange,300_000);
    }

    /** Result of running the AI depth fill over all calibrated photos. */
    static final class AiResult{
        final List<SurfaceBuilder.Frame> frames=new ArrayList<>();
        int photos,fitted,cells;long inferenceMs,totalMs;String model;
    }

    static final int AI_GW=80,AI_GH=60;   // photo cell grid (8 px cells on a 640×480 photo)

    /**
     * Runs the depth model on each calibrated photo and turns enclosed, same-surface holes into extra
     * surface frames (GuidedFill). Measured cells are included so the estimate joins the measured surface.
     */
    AiResult aiFrames(DepthModel model){
        AiResult result=new AiResult();
        result.model=model.label();
        long start=System.nanoTime();
        int gw=AI_GW,gh=AI_GH,side=model.side();
        for(Photo p:photos){
            Bitmap image=BitmapFactory.decodeByteArray(p.jpeg,0,p.jpeg.length);
            if(image==null)continue;
            try{
                result.photos++;
                long t=System.nanoTime();
                float[] out;
                try{out=model.predict(image);}catch(Exception e){continue;}
                result.inferenceMs+=(System.nanoTime()-t)/1_000_000L;
                double cellW=(double)p.width/gw,cellH=(double)p.height/gh;
                // Measured depth per cell: nearest visible cloud point (the cloud is the measured data).
                float[] depth=new float[gw*gh];
                double[] r=p.pose;
                for(int i=0;i<colors.length;i++){
                    double x=xyz[3*i]-r[12],y=xyz[3*i+1]-r[13],z=xyz[3*i+2]-r[14];
                    double a=r[0]*x+r[1]*y+r[2]*z,b=r[4]*x+r[5]*y+r[6]*z,c=-(r[8]*x+r[9]*y+r[10]*z);
                    if(c<=.1)continue;
                    double u=p.fx*a/c+p.cx,v=p.cy-p.fy*b/c;
                    if(u<0||v<0||u>=p.width||v>=p.height)continue;
                    int g=(int)(v/cellH)*gw+(int)(u/cellW);
                    if(depth[g]==0||c<depth[g])depth[g]=(float)c;
                }
                // Model value at each cell centre (bilinear on the side×side output).
                float[] cell=new float[gw*gh];
                for(int j=0;j<gh;j++)for(int i=0;i<gw;i++){
                    float sx=(i+.5f)*side/gw-.5f,sy=(j+.5f)*side/gh-.5f;
                    int x0=Math.max(0,Math.min(side-2,(int)Math.floor(sx))),y0=Math.max(0,Math.min(side-2,(int)Math.floor(sy)));
                    float fx=Math.max(0,Math.min(1,sx-x0)),fy=Math.max(0,Math.min(1,sy-y0));
                    cell[j*gw+i]=(1-fy)*((1-fx)*out[y0*side+x0]+fx*out[y0*side+x0+1])+fy*((1-fx)*out[(y0+1)*side+x0]+fx*out[(y0+1)*side+x0+1]);
                }
                boolean[] filled=new boolean[gw*gh];
                float[] z=GuidedFill.fill(depth,cell,gw,gh,maxRange,filled);
                if(z==null)continue;
                result.fitted++;
                int count=0;for(boolean f:filled)if(f)count++;
                if(count==0)continue;
                result.cells+=count;
                // Surface frame at twice the cell grid, so SurfaceBuilder's 2-pixel stride lands on one sample per cell.
                SurfaceBuilder.Frame frame=new SurfaceBuilder.Frame();
                int w=2*gw,h=2*gh;
                frame.width=w;frame.height=h;
                frame.depthMm=new short[w*h];frame.modelMask=new byte[w*h];frame.pixelColor=new int[w*h];
                for(int y=0;y<h;y++)for(int x=0;x<w;x++){
                    int g=(y/2)*gw+x/2,at=y*w+x;
                    frame.depthMm[at]=(short)Math.min(65535,Math.round(z[g]*1000));
                    if(filled[g]){
                        frame.modelMask[at]=1;
                        int px=(int)Math.min(p.width-1,(x/2+.5)*cellW),py=(int)Math.min(p.height-1,(y/2+.5)*cellH);
                        frame.pixelColor[at]=image.getPixel(px,py);
                    }
                }
                double scale=(double)w/p.width;
                frame.fx=p.fx*scale;frame.fy=p.fy*scale;frame.cx=(p.cx+.5)*scale-.5;frame.cy=(p.cy+.5)*scale-.5;
                frame.pose=p.pose;frame.measured=0;
                result.frames.add(frame);
            }finally{image.recycle();}
        }
        result.totalMs=(System.nanoTime()-start)/1_000_000L;
        return result;
    }

    private void preparePreview(){
        System.arraycopy(colors,0,previewColors,0,colors.length);
        System.arraycopy(source,0,previewSource,0,source.length);
        for(int i=0;i<colors.length;i++){
            int neighbors=0, paint=0;
            float x=0,y=0,z=0;
            int r=0,g=0,b=0,min=255,max=0;
            for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++)for(int dz=-1;dz<=1;dz++){
                Integer j=cells.get(cell(i,dx,dy,dz));
                if(j==null||j==i)continue;
                float a=xyz[3*j]-xyz[3*i],c=xyz[3*j+1]-xyz[3*i+1],d=xyz[3*j+2]-xyz[3*i+2];
                if(a*a+c*c+d*d>voxel*voxel*1.96f)continue;
                x+=a;y+=c;z+=d;neighbors++;
                if(source[j]!=OBSERVED)continue;
                int red=Color.red(colors[j]),green=Color.green(colors[j]),blue=Color.blue(colors[j]);
                int lum=(red+2*green+blue)/4;
                min=Math.min(min,lum);max=Math.max(max,lum);r+=red;g+=green;b+=blue;paint++;
            }
            if(neighbors>=4){
                float factor=.18f/neighbors;
                smooth[3*i]+=factor*x;smooth[3*i+1]+=factor*y;smooth[3*i+2]+=factor*z;
            }
            if(source[i]==UNKNOWN&&paint>=4&&max-min<30){
                previewColors[i]=Color.rgb(r/paint,g/paint,b/paint);
                previewSource[i]=ESTIMATED;estimated++;
            }
        }
    }

    private long cell(int i,int dx,int dy,int dz){
        return key(xyz[3*i],dx)<<42|key(xyz[3*i+1],dy)<<21|key(xyz[3*i+2],dz);
    }
    private long key(float n,int delta){return ((long)Math.floor(n/voxel)+delta)&0x1fffffL;}
    private static ZipEntry entry(ZipFile zip,String name,int max) throws IOException {
        ZipEntry entry=zip.getEntry(name);
        if(entry==null||entry.getSize()<0||entry.getSize()>max)throw new IOException("Missing or oversized entry: "+name);
        return entry;
    }
    private static byte[] bytes(ZipFile zip,String name,int max) throws IOException{
        ZipEntry e=entry(zip,name,max);
        byte[] data=new byte[(int)e.getSize()];
        try(DataInputStream in=new DataInputStream(zip.getInputStream(e))){
            in.readFully(data);
            if(in.read()!=-1)throw new IOException("Invalid ZIP size.");
        }
        return data;
    }
    private static JSONObject json(ZipFile zip,String name)throws Exception{
        return new JSONObject(new String(bytes(zip,name,1<<20),StandardCharsets.UTF_8));
    }
}
