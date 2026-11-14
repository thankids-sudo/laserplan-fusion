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
            float[][] votes=new float[4][size/3];
            int n=0;
            for(Enumeration<? extends ZipEntry> entries=zip.entries();entries.hasMoreElements();){
                String name=entries.nextElement().getName();
                if(!name.matches("frames/[0-9]+[.]json"))continue;
                if(++n>300)throw new IOException("Too many scan frames.");
                JSONObject frame=json(zip,name);
                if(!frame.optBoolean("valid")||Math.abs(frame.optDouble("depthToCameraDeltaNs",Double.POSITIVE_INFINITY))>
                    Math.min(50_000_000,settings.optDouble("maxDepthPoseDeltaNs",50_000_000)))continue;
                JSONObject rgb=frame.optJSONObject("rgb");
                if(rgb==null||!rgb.has("file")||zip.getEntry(rgb.optString("file"))==null)continue;
                if(result.project(zip,frame,rgb,settings.optInt("minConfidence",128),votes))result.photoFrames++;
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

    private boolean project(ZipFile zip,JSONObject frame,JSONObject rgb,int confidence,float[][] votes) throws Exception {
        JSONObject d=frame.getJSONObject("intrinsics").getJSONObject("depth");
        int w=d.getInt("width"),h=d.getInt("height");
        if(w<2||h<2||(long)w*h>500000)throw new IOException("Invalid depth dimensions.");
        byte[] depth=bytes(zip,frame.getJSONObject("depth").getString("file"),2*w*h);
        byte[] conf=bytes(zip,frame.getJSONObject("confidence").getString("file"),w*h);
        if(depth.length!=2*w*h||conf.length!=w*h)throw new IOException("Truncated depth frame.");
        JSONArray matrix=frame.getJSONObject("pose").getJSONArray("matrixColumnMajor");
        double[] m=new double[16];
        for(int i=0;i<16;i++)m[i]=matrix.getDouble(i);
        double fx=d.getDouble("fx"),fy=d.getDouble("fy"),cx=d.getDouble("cx"),cy=d.getDouble("cy");
        if(fx<=0||fy<=0)throw new IOException("Invalid camera focal length.");
        byte[] photo=bytes(zip,rgb.getString("file"),12<<20);
        BitmapFactory.Options bounds=new BitmapFactory.Options();
        bounds.inJustDecodeBounds=true;
        BitmapFactory.decodeByteArray(photo,0,photo.length,bounds);
        int rw=bounds.outWidth,rh=bounds.outHeight;
        if(rw<1||rh<1||(long)rw*rh>12_000_000||rw!=rgb.optInt("width")||rh!=rgb.optInt("height"))return false;
        Bitmap image=BitmapFactory.decodeByteArray(photo,0,photo.length);
        if(image==null)return false;
        try{
            int[] pixels=new int[rw*rh];
            image.getPixels(pixels,0,rw,0,0,rw,rh);
            JSONObject k=rgb.optJSONObject("intrinsics");
            // Older LaserPlan exports lack RGB intrinsics. Use a centered crop preview.
            double crop=Math.max((double)rw/w,(double)rh/h);
            double rfx=k==null?fx*crop:k.getDouble("fx"),rfy=k==null?fy*crop:k.getDouble("fy");
            double rcx=k==null?(cx+.5-w*.5)*crop+rw*.5-.5:k.getDouble("cx");
            double rcy=k==null?(cy+.5-h*.5)*crop+rh*.5-.5:k.getDouble("cy");
            boolean contributed=false;
            for(int i=0;i<colors.length;i++){
                double x=xyz[3*i]-m[12],y=xyz[3*i+1]-m[13],z=xyz[3*i+2]-m[14];
                double a=m[0]*x+m[1]*y+m[2]*z,b=-(m[4]*x+m[5]*y+m[6]*z),c=-(m[8]*x+m[9]*y+m[10]*z);
                if(c<=.1)continue;
                int u=(int)Math.round(fx*a/c+cx),v=(int)Math.round(fy*b/c+cy);
                if(u<0||v<0||u>=w||v>=h)continue;
                int at=v*w+u, quality=conf[at]&255;
                double measured=((depth[2*at]&255)|((depth[2*at+1]&255)<<8))*.001;
                double tolerance=Math.max(.028,.018*c), error=Math.abs(c-measured);
                if(quality<confidence||measured==0||error>=tolerance)continue;
                int px=(int)Math.round(rfx*a/c+rcx),py=(int)Math.round(rfy*b/c+rcy);
                if(px<0||py<0||px>=rw||py>=rh)continue;
                int color=pixels[py*rw+px];
                float weight=(float)(quality/255.0*(1-error/tolerance)/(1+c*c*.25));
                votes[0][i]+=weight*Color.red(color);
                votes[1][i]+=weight*Color.green(color);
                votes[2][i]+=weight*Color.blue(color);
                votes[3][i]+=weight;
                contributed=true;
            }
            return contributed;
        }finally{image.recycle();}
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
