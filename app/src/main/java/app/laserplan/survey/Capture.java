package app.laserplan.survey;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import com.google.ar.core.CameraIntrinsics;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;

final class Capture {
    static final long MAX_DELTA = 50_000_000L;
    static final int MAX_FRAMES = 90;
    static final int MAX_VOXELS = 600_000;
    static final float VOXEL = .02f;
    final int confidence, range;
    final long interval;
    private final File dir;
    private final Map<Long, Sum> voxels = new LinkedHashMap<>();
    int frames, rejected, depthNotReady, trackingLost, emptyDepth, smoothedHadData;
    int confMismatch, poseMatched, poseApprox, poseTooOld;
    long bestMatchErrorUs=-1;
    long depthLagMs=-1;
    volatile int points;
    volatile long pixelsKept, pixelsZero, pixelsFar, pixelsLowConfidence;
    private int photos;
    private boolean capReached;

    Capture(File cache, int confidence, int range, long interval) throws IOException {
        this.confidence = confidence;
        this.range = range;
        this.interval = interval;
        dir = java.nio.file.Files.createTempDirectory(cache.toPath(), "scan-").toFile();
        for (String name : new String[]{"frames","depth","confidence","rgb"}) {
            if (!new File(dir, name).mkdir()) throw new IOException("Could not create survey folder");
        }
    }

    /** Number of non-zero pixels in a 16-bit depth image (any strides). */
    static int nonZero16(Image image) {
        Image.Plane plane=image.getPlanes()[0];
        ByteBuffer src=plane.getBuffer().duplicate();
        src.rewind();
        byte[] all=new byte[src.remaining()];
        src.get(all);
        return CaptureRules.nonZero16(all,0,image.getWidth(),image.getHeight(),plane.getRowStride(),plane.getPixelStride());
    }

    static byte[] pack(Image image, int bytesPerPixel) {
        byte[] data = new byte[image.getWidth()*image.getHeight()*bytesPerPixel];
        copyPlane(image.getPlanes()[0], image.getWidth(), image.getHeight(), bytesPerPixel,
                data, 0, bytesPerPixel);
        return data;
    }

    private static void copyPlane(Image.Plane plane, int w, int h, int bytes,
                                  byte[] dst, int offset, int step) {
        ByteBuffer src = plane.getBuffer();
        int origin = 0; // absolute from the buffer start, as LaserPlan 13.2 does (duplicate + rewind)
        for (int y=0; y<h; y++) for (int x=0; x<w; x++) {
            int from = origin+y*plane.getRowStride()+x*plane.getPixelStride();
            for (int b=0; b<bytes; b++) dst[offset++] = src.get(from+b);
            offset += step-bytes;
        }
    }

    static byte[] jpeg(Image image) {
        if (image == null || image.getFormat() != ImageFormat.YUV_420_888) return null;
        int w=image.getWidth(), h=image.getHeight();
        if (w%2!=0 || h%2!=0) return null;
        byte[] nv21 = new byte[w*h*3/2];
        Image.Plane[] p = image.getPlanes();
        copyPlane(p[0],w,h,1,nv21,0,1);
        copyPlane(p[2],w/2,h/2,1,nv21,w*h,2);
        copyPlane(p[1],w/2,h/2,1,nv21,w*h+1,2);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        return new YuvImage(nv21,ImageFormat.NV21,w,h,null)
                .compressToJpeg(new Rect(0,0,w,h),85,out) ? out.toByteArray() : null;
    }

    static float[] calibration(CameraIntrinsics camera, int w, int h) {
        return calibration(camera.getFocalLength(), camera.getPrincipalPoint(), camera.getImageDimensions(), w, h);
    }

    /** Scales intrinsics measured at size[] to an image of w×h (same aspect and orientation). */
    static float[] calibration(float[] f, float[] p, int[] size, int w, int h) {
        float sx=(float)w/size[0], sy=(float)h/size[1];
        return new float[]{f[0]*sx,f[1]*sy,p[0]*sx,p[1]*sy};
    }

    private static JSONArray matrix(float[] values) throws Exception {
        JSONArray out=new JSONArray();
        for(float value:values) out.put(value);
        return out;
    }

    private JSONObject intrinsics(float[] k, int w, int h) throws Exception {
        return new JSONObject().put("fx",k[0]).put("fy",k[1]).put("cx",k[2]).put("cy",k[3])
                .put("width",w).put("height",h);
    }

    void writeFrame(long time, long depthTime, int w, int h, byte[] depth, byte[] conf,
                    float[] k, float[] pose, String poseSource, byte[] jpeg, int rw, int rh, float[] rgbK,
                    float[] rgbPose) throws Exception {
        int index=frames+1;
        String name=String.format(Locale.US,"%04d",index);
        String depthName="depth/"+name+".raw16le", confName="confidence/"+name+".raw8";
        int count=0;
        long zero=0, far=0, low=0;
        for (int v=0;v<h;v++) for (int u=0;u<w;u++) {
            int i=v*w+u, mm=(depth[i*2]&255)|((depth[i*2+1]&255)<<8), c=conf[i]&255;
            if (mm==0) {zero++; continue;}
            if (mm>range*1000) {far++; continue;}
            if (c<confidence) {low++; continue;}
            double d=mm*.001, x=(u-k[2])*d/k[0], y=(k[3]-v)*d/k[1], z=-d;
            double wx=pose[0]*x+pose[4]*y+pose[8]*z+pose[12];
            double wy=pose[1]*x+pose[5]*y+pose[9]*z+pose[13];
            double wz=pose[2]*x+pose[6]*y+pose[10]*z+pose[14];
            if (!Double.isFinite(wx)||!Double.isFinite(wy)||!Double.isFinite(wz)) continue;
            count++;
            long key=cell(wx)<<42 | cell(wy)<<21 | cell(wz);
            Sum sum=voxels.get(key);
            if(sum==null && voxels.size()==MAX_VOXELS) {capReached=true; continue;}
            if(sum==null) {sum=new Sum(); voxels.put(key,sum);}
            sum.x+=wx; sum.y+=wy; sum.z+=wz; sum.conf+=c/255.0; sum.n++;
        }
        pixelsKept+=count; pixelsZero+=zero; pixelsFar+=far; pixelsLowConfidence+=low;
        points=voxels.size();
        write(new File(dir,depthName),depth);
        write(new File(dir,confName),conf);
        JSONObject rgb=new JSONObject().put("status","unavailable");
        if(jpeg!=null) {
            String photo="rgb/"+name+".jpg";
            write(new File(dir,photo),jpeg);
            rgb.put("status","saved").put("file",photo).put("width",rw).put("height",rh)
               .put("intrinsics",intrinsics(rgbK,rw,rh));
            // The photo is from the live camera frame, which can be newer than the depth frame.
            if(rgbPose!=null) rgb.put("cameraTimestampNs",time).put("pose",new JSONObject().put("matrixColumnMajor",matrix(rgbPose)));
            photos++;
        }
        JSONObject frame=new JSONObject().put("index",index).put("valid",true)
                .put("cameraTimestampNs",time).put("depthTimestampNs",depthTime)
                .put("depthToCameraDeltaNs",depthTime-time)
                .put("depth",new JSONObject().put("file",depthName).put("width",w).put("height",h))
                .put("confidence",new JSONObject().put("file",confName).put("width",w).put("height",h))
                .put("intrinsics",new JSONObject().put("depth",intrinsics(k,w,h)))
                .put("pose",new JSONObject().put("matrixColumnMajor",matrix(pose)).put("source",poseSource))
                .put("pixels",new JSONObject().put("accepted",count).put("noDepth",zero).put("tooFar",far).put("lowConfidence",low)).put("rgb",rgb);
        write(new File(dir,"frames/"+name+".json"),frame.toString().getBytes(StandardCharsets.UTF_8));
        frames++;
    }

    private static long cell(double value) {return ((long)Math.floor(value/VOXEL))&0x1fffffL;}
    private static class Sum {double x,y,z,conf; int n;}
    private static void write(File f, byte[] bytes) throws IOException {
        try(OutputStream out=new FileOutputStream(f)){out.write(bytes);}
    }

    boolean hasPoints(){return !voxels.isEmpty();}

    File finish(File output) throws Exception {
        JSONObject manifest=new JSONObject().put("schema","laserplan.ar-survey").put("version",1)
                .put("app",new JSONObject().put("name","LaserPlan Fusion"))
                .put("frameCount",frames).put("validFrameCount",frames).put("pointCount",voxels.size())
                .put("rejectedTimingAttempts",rejected).put("photoCount",photos)
                .put("diagnostics",new JSONObject().put("depthNotReady",depthNotReady)
                      .put("emptyRawDepthImagesSkipped",emptyDepth)
                      .put("confidenceTimestampDiffersNotRejected",confMismatch).put("poseMatchedFrames",poseMatched)
                      .put("approximatePoseFrames",poseApprox).put("depthTooOld",poseTooOld)
                      .put("lastClosestPoseErrorUs",bestMatchErrorUs).put("smoothedDepthHadDataWhenRawEmpty",smoothedHadData).put("trackingLostFrames",trackingLost)
                      .put("lastDepthLagMs",depthLagMs).put("pixelsKept",pixelsKept).put("pixelsNoDepth",pixelsZero)
                      .put("pixelsTooFar",pixelsFar).put("pixelsLowConfidence",pixelsLowConfidence)
                      .put("poseMethod","depth uses the pose of the camera frame matching the depth timestamp (34 ms); otherwise the live pose when depth is at most 100 ms old, marked approximate in frames/NNNN.json pose.source"))
                .put("voxelCapReached",capReached)
                .put("coordinates",new JSONObject().put("world","ARCore world, metres, +Y up")
                      .put("camera","+X right, +Y up, -Z forward"))
                .put("settings",new JSONObject().put("minConfidence",confidence).put("maxRangeM",range)
                      .put("voxelM",VOXEL).put("maxDepthPoseDeltaNs",MAX_DELTA))
                .put("files",new JSONObject().put("cloud","cloud.xyzc"));
        write(new File(dir,"manifest.json"),manifest.toString(2).getBytes(StandardCharsets.UTF_8));
        try(PrintWriter out=new PrintWriter(new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(new File(dir,"cloud.xyzc")),StandardCharsets.UTF_8)))) {
            out.println("# x y z confidence; metres, ARCore +Y up");
            for(Sum s:voxels.values()) out.printf(Locale.US,"%.5f %.5f %.5f %.4f%n",
                    s.x/s.n,s.y/s.n,s.z/s.n,s.conf/s.n);
            if(out.checkError()) throw new IOException("Could not write point cloud");
        }
        File temp=new File(output.getPath()+".tmp");
        try {
            try(ZipOutputStream zip=new ZipOutputStream(new BufferedOutputStream(new FileOutputStream(temp)))) {
                archive(zip,dir,"");
            }
            if(!temp.renameTo(output)) throw new IOException("Could not save scan");
        } finally {temp.delete();}
        return output;
    }

    private static void archive(ZipOutputStream zip,File file,String prefix) throws IOException {
        File[] files=file.listFiles();
        if(files==null) throw new IOException("Cannot list scan files");
        for(File child:files){
            String path=prefix+child.getName();
            if(child.isDirectory()){archive(zip,child,path+"/");continue;}
            zip.putNextEntry(new ZipEntry(path));
            try(InputStream in=new FileInputStream(child)){
                byte[] buffer=new byte[32768]; int n;
                while((n=in.read(buffer))!=-1) zip.write(buffer,0,n);
            }
            zip.closeEntry();
        }
    }

    void discard(){remove(dir);}
    private static void remove(File file){
        File[] children=file.listFiles();
        if(children!=null) for(File child:children) remove(child);
        file.delete();
    }
}
