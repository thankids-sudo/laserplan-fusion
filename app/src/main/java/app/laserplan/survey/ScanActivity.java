package app.laserplan.survey;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.Image;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.widget.*;
import com.google.ar.core.*;
import com.google.ar.core.exceptions.NotYetAvailableException;
import java.io.File;
import java.nio.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public final class ScanActivity extends Activity implements GLSurfaceView.Renderer {
    private final ExecutorService writer=Executors.newSingleThreadExecutor();
    private final FloatBuffer quad=buffer(-1,-1, 1,-1, -1,1, 1,1);
    private final FloatBuffer uv=buffer(0,0,0,0,0,0,0,0);
    private GLSurfaceView surface;
    private TextView status;
    private Button start,pause,finish;
    private Spinner confidence,range,interval;
    private volatile Session session;
    private int texture,program,width,height,rotation;
    private boolean installRequested;
    private volatile boolean recording,busy,saving;
    private volatile Capture capture;
    private long lastAccepted,lastRaw,lastStatus;
    private volatile int accepted,timingRejected,depthNotReady,trackingLost,emptyDepth,smoothedHadData;
    private volatile int confMismatch,poseMatched,poseApprox,poseTooOld;
    private volatile long bestMatchErrorUs=-1;
    private static final long APPROX_POSE_NS=100_000_000L;
    private volatile long depthLagMs=-1;
    // Recent camera poses by frame timestamp. ARCore's raw depth image usually belongs to an earlier
    // camera frame, so its points must use that frame's pose, not the live one (same approach as LaserPlan 13.2).
    private static final int POSES=120;
    // One camera-frame interval at 30 fps: the depth timestamp names one remembered frame.
    private static final long POSE_MATCH_NS=34_000_000L;
    private final long[] poseTime=new long[POSES],poseAndroidTime=new long[POSES];
    private final float[][] poseMatrix=new float[POSES][16],poseIntrinsics=new float[POSES][6];
    private int poseNext;

    @Override public void onCreate(Bundle state){
        super.onCreate(state);
        LinearLayout layout=new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        status=new TextView(this);
        status.setPadding(16,12,16,12);
        status.setText(version()+" · Starting AR camera…");
        // Long-press shares the full capture log, so one test on the phone returns all the data.
        status.setOnLongClickListener(v->{shareLog();return true;});
        layout.addView(status);
        surface=new GLSurfaceView(this);
        surface.setEGLContextClientVersion(2);
        surface.setRenderer(this);
        surface.onPause();
        layout.addView(surface,new LinearLayout.LayoutParams(-1,0,1));
        confidence=selector(layout,"Min. confidence","96","128","192"); confidence.setSelection(1);
        range=selector(layout,"Max. depth","3 m","5 m","8 m"); range.setSelection(1);
        interval=selector(layout,"Frame interval","300 ms","600 ms","1000 ms"); interval.setSelection(1);
        LinearLayout buttons=new LinearLayout(this);
        start=button(buttons,"Start");
        pause=button(buttons,"Pause");
        finish=button(buttons,"Finish & view");
        start.setEnabled(false);pause.setEnabled(false);finish.setEnabled(false);
        layout.addView(buttons);
        setContentView(layout);
        start.setOnClickListener(v->begin());
        pause.setOnClickListener(v->{
            recording=!recording;
            pause.setText(recording?"Pause":"Resume");
            if(!recording) status.setText("Paused · "+accepted+" frames captured");
        });
        finish.setOnClickListener(v->complete());
    }

    private Spinner selector(LinearLayout parent,String label,String... choices){
        LinearLayout row=new LinearLayout(this);
        row.setGravity(android.view.Gravity.CENTER_VERTICAL);
        TextView title=new TextView(this);
        title.setText(label);
        title.setPadding(16,0,8,0);
        row.addView(title,new LinearLayout.LayoutParams(0,-2,1));
        Spinner picker=new Spinner(this);
        ArrayAdapter<String> adapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,choices);
        picker.setAdapter(adapter);
        row.addView(picker);
        parent.addView(row);
        return picker;
    }
    private Button button(LinearLayout row,String title){
        Button b=new Button(this);b.setText(title);
        row.addView(b,new LinearLayout.LayoutParams(0,-2,1));
        return b;
    }

    @Override protected void onResume(){super.onResume();openSession();}
    private void openSession(){
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){
            requestPermissions(new String[]{Manifest.permission.CAMERA},1);return;
        }
        try{
            if(session==null){
                if(ArCoreApk.getInstance().requestInstall(this,!installRequested)==ArCoreApk.InstallStatus.INSTALL_REQUESTED){
                    installRequested=true;return;
                }
                session=new Session(this);
                if(!session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)){
                    session.close();session=null;
                    status.setText("ARCore Depth is unavailable on this device.");return;
                }
                Config config=new Config(session);
                config.setDepthMode(Config.DepthMode.AUTOMATIC);
                // Fixed focus (ARCore's default), as in LaserPlan 13.2: autofocus keeps changing the lens
                // and the intrinsics, which hurts depth-from-motion.
                config.setFocusMode(Config.FocusMode.FIXED);
                session.configure(config);
            }
            session.resume();
            surface.onResume();
            start.setEnabled(capture==null);
            status.setText(capture==null?"Move slowly to establish tracking.":"Paused · "+accepted+" frames captured");
        }catch(Exception e){status.setText("Could not start AR camera: "+e.getMessage());}
    }
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] results){
        super.onRequestPermissionsResult(request,permissions,results);
        if(request!=1)return;
        if(results.length>0&&results[0]==PackageManager.PERMISSION_GRANTED){
            if(session==null)openSession();
        }else status.setText("Camera permission is required for scanning.");
    }
    @Override protected void onPause(){
        recording=false;
        if(capture!=null&&!saving)pause.setText("Resume");
        surface.onPause();
        if(session!=null)session.pause();
        super.onPause();
    }
    @Override protected void onDestroy(){
        if(session!=null) session.close();
        writer.execute(()->{if(capture!=null)capture.discard();});
        writer.shutdown();
        super.onDestroy();
    }
    private void begin(){
        try {
            capture=new Capture(getCacheDir(),new int[]{96,128,192}[confidence.getSelectedItemPosition()],
                new int[]{3,5,8}[range.getSelectedItemPosition()],
                new long[]{300,600,1000}[interval.getSelectedItemPosition()]*1_000_000L);
            java.util.Arrays.fill(poseTime,0);java.util.Arrays.fill(poseAndroidTime,0);
            timingRejected=0;depthNotReady=0;trackingLost=0;emptyDepth=0;smoothedHadData=0;
            confMismatch=0;poseMatched=0;poseApprox=0;poseTooOld=0;bestMatchErrorUs=-1;depthLagMs=-1;lastRaw=0;lastAccepted=0;
            for(android.view.View v:new android.view.View[]{confidence,range,interval,start})v.setEnabled(false);
            pause.setText("Pause");pause.setEnabled(true);finish.setEnabled(true);
            recording=true;
        }catch(Exception e){status.setText(e.getMessage());}
    }
    private void complete(){
        recording=false;
        saving=true;
        pause.setEnabled(false);finish.setEnabled(false);
        status.setText("Preparing survey…");
        // Queue behind GL acquisition and any outstanding frame write.
        surface.queueEvent(()->writer.execute(()->{
            try{
                capture.rejected=timingRejected;
                capture.depthNotReady=depthNotReady;
                capture.trackingLost=trackingLost;
                capture.depthLagMs=depthLagMs;
                capture.emptyDepth=emptyDepth;
                capture.smoothedHadData=smoothedHadData;
                capture.confMismatch=confMismatch;capture.poseMatched=poseMatched;
                capture.poseApprox=poseApprox;capture.poseTooOld=poseTooOld;capture.bestMatchErrorUs=bestMatchErrorUs;
                if(!capture.hasPoints())throw new IllegalStateException(version()+" · No usable points yet. "+diagnosis()
                        +" Long-press this message to share the capture log.");
                File zip=capture.finish(new File(getFilesDir(),"LaserPlan-survey-"+System.currentTimeMillis()+".zip"));
                runOnUiThread(()->{
                    if(isDestroyed())return;
                    setResult(RESULT_OK,new Intent().putExtra("survey",zip.getName()));
                    finish();
                });
            }catch(Exception e){
                runOnUiThread(()->{
                    saving=false;pause.setText("Resume");pause.setEnabled(true);
                    finish.setEnabled(true);status.setText(e.getMessage());
                });
            }
        }));
    }
    @Override public void onSurfaceCreated(GL10 unused,EGLConfig config){
        int[] id=new int[1];
        GLES20.glGenTextures(1,id,0);texture=id[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,texture);
        for(int p:new int[]{GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_TEXTURE_MAG_FILTER})
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,p,GLES20.GL_LINEAR);
        for(int p:new int[]{GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_TEXTURE_WRAP_T})
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,p,GLES20.GL_CLAMP_TO_EDGE);
        program=GLES20.glCreateProgram();
        GLES20.glAttachShader(program,shader(GLES20.GL_VERTEX_SHADER,
            "attribute vec2 pos, tex; varying vec2 uv; void main(){uv=tex;gl_Position=vec4(pos,0.,1.);}"));
        GLES20.glAttachShader(program,shader(GLES20.GL_FRAGMENT_SHADER,
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float; varying vec2 uv; uniform samplerExternalOES camera; void main(){gl_FragColor=texture2D(camera,uv);}"));
        GLES20.glLinkProgram(program);
    }
    private static int shader(int type,String source){
        int id=GLES20.glCreateShader(type);
        GLES20.glShaderSource(id,source);GLES20.glCompileShader(id);
        return id;
    }
    private static FloatBuffer buffer(float... data){
        FloatBuffer b=ByteBuffer.allocateDirect(data.length*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        b.put(data);b.position(0);return b;
    }
    @Override public void onSurfaceChanged(GL10 unused,int w,int h){
        width=w;height=h;rotation=-1;
        GLES20.glViewport(0,0,w,h);
    }
    @Override public void onDrawFrame(GL10 unused){
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        if(session==null||width==0)return;
        try{
            int current=getWindowManager().getDefaultDisplay().getRotation();
            if(current!=rotation){session.setDisplayGeometry(current,width,height);rotation=current;}
            session.setCameraTextureName(texture);
            Frame frame=session.update();
            if(frame.getTimestamp()==0)return;
            quad.position(0);uv.position(0);
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,quad,
                    Coordinates2d.TEXTURE_NORMALIZED,uv);
            drawCamera();
            boolean tracked=frame.getCamera().getTrackingState()==TrackingState.TRACKING;
            if(recording){
                if(tracked)remember(frame);else trackingLost++;
                if(!busy)acquire(frame);
            }
            long now=android.os.SystemClock.uptimeMillis();
            if(!saving&&now-lastStatus>350&&(recording||capture==null)){
                lastStatus=now;
                Capture c=capture;
                int points=c==null?0:c.points;
                runOnUiThread(()->{
                    if(saving||(!recording&&capture!=null))return;
                    status.setText((tracked?"Tracking":"Move slowly to recover tracking")
                            +(c==null?"":String.format(java.util.Locale.US," · %d frames · %,d points · depth lag %s · %d skips",
                                accepted,points,depthLagMs<0?"–":depthLagMs+" ms",timingRejected+emptyDepth)));
                });
            }
        }catch(Exception e){recording=false;runOnUiThread(()->{
            status.setText("AR camera error: "+e.getMessage());
            if(capture!=null&&!saving)pause.setText("Resume");
        });}
    }
    private void drawCamera(){
        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,texture);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program,"camera"),0);
        int p=GLES20.glGetAttribLocation(program,"pos"), t=GLES20.glGetAttribLocation(program,"tex");
        quad.position(0);uv.position(0);
        GLES20.glVertexAttribPointer(p,2,GLES20.GL_FLOAT,false,0,quad);
        GLES20.glVertexAttribPointer(t,2,GLES20.GL_FLOAT,false,0,uv);
        GLES20.glEnableVertexAttribArray(p);GLES20.glEnableVertexAttribArray(t);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP,0,4);
    }
    private String diagnosis(){
        Capture c=capture;
        return String.format(java.util.Locale.US,
            "Frames saved %d (pose matched %d, approximate %d) · empty depth %d (smoothed had data %d) · confidence time differs %d (not rejected) · depth too old %d · closest pose %s · depth not ready %d · tracking lost %d · depth lag %s · pixels kept %,d, no depth %,d, too far %,d, low confidence %,d.",
            accepted,poseMatched,poseApprox,emptyDepth,smoothedHadData,confMismatch,poseTooOld,
            bestMatchErrorUs<0?"none":String.format(java.util.Locale.US,"%.1f ms",bestMatchErrorUs/1000.0),
            depthNotReady,trackingLost,depthLagMs<0?"unknown":depthLagMs+" ms",
            c==null?0:c.pixelsKept,c==null?0:c.pixelsZero,c==null?0:c.pixelsFar,c==null?0:c.pixelsLowConfidence);
    }

    /** GL thread: remembers this frame's pose and texture intrinsics for depth that arrives later. */
    private void remember(Frame frame){
        Camera camera=frame.getCamera();
        int slot=poseNext;poseNext=(poseNext+1)%POSES;
        poseTime[slot]=frame.getTimestamp();
        poseAndroidTime[slot]=frame.getAndroidCameraTimestamp();
        camera.getPose().toMatrix(poseMatrix[slot],0);
        CameraIntrinsics t=camera.getTextureIntrinsics();
        float[] f=t.getFocalLength(),p=t.getPrincipalPoint();int[] d=t.getImageDimensions();
        float[] k=poseIntrinsics[slot];
        k[0]=f[0];k[1]=f[1];k[2]=p[0];k[3]=p[1];k[4]=d[0];k[5]=d[1];
    }

    /** Slot of the remembered frame whose timestamp matches [time] within 8 ms, or -1. */
    private int findPose(long time){
        long[] error=new long[1];
        int slot=CaptureRules.findPose(poseTime,poseAndroidTime,time,POSE_MATCH_NS,error);
        bestMatchErrorUs=error[0]<0?-1:error[0]/1000L;
        return slot;
    }

    private final java.util.ArrayDeque<String> attempts=new java.util.ArrayDeque<>();

    private void log(String line){
        synchronized(attempts){
            attempts.addLast(line);
            while(attempts.size()>400)attempts.removeFirst();
        }
    }

    private String version(){
        try{
            android.content.pm.PackageInfo info=getPackageManager().getPackageInfo(getPackageName(),0);
            return "Fusion "+info.versionName+" ("+info.versionCode+")";
        }catch(Exception e){return "Fusion";}
    }

    private void shareLog(){
        StringBuilder out=new StringBuilder();
        out.append(version()).append('\n')
           .append(android.os.Build.MANUFACTURER).append(' ').append(android.os.Build.MODEL)
           .append(" · Android ").append(android.os.Build.VERSION.RELEASE).append('\n')
           .append(capture==null?"No capture yet":diagnosis()).append("\n\n")
           .append("frameMs depthMs confDeltaMs lagMs size nonZero closestPoseMs outcome\n");
        synchronized(attempts){for(String line:attempts)out.append(line).append('\n');}
        android.content.Intent send=new android.content.Intent(android.content.Intent.ACTION_SEND)
            .setType("text/plain")
            .putExtra(android.content.Intent.EXTRA_SUBJECT,"LaserPlan Fusion capture log")
            .putExtra(android.content.Intent.EXTRA_TEXT,out.toString());
        startActivity(android.content.Intent.createChooser(send,"Share capture log"));
    }

    private void acquire(Frame frame) throws Exception{
        Capture target=capture;
        long time=frame.getTimestamp();
        Camera camera=frame.getCamera();
        if(camera.getTrackingState()!=TrackingState.TRACKING||time-lastAccepted<target.interval)return;
        try(Image depth=frame.acquireRawDepthImage16Bits();Image conf=frame.acquireRawDepthConfidenceImage()){
            long rawTime=depth.getTimestamp();
            if(rawTime==lastRaw)return;
            lastRaw=rawTime;
            depthLagMs=Math.abs(time-rawTime)/1_000_000L;
            // LaserPlan 13.2 parity: the confidence image is used whenever its size matches; its timestamp
            // is logged but never rejects depth (0.3–0.5 rejected on it).
            long confDeltaMs=(conf.getTimestamp()-rawTime)/1_000_000L;
            if(confDeltaMs!=0)confMismatch++;
            int w=depth.getWidth(),h=depth.getHeight();
            String head=String.format(java.util.Locale.US,"%d %d %d %d %dx%d",
                time/1_000_000L,rawTime/1_000_000L,confDeltaMs,depthLagMs,w,h);
            if(w!=conf.getWidth()||h!=conf.getHeight()){log(head+" - - confidence-size-"+conf.getWidth()+"x"+conf.getHeight());return;}
            // ARCore often hands back raw depth with no measured pixels. Skip it at once and keep trying
            // on the next frames instead of saving an empty frame and waiting a whole interval.
            int nonZero=Capture.nonZero16(depth);
            if(CaptureRules.isEmpty(nonZero,w,h)){
                emptyDepth++;
                log(head+" "+nonZero+" - empty");
                try(Image smooth=frame.acquireDepthImage16Bits()){
                    if(Capture.nonZero16(smooth)>0)smoothedHadData++;
                }catch(Exception ignored){}
                return;
            }
            float[] pose=new float[16],k;
            String poseSource;
            int slot=findPose(rawTime);
            String match=bestMatchErrorUs<0?"-":String.format(java.util.Locale.US,"%.1f",bestMatchErrorUs/1000.0);
            if(slot>=0){
                System.arraycopy(poseMatrix[slot],0,pose,0,16);
                float[] m=poseIntrinsics[slot];
                k=Capture.calibration(new float[]{m[0],m[1]},new float[]{m[2],m[3]},new int[]{(int)m[4],(int)m[5]},w,h);
                poseSource="matched-frame";
                poseMatched++;
            }else if(Math.abs(time-rawTime)<=APPROX_POSE_NS){
                // Last resort: the live pose. Off by the camera movement during the lag (about 2 cm at a slow walk).
                camera.getPose().toMatrix(pose,0);
                k=Capture.calibration(camera.getTextureIntrinsics(),w,h);
                poseSource=Math.abs(time-rawTime)<=Capture.MAX_DELTA?"current-frame":"approximate-live-pose";
                poseApprox++;
            }else{timingRejected++;poseTooOld++;log(head+" "+nonZero+" "+match+" too-old");return;}
            log(head+" "+nonZero+" "+match+" saved:"+poseSource);
            byte[] d=Capture.pack(depth,2),c=Capture.pack(conf,1);
            byte[] color=null;int rw=0,rh=0;float[] rgbK=null,rgbPose=null;
            if(accepted%3==0){
                try(Image rgb=frame.acquireCameraImage()){
                    rw=rgb.getWidth();rh=rgb.getHeight();
                    rgbK=Capture.calibration(camera.getImageIntrinsics(),rw,rh);
                    color=Capture.jpeg(rgb);
                    rgbPose=new float[16];camera.getPose().toMatrix(rgbPose,0);
                }catch(NotYetAvailableException ignored){}
            }
            final byte[] jpeg=color; final int photoW=rw,photoH=rh; final float[] photoK=rgbK,photoPose=rgbPose;
            final String source=poseSource;
            lastAccepted=time;
            busy=true;
            writer.execute(()->{
                try{
                    target.writeFrame(time,rawTime,w,h,d,c,k,pose,source,jpeg,photoW,photoH,photoK,photoPose);
                    accepted=target.frames;
                    if(accepted>=Capture.MAX_FRAMES){
                        recording=false;
                        runOnUiThread(()->{
                            pause.setEnabled(false);
                            status.setText("90 frames captured. Tap Finish & view.");
                        });
                    }
                }catch(Exception e){
                    recording=false;
                    runOnUiThread(()->{
                        pause.setEnabled(false);status.setText("Capture stopped: "+e.getMessage());
                    });
                }finally{busy=false;}
            });
        }catch(NotYetAvailableException ignored){depthNotReady++;}
    }
}
