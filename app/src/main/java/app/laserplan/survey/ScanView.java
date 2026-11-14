package app.laserplan.survey;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.view.MotionEvent;
import java.nio.*;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

final class ScanView extends GLSurfaceView implements GLSurfaceView.Renderer {
    private Fusion scan;
    private SurfaceBuilder.Mesh mesh;
    private int program,buffer,count,mode,meshBuffer,meshCount;
    static final int RAW=0,PHOTO_POINTS=1,SURFACE=2,COVERAGE=3;
    private float yaw=.4f,pitch=.25f,zoom=1,panX,panY,pointSize=4,aspect=1;
    private float lastX,lastY,lastDistance;

    ScanView(Context context){
        super(context);
        setEGLContextClientVersion(2);
        setPreserveEGLContextOnPause(true);
        setRenderer(this);
        setRenderMode(RENDERMODE_WHEN_DIRTY);
    }

    void setScan(Fusion value){update(()->{scan=value;mesh=null;meshCount=0;fit();upload();});}
    void setMesh(SurfaceBuilder.Mesh value){update(()->{mesh=value;uploadMesh();});}
    void setMode(int value){update(()->{mode=value;upload();uploadMesh();});}
    void setSize(float value){update(()->pointSize=value);}
    void reset(){update(this::fit);}
    void top(){update(()->{fit();yaw=0;pitch=(float)Math.PI/2;});}
    private void update(Runnable r){queueEvent(r);requestRender();}
    private void fit(){yaw=.4f;pitch=.25f;zoom=1;panX=panY=0;}

    @Override public void onSurfaceCreated(GL10 unused,EGLConfig config){
        program=GLES20.glCreateProgram();
        attach(GLES20.GL_VERTEX_SHADER,
            "attribute vec3 p,c; uniform vec3 center; uniform vec4 orbit; uniform vec2 pan; uniform float scale,aspect,size; varying vec3 rgb;"
            +"void main(){vec3 v=(p-center)*scale; float x=orbit.x*v.x-orbit.y*v.z; float z=orbit.y*v.x+orbit.x*v.z;"
            +"gl_Position=vec4(x/aspect+pan.x,orbit.z*v.y-orbit.w*z+pan.y,-.05*(orbit.w*v.y+orbit.z*z),1.);"
            +"gl_PointSize=size;rgb=c;}");
        attach(GLES20.GL_FRAGMENT_SHADER,
            "precision mediump float; varying vec3 rgb; void main(){if(length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(rgb,1.);}");
        GLES20.glLinkProgram(program);
        int[] id=new int[1];
        GLES20.glGenBuffers(1,id,0);buffer=id[0];
        GLES20.glGenBuffers(1,id,0);meshBuffer=id[0];
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        GLES20.glClearColor(.045f,.075f,.105f,1);
        upload();
        uploadMesh();
    }
    private static int coverageColor(int source){
        return source==Fusion.OBSERVED?0xff72d9b2:source==Fusion.ESTIMATED?0xffffb05e:0xff778898;
    }
    /** Surface triangles: photo colours, or mint/amber/grey for photo/estimated/uncovered. */
    private void uploadMesh(){
        if(meshBuffer==0)return;
        if(mesh==null||mesh.triangles==0||(mode!=SURFACE&&mode!=COVERAGE)){meshCount=0;return;}
        meshCount=mesh.triangles*3;
        FloatBuffer out=ByteBuffer.allocateDirect(meshCount*24).order(ByteOrder.nativeOrder()).asFloatBuffer();
        for(int v=0;v<meshCount;v++){
            int source=mesh.source[v];
            int color=mode==COVERAGE?(source==SurfaceBuilder.PHOTO?0xff72d9b2:source==SurfaceBuilder.ESTIMATED?0xffffb05e:0xff778898)
                    :mesh.color[v];
            out.put(mesh.xyz,3*v,3);
            out.put(((color>>16)&255)/255f).put(((color>>8)&255)/255f).put((color&255)/255f);
        }
        out.flip();
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,meshBuffer);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER,meshCount*24,out,GLES20.GL_STATIC_DRAW);
    }
    private void attach(int kind,String code){
        int shader=GLES20.glCreateShader(kind);
        GLES20.glShaderSource(shader,code);GLES20.glCompileShader(shader);
        GLES20.glAttachShader(program,shader);
        GLES20.glDeleteShader(shader);
    }
    private void upload(){
        if(scan==null||buffer==0)return;
        float[] xyz=mode==2||mode==3?scan.smooth:scan.xyz;
        int[] colors=mode==1?scan.colors:scan.previewColors;
        count=xyz.length/3;
        FloatBuffer out=ByteBuffer.allocateDirect(count*24).order(ByteOrder.nativeOrder()).asFloatBuffer();
        for(int i=0;i<count;i++){
            int color=mode==0?Fusion.NEUTRAL:colors[i];
            if(mode==COVERAGE)color=coverageColor(scan.previewSource[i]);
            out.put(xyz,3*i,3);
            out.put(((color>>16)&255)/255f).put(((color>>8)&255)/255f).put((color&255)/255f);
        }
        out.flip();
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,buffer);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER,count*24,out,GLES20.GL_STATIC_DRAW);
    }
    @Override public void onSurfaceChanged(GL10 unused,int w,int h){
        aspect=(float)w/Math.max(1,h);GLES20.glViewport(0,0,w,h);
    }
    private int uniform(String name){return GLES20.glGetUniformLocation(program,name);}
    @Override public void onDrawFrame(GL10 unused){
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT|GLES20.GL_DEPTH_BUFFER_BIT);
        if(scan==null||count==0)return;
        GLES20.glUseProgram(program);
        boolean surface=meshCount>0&&(mode==SURFACE||mode==COVERAGE);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,surface?meshBuffer:buffer);
        int p=GLES20.glGetAttribLocation(program,"p"), c=GLES20.glGetAttribLocation(program,"c");
        if(p<0||c<0)return;
        GLES20.glEnableVertexAttribArray(p);GLES20.glEnableVertexAttribArray(c);
        GLES20.glVertexAttribPointer(p,3,GLES20.GL_FLOAT,false,24,0);
        GLES20.glVertexAttribPointer(c,3,GLES20.GL_FLOAT,false,24,12);
        GLES20.glUniform3fv(uniform("center"),1,scan.center,0);
        GLES20.glUniform4f(uniform("orbit"),(float)Math.cos(yaw),(float)Math.sin(yaw),
                (float)Math.cos(pitch),(float)Math.sin(pitch));
        GLES20.glUniform2f(uniform("pan"),panX,panY);
        GLES20.glUniform1f(uniform("scale"),1.25f*zoom*Math.min(1,aspect)/scan.span);
        GLES20.glUniform1f(uniform("aspect"),aspect);
        GLES20.glUniform1f(uniform("size"),pointSize*getResources().getDisplayMetrics().density);
        GLES20.glDrawArrays(surface?GLES20.GL_TRIANGLES:GLES20.GL_POINTS,0,surface?meshCount:count);
    }
    @Override public boolean onTouchEvent(MotionEvent e){
        int action=e.getActionMasked();
        if(action==MotionEvent.ACTION_CANCEL||action==MotionEvent.ACTION_UP)return true;
        int first=action==MotionEvent.ACTION_POINTER_UP&&e.getActionIndex()==0?1:0;
        boolean two=e.getPointerCount()>1&&action!=MotionEvent.ACTION_POINTER_UP;
        float x=e.getX(first),y=e.getY(first),distance=0;
        if(two){
            distance=(float)Math.hypot(x-e.getX(1),y-e.getY(1));
            x=(x+e.getX(1))/2;y=(y+e.getY(1))/2;
        }
        if(action==MotionEvent.ACTION_MOVE){
            float dx=x-lastX,dy=y-lastY,ratio=lastDistance>4?distance/lastDistance:1;
            update(()->{
                if(two){
                    zoom=Math.max(.25f,Math.min(8,zoom*ratio));
                    panX+=2*dx/Math.max(1,getWidth());panY-=2*dy/Math.max(1,getHeight());
                }else{
                    yaw+=dx*.008f;
                    pitch=Math.max(-1.57f,Math.min(1.57f,pitch-dy*.008f));
                }
            });
        }
        lastX=x;lastY=y;lastDistance=distance;
        return true;
    }
}
