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
    private Fusion.Textured textured;
    private int program,buffer,count,mode,meshBuffer,meshCount;
    private int textureProgram,atlasTexture;
    private boolean meshTextured;
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
    void setMesh(SurfaceBuilder.Mesh value){setMesh(value,null);}
    /** Surface plus optional photo texture (atlas + per-vertex coordinates). */
    void setMesh(SurfaceBuilder.Mesh value,Fusion.Textured texture){
        update(()->{
            if(textured!=null&&textured!=texture&&textured.atlas!=null)textured.atlas.recycle();
            mesh=value;textured=texture;uploadAtlas();uploadMesh();
        });
    }
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
        // Textured surface: each fragment samples its photo from the atlas; untextured triangles keep vertex colour.
        // highp texture coordinates: mediump is too coarse for a 3840 px atlas.
        textureProgram=GLES20.glCreateProgram();
        attachTo(textureProgram,GLES20.GL_VERTEX_SHADER,
            "attribute vec3 p,c; attribute vec3 t; uniform vec3 center; uniform vec4 orbit; uniform vec2 pan; uniform float scale,aspect,size;"
            +"varying vec3 rgb; varying vec3 uvw;"
            +"void main(){vec3 v=(p-center)*scale; float x=orbit.x*v.x-orbit.y*v.z; float z=orbit.y*v.x+orbit.x*v.z;"
            +"gl_Position=vec4(x/aspect+pan.x,orbit.z*v.y-orbit.w*z+pan.y,-.05*(orbit.w*v.y+orbit.z*z),1.);rgb=c;uvw=t;}");
        attachTo(textureProgram,GLES20.GL_FRAGMENT_SHADER,
            "#ifdef GL_FRAGMENT_PRECISION_HIGH\nprecision highp float;\n#else\nprecision mediump float;\n#endif\n"
            +"uniform sampler2D atlas; varying vec3 rgb; varying vec3 uvw;"
            +"void main(){gl_FragColor=uvw.z>.5?vec4(texture2D(atlas,uvw.xy).rgb,1.):vec4(rgb,1.);}");
        GLES20.glLinkProgram(textureProgram);
        GLES20.glGenTextures(1,id,0);atlasTexture=id[0];
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        GLES20.glClearColor(.045f,.075f,.105f,1);
        upload();
        uploadAtlas();
        uploadMesh();
    }
    private void uploadAtlas(){
        if(atlasTexture==0||textured==null||textured.atlas==null||textured.atlas.isRecycled())return;
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,atlasTexture);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE);
        android.opengl.GLUtils.texImage2D(GLES20.GL_TEXTURE_2D,0,textured.atlas,0);
    }
    private static int coverageColor(int source){
        return source==Fusion.OBSERVED?0xff72d9b2:source==Fusion.ESTIMATED?0xffffb05e:0xff778898;
    }
    /** Surface triangles: photo colours, or mint/amber/grey for photo/estimated/uncovered. */
    private void uploadMesh(){
        if(meshBuffer==0)return;
        if(mesh==null||mesh.triangles==0||(mode!=SURFACE&&mode!=COVERAGE)){meshCount=0;return;}
        meshCount=mesh.triangles*3;
        // 9 floats per vertex: position, colour, atlas u, v and a textured flag
        meshTextured=mode==SURFACE&&textured!=null&&textured.atlas!=null;
        int[] slot=new int[0];
        float aw=1,ah=1;
        if(meshTextured){
            int views=0;for(int u:textured.map.used)views=Math.max(views,u+1);
            slot=new int[Math.max(views,1)];java.util.Arrays.fill(slot,-1);
            for(int i=0;i<textured.map.used.length;i++)slot[textured.map.used[i]]=i;
            aw=textured.atlas.getWidth();ah=textured.atlas.getHeight();
        }
        FloatBuffer out=ByteBuffer.allocateDirect(meshCount*36).order(ByteOrder.nativeOrder()).asFloatBuffer();
        for(int v=0;v<meshCount;v++){
            int source=mesh.source[v];
            int color=mode==COVERAGE?(source==SurfaceBuilder.PHOTO?0xff72d9b2:source==SurfaceBuilder.ESTIMATED?0xffffb05e
                    :source==SurfaceBuilder.MODEL?0xffb48cff:0xff778898)
                    :mesh.color[v];
            out.put(mesh.xyz,3*v,3);
            out.put(((color>>16)&255)/255f).put(((color>>8)&255)/255f).put((color&255)/255f);
            int photo=meshTextured?textured.map.photo[v/3]:-1;
            int s=photo>=0&&photo<slot.length?slot[photo]:-1;
            if(s>=0){
                float u=textured.map.uv[2*v],w=textured.map.uv[2*v+1];
                out.put(((s%textured.columns)*textured.slotW+u+.5f)/aw).put(((s/textured.columns)*textured.slotH+w+.5f)/ah).put(1f);
            }else out.put(0f).put(0f).put(0f);
        }
        out.flip();
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,meshBuffer);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER,meshCount*36,out,GLES20.GL_STATIC_DRAW);
    }
    private void attach(int kind,String code){attachTo(program,kind,code);}
    private static void attachTo(int target,int kind,String code){
        int shader=GLES20.glCreateShader(kind);
        GLES20.glShaderSource(shader,code);GLES20.glCompileShader(shader);
        GLES20.glAttachShader(target,shader);
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
    private static int uniform(int target,String name){return GLES20.glGetUniformLocation(target,name);}
    @Override public void onDrawFrame(GL10 unused){
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT|GLES20.GL_DEPTH_BUFFER_BIT);
        if(scan==null||count==0)return;
        boolean surface=meshCount>0&&(mode==SURFACE||mode==COVERAGE);
        int target=surface?textureProgram:program, stride=surface?36:24;
        GLES20.glUseProgram(target);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,surface?meshBuffer:buffer);
        int p=GLES20.glGetAttribLocation(target,"p"), c=GLES20.glGetAttribLocation(target,"c");
        if(p<0||c<0)return;
        GLES20.glEnableVertexAttribArray(p);GLES20.glEnableVertexAttribArray(c);
        GLES20.glVertexAttribPointer(p,3,GLES20.GL_FLOAT,false,stride,0);
        GLES20.glVertexAttribPointer(c,3,GLES20.GL_FLOAT,false,stride,12);
        int t=-1;
        if(surface){
            t=GLES20.glGetAttribLocation(target,"t");
            if(t>=0){GLES20.glEnableVertexAttribArray(t);GLES20.glVertexAttribPointer(t,3,GLES20.GL_FLOAT,false,stride,24);}
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,atlasTexture);
            GLES20.glUniform1i(uniform(target,"atlas"),0);
        }
        GLES20.glUniform3fv(uniform(target,"center"),1,scan.center,0);
        GLES20.glUniform4f(uniform(target,"orbit"),(float)Math.cos(yaw),(float)Math.sin(yaw),
                (float)Math.cos(pitch),(float)Math.sin(pitch));
        GLES20.glUniform2f(uniform(target,"pan"),panX,panY);
        GLES20.glUniform1f(uniform(target,"scale"),1.25f*zoom*Math.min(1,aspect)/scan.span);
        GLES20.glUniform1f(uniform(target,"aspect"),aspect);
        GLES20.glUniform1f(uniform(target,"size"),pointSize*getResources().getDisplayMetrics().density);
        GLES20.glDrawArrays(surface?GLES20.GL_TRIANGLES:GLES20.GL_POINTS,0,surface?meshCount:count);
        if(t>=0)GLES20.glDisableVertexAttribArray(t);
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
