// Offline only: runs the app TextureMapper on a mesh (tools/surface/SurfaceCheck); args: mesh views.txt out.bin [maxPhotos] [views_depth.txt]
package app.laserplan.survey;
import java.nio.*; import java.nio.file.*; import java.util.*;
public class TextureCheck { public static void main(String[] a) throws Exception {
  byte[] raw=Files.readAllBytes(Paths.get(a[0])); int T=raw.length/51; float[] xyz=new float[9*T];
  ByteBuffer.wrap(raw,0,36*T).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(xyz);
  List<TextureMapper.View> views=new ArrayList<>(); List<String> names=new ArrayList<>();
  for(String l:Files.readAllLines(Paths.get(a[1]))){String[] t=l.split(" ");TextureMapper.View v=new TextureMapper.View();names.add(t[0]);
    v.width=Integer.parseInt(t[1]);v.height=Integer.parseInt(t[2]);v.fx=Double.parseDouble(t[3]);v.fy=Double.parseDouble(t[4]);v.cx=Double.parseDouble(t[5]);v.cy=Double.parseDouble(t[6]);
    v.pose=new double[16];for(int i=0;i<16;i++)v.pose[i]=Double.parseDouble(t[7+i]);views.add(v);}
  // optional 5th arg: views_depth.txt (per photo: raw and smoothed depth checks, as the app uses)
  if(a.length>4){ List<String> dl=Files.readAllLines(Paths.get(a[4]));
    for(int vi=0;vi<views.size()&&vi<dl.size();vi++){ String[] t=dl.get(vi).split(" "); List<TextureMapper.DepthCheck> cs=new ArrayList<>();
      for(int p=0;p+25<=t.length;p+=25){ TextureMapper.DepthCheck d=new TextureMapper.DepthCheck(); d.width=Integer.parseInt(t[p+1]); d.height=Integer.parseInt(t[p+2]);
        d.fx=Double.parseDouble(t[p+3]);d.fy=Double.parseDouble(t[p+4]);d.cx=Double.parseDouble(t[p+5]);d.cy=Double.parseDouble(t[p+6]);
        byte[] db=Files.readAllBytes(Paths.get(t[p+7])); short[] mm=new short[d.width*d.height]; ByteBuffer.wrap(db).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(mm);
        byte[] cf=t[p+8].equals("-")?null:Files.readAllBytes(Paths.get(t[p+8])); int minc=t[p].equals("raw")?32:0;
        d.metres=new float[mm.length]; for(int i=0;i<mm.length;i++){int m=mm[i]&0xffff,c=cf==null?255:cf[i]&255; if(m>0&&c>=minc&&m<=5000)d.metres[i]=m*0.001f;}
        d.pose=new double[16]; for(int i=0;i<16;i++)d.pose[i]=Double.parseDouble(t[p+9+i]); cs.add(d);}
      views.get(vi).depth=cs.toArray(new TextureMapper.DepthCheck[0]); } }
  long t0=System.nanoTime(); TextureMapper.Result r=TextureMapper.map(xyz,T,views,a.length>3?Integer.parseInt(a[3]):48); long ms=(System.nanoTime()-t0)/1000000;
  ByteBuffer o=ByteBuffer.allocate(4*T+24*T).order(ByteOrder.LITTLE_ENDIAN); for(int p:r.photo)o.putInt(p); for(float f:r.uv)o.putFloat(f);
  Files.write(Paths.get(a[2]),o.array());
  System.out.printf("triangles %d, textured %.1f%%, photos used %d of %d, mapping %d ms (desktop JVM)%n",T,100.0*r.textured/T,r.used.length,views.size(),ms);
}}
