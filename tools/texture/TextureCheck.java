// Offline only: runs the app TextureMapper on a mesh from tools/surface/SurfaceCheck and writes the mapping.
package app.laserplan.survey;
import java.nio.*; import java.nio.file.*; import java.util.*;
public class TextureCheck { public static void main(String[] a) throws Exception {
  byte[] raw=Files.readAllBytes(Paths.get(a[0])); int T=raw.length/51; float[] xyz=new float[9*T];
  ByteBuffer.wrap(raw,0,36*T).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(xyz);
  List<TextureMapper.View> views=new ArrayList<>(); List<String> names=new ArrayList<>();
  for(String l:Files.readAllLines(Paths.get(a[1]))){String[] t=l.split(" ");TextureMapper.View v=new TextureMapper.View();names.add(t[0]);
    v.width=Integer.parseInt(t[1]);v.height=Integer.parseInt(t[2]);v.fx=Double.parseDouble(t[3]);v.fy=Double.parseDouble(t[4]);v.cx=Double.parseDouble(t[5]);v.cy=Double.parseDouble(t[6]);
    v.pose=new double[16];for(int i=0;i<16;i++)v.pose[i]=Double.parseDouble(t[7+i]);views.add(v);}
  long t0=System.nanoTime(); TextureMapper.Result r=TextureMapper.map(xyz,T,views,24); long ms=(System.nanoTime()-t0)/1000000;
  ByteBuffer o=ByteBuffer.allocate(4*T+24*T).order(ByteOrder.LITTLE_ENDIAN); for(int p:r.photo)o.putInt(p); for(float f:r.uv)o.putFloat(f);
  Files.write(Paths.get(a[2]),o.array());
  System.out.printf("triangles %d, textured %.1f%%, photos used %d of %d, mapping %d ms (desktop JVM)%n",T,100.0*r.textured/T,r.used.length,views.size(),ms);
}}
