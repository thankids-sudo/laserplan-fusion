// Offline check of the app GuidedFill on hold-out cases exported from a survey (tools only, not in the APK).
package app.laserplan.survey;
import java.nio.*; import java.nio.file.*;
public class GuidedCheck {
  static float[] read(Path p) throws Exception { byte[] b=Files.readAllBytes(p); float[] f=new float[b.length/4]; ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(f); return f; }
  public static void main(String[] a) throws Exception {
    int GW=80,GH=60; double sumAi=0,sumLoc=0; int n=0,filledHidden=0,hiddenTotal=0,w5ai=0,w5loc=0,falseFill=0;
    for (int c=1;;c++){ Path k=Paths.get(a[0],String.format("%02d_known.f32",c)); if(!Files.exists(k))break;
      float[] known=read(k), truth=read(Paths.get(a[0],String.format("%02d_truth.f32",c))), model=read(Paths.get(a[0],String.format("%02d_model.f32",c)));
      boolean[] filled=new boolean[GW*GH]; float[] z=GuidedFill.fill(known,model,GW,GH,5f,filled);
      // local interpolation baseline = GuidedFill.diffuse on depth
      boolean[] kn=new boolean[GW*GH]; float[] loc=known.clone(); for(int i=0;i<kn.length;i++)kn[i]=known[i]>0; GuidedFill.diffuse(loc,kn,GW,GH,300);
      for(int i=0;i<GW*GH;i++){ boolean hidden=known[i]==0&&truth[i]>0; if(!hidden){ if(filled[i]&&truth[i]==0)falseFill++; continue; }
        hiddenTotal++; if(z==null||!filled[i])continue; filledHidden++;
        double eA=Math.abs(z[i]-truth[i])*100, eL=Math.abs(loc[i]-truth[i])*100; sumAi+=eA; sumLoc+=eL; n++; if(eA<5)w5ai++; if(eL<5)w5loc++; }
    }
    System.out.printf(java.util.Locale.US,"hidden cells %d, filled by GuidedFill %d (%.0f%%)%non those cells: AI MAE %.2f cm (<5 cm %.1f%%) vs local interpolation MAE %.2f cm (<5 cm %.1f%%)%nfilled cells with no measured truth (unverifiable, e.g. true gaps): %d%n",
      hiddenTotal,filledHidden,100.0*filledHidden/hiddenTotal,sumAi/n,100.0*w5ai/n,sumLoc/n,100.0*w5loc/n,falseFill);
  }
}
