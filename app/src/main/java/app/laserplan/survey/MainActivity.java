package app.laserplan.survey;

import android.app.*;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.*;
import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

public final class MainActivity extends Activity {
    private final ExecutorService jobs=Executors.newSingleThreadExecutor();
    private ScanView view;
    private TextView status,legend,surfaceLabel;
    private File selected;
    private int generation,mode,smoothing=2,fill=3;
    private Fusion current;
    private DepthModel depthModel;
    private boolean aiEnabled;
    private volatile Fusion.AiResult ai;
    private volatile Fusion aiFor;
    private volatile String aiReport="";
    private LinearLayout surfaceControls;

    @Override public void onCreate(Bundle state){
        super.onCreate(state);
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);
        LinearLayout actions=new LinearLayout(this);
        button(actions,"Scan",()->startActivityForResult(new Intent(this,ScanActivity.class),11));
        button(actions,"Open ZIP",()->picker(false));
        button(actions,"Saved",this::saved);
        button(actions,"Export",()->picker(true));
        root.addView(actions);
        Spinner modes=new Spinner(this);
        modes.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,
                new String[]{"Raw points","Photo points","Photo surface","Coverage"}));
        root.addView(modes);
        view=new ScanView(this);
        root.addView(view,new LinearLayout.LayoutParams(-1,0,1));
        LinearLayout controls=new LinearLayout(this);
        button(controls,"Fit",view::reset);
        button(controls,"Top",view::top);
        SeekBar size=new SeekBar(this);size.setMax(7);size.setProgress(3);
        size.setContentDescription("Point size");
        controls.addView(size,new LinearLayout.LayoutParams(0,-2,2));
        root.addView(controls);
        // Surface controls: smoothing (edge-preserving) and estimated fill (only holes bounded by measured depth).
        surfaceControls=new LinearLayout(this);surfaceControls.setOrientation(LinearLayout.VERTICAL);
        surfaceLabel=new TextView(this);surfaceLabel.setPadding(16,4,16,0);
        surfaceControls.addView(surfaceLabel);
        LinearLayout sliders=new LinearLayout(this);
        SeekBar smooth=slider(sliders,"Smoothing",smoothing);
        SeekBar estimate=slider(sliders,"Estimated fill",fill);
        surfaceControls.addView(sliders);
        // Optional AI depth fill (ai build only). Off by default; its output is always labelled an estimate.
        depthModel=DepthModels.open(this);
        if(depthModel!=null){
            CheckBox aiBox=new CheckBox(this);
            aiBox.setText("AI depth fill (estimate) · large enclosed holes only");
            aiBox.setChecked(false);
            aiBox.setOnCheckedChangeListener((box,on)->{aiEnabled=on;buildSurface();});
            surfaceControls.addView(aiBox);
        }
        surfaceControls.setVisibility(View.GONE);
        root.addView(surfaceControls);
        SeekBar.OnSeekBarChangeListener rebuild=new SeekBar.OnSeekBarChangeListener(){
            public void onProgressChanged(SeekBar bar,int progress,boolean user){
                if(bar==smooth)smoothing=progress;else fill=progress;
                labelSurface(null);
            }
            public void onStartTrackingTouch(SeekBar bar){}
            public void onStopTrackingTouch(SeekBar bar){buildSurface();}
        };
        smooth.setOnSeekBarChangeListener(rebuild);estimate.setOnSeekBarChangeListener(rebuild);
        labelSurface(null);
        status=new TextView(this);status.setPadding(16,8,16,8);
        status.setText("Start an AR scan or open a LaserPlan ZIP.");root.addView(status);
        legend=new TextView(this);legend.setPadding(16,0,16,12);root.addView(legend);
        modes.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener(){
            public void onNothingSelected(AdapterView<?> parent){}
            public void onItemSelected(AdapterView<?> parent,View v,int index,long id){
                mode=index;
                view.setMode(index);
                surfaceControls.setVisibility(index>=ScanView.SURFACE?View.VISIBLE:View.GONE);
                updateLegend();
            }
        });
        size.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
            public void onProgressChanged(SeekBar bar,int progress,boolean user){view.setSize(1+progress);}
            public void onStartTrackingTouch(SeekBar bar){}
            public void onStopTrackingTouch(SeekBar bar){}
        });
        setContentView(root);
    }

    private SeekBar slider(LinearLayout row,String name,int value){
        SeekBar bar=new SeekBar(this);bar.setMax(4);bar.setProgress(value);bar.setContentDescription(name);
        row.addView(bar,new LinearLayout.LayoutParams(0,-2,1));
        return bar;
    }
    private void updateLegend(){
        boolean approx=current!=null&&current.approximateAlignment;
        legend.setText(new String[]{"Measured points, original positions · metres",
                "Points coloured from visible photos"+(approx?" · APPROXIMATE alignment (older ZIP without photo calibration)":""),
                "Surface from measured depth, textured with the best photo per triangle · edges and gaps kept open",
                "Mint: photo sampled · Amber: estimated (fill or colour) · Violet: AI estimate · Grey: no photo"}[mode]
                +(mode>=ScanView.SURFACE&&aiEnabled?"\n"+aiReport:""));
    }
    private void labelSurface(SurfaceBuilder.Mesh mesh){
        String settings="Smoothing "+smoothing+" · Estimated fill "+fill;
        if(mesh==null){surfaceLabel.setText(settings);return;}
        float t=Math.max(1,mesh.triangles);
        surfaceLabel.setText(String.format(java.util.Locale.US,"%s · %,d triangles · photo %.0f%% · estimated %.0f%%%s · no photo %.0f%%",
                settings,mesh.triangles,100*mesh.photo/t,100*mesh.estimated/t,
                mesh.model>0?String.format(java.util.Locale.US," · AI estimate %.0f%%",100*mesh.model/t):"",100*mesh.uncovered/t));
    }
    private void buildSurface(){
        Fusion data=current;
        if(data==null)return;
        int request=generation,s=smoothing,f=fill;
        boolean useAi=aiEnabled&&depthModel!=null;
        surfaceLabel.setText(useAi&&aiFor!=data?"Running AI depth model on photos…":"Building surface…");
        jobs.execute(()->{
            List<SurfaceBuilder.Frame> extra=null;
            if(useAi){
                if(aiFor!=data){
                    ai=data.aiFrames(depthModel);aiFor=data;
                    aiReport=String.format(java.util.Locale.US,
                        "AI estimate: %s · %d photos, %d fitted · %,d cells filled · model %d ms total (%d ms/photo) · %d ms with fitting",
                        ai.model,ai.photos,ai.fitted,ai.cells,ai.inferenceMs,ai.photos==0?0:ai.inferenceMs/ai.photos,ai.totalMs);
                }
                extra=ai.frames;
            }
            SurfaceBuilder.Mesh mesh=data.surface(s,f,extra);
            runOnUiThread(()->{if(!isDestroyed()&&request==generation)surfaceLabel.setText("Texturing surface from photos…");});
            long t0=System.nanoTime();
            Fusion.Textured texture=data.texture(mesh);
            long textureMs=(System.nanoTime()-t0)/1_000_000L;
            runOnUiThread(()->{
                if(isDestroyed()||request!=generation||data!=current){if(texture!=null&&texture.atlas!=null)texture.atlas.recycle();return;}
                view.setMesh(mesh,texture);labelSurface(mesh);updateLegend();
                if(texture!=null)surfaceLabel.append(String.format(java.util.Locale.US,
                    "\nPhoto texture: %.0f%% of triangles from %d photos · %d ms",
                    100f*texture.map.textured/Math.max(1,mesh.triangles),texture.map.used.length,textureMs));
            });
        });
    }

    private void button(LinearLayout row,String title,Runnable click){
        Button b=new Button(this);b.setText(title);b.setOnClickListener(v->click.run());
        row.addView(b,new LinearLayout.LayoutParams(0,-2,1));
    }
    @Override protected void onResume(){super.onResume();view.onResume();}
    @Override protected void onPause(){view.onPause();super.onPause();}
    @Override protected void onDestroy(){
        jobs.execute(()->{if(depthModel!=null)depthModel.close();});
        jobs.shutdown();super.onDestroy();
    }

    private void picker(boolean export){
        if(export&&selected==null){status.setText("Choose a scan first.");return;}
        Intent intent=new Intent(export?Intent.ACTION_CREATE_DOCUMENT:Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType(export?"application/zip":"*/*");
        if(export)intent.putExtra(Intent.EXTRA_TITLE,"LaserPlan-survey.zip");
        startActivityForResult(intent,export?12:10);
    }
    private void saved(){
        String[] names=getFilesDir().list((dir,name)->name.matches("LaserPlan-survey-[0-9]+[.]zip"));
        if(names==null||names.length==0){status.setText("Finish an AR scan to save it here.");return;}
        Arrays.sort(names,java.util.Collections.reverseOrder());
        new AlertDialog.Builder(this).setTitle("Saved scans")
            .setItems(names,(dialog,index)->load(new File(getFilesDir(),names[index]),null)).show();
    }
    private void load(File file,android.net.Uri uri){
        int request=++generation;
        status.setText("Aligning photos with scan geometry…");
        jobs.execute(()->{
            File candidate=file;
            try{
                if(uri!=null){
                    candidate=File.createTempFile("import-",".zip",getCacheDir());
                    try(InputStream input=getContentResolver().openInputStream(uri);
                        OutputStream output=new FileOutputStream(candidate)){
                        if(input==null)throw new IOException("Cannot open ZIP.");
                        transfer(input,output);
                    }
                }
                Fusion data=Fusion.read(candidate);
                File archive=candidate;
                runOnUiThread(()->{
                    if(isDestroyed()||request!=generation){if(uri!=null)archive.delete();return;}
                    // Imported ZIPs remain available for export during this app session.
                    selected=archive;current=data;view.setScan(data);updateLegend();buildSurface();
                    status.setText(String.format(java.util.Locale.US,
                        "%,d points · %d photo frames · %,d photo-sampled · %,d estimated",
                        data.colors.length,data.photoFrames,data.observed,data.estimated));
                });
            }catch(Exception error){
                if(uri!=null&&candidate!=null)candidate.delete();
                report(request,"Could not open scan: "+error.getMessage());
            }
        });
    }
    private void report(int request,String text){
        runOnUiThread(()->{if(!isDestroyed()&&request==generation)status.setText(text);});
    }
    private static void transfer(InputStream in,OutputStream out)throws IOException{
        byte[] buffer=new byte[32768];long length=0;
        for(int n;(n=in.read(buffer))!=-1;){
            length+=n;
            if(length>256L*1024*1024)throw new IOException("ZIP exceeds 256 MB.");
            out.write(buffer,0,n);
        }
    }
    @Override protected void onActivityResult(int request,int result,Intent data){
        super.onActivityResult(request,result,data);
        if(result!=RESULT_OK||data==null)return;
        if(request==11){
            String name=data.getStringExtra("survey");
            if(name!=null&&name.matches("LaserPlan-survey-[0-9]+[.]zip"))load(new File(getFilesDir(),name),null);
        }
        if(request==10&&data.getData()!=null)load(null,data.getData());
        if(request==12&&selected!=null&&data.getData()!=null){
            File file=selected;int job=generation;
            jobs.execute(()->{
                try(InputStream in=new FileInputStream(file);
                    OutputStream out=getContentResolver().openOutputStream(data.getData(),"wt")){
                    if(out==null)throw new IOException("Destination unavailable.");
                    transfer(in,out);report(job,"Survey ZIP exported.");
                }catch(Exception e){report(job,"Export failed: "+e.getMessage());}
            });
        }
    }
}
