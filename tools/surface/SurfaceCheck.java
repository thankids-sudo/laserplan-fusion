package app.laserplan.survey;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/** Offline harness: runs app's SurfaceBuilder on a prepared survey and writes mesh.bin + stats. */
/** Offline only (tools/): compiled with javac next to SurfaceBuilder.java, never packaged in the APK. */
public class SurfaceCheck {
    public static void main(String[] a) throws Exception {
        String dir = a[0]; int smoothing = Integer.parseInt(a[1]), fill = Integer.parseInt(a[2]);
        int minConf = Integer.parseInt(a[3]);
        List<SurfaceBuilder.Frame> frames = new ArrayList<>();
        for (String line : Files.readAllLines(Paths.get(dir, "frames.txt"))) {
            String[] t = line.trim().split(" ");
            var f = new SurfaceBuilder.Frame();
            f.width = Integer.parseInt(t[0]); f.height = Integer.parseInt(t[1]);
            f.fx = Double.parseDouble(t[2]); f.fy = Double.parseDouble(t[3]); f.cx = Double.parseDouble(t[4]); f.cy = Double.parseDouble(t[5]);
            byte[] d = Files.readAllBytes(Paths.get(t[6]));
            f.depthMm = new short[f.width * f.height];
            ByteBuffer.wrap(d).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(f.depthMm);
            f.confidence = t[7].equals("-") ? null : Files.readAllBytes(Paths.get(t[7]));
            f.measured = Integer.parseInt(t[8]);
            f.pose = new double[16]; for (int i = 0; i < 16; i++) f.pose[i] = Double.parseDouble(t[9 + i]);
            frames.add(f);
        }
        byte[] raw = Files.readAllBytes(Paths.get(dir, "colors.bin"));
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int n = raw.length / 20; float[] p = new float[3 * n]; int[] c = new int[n], s = new int[n];
        HashMap<Long, List<Integer>> cells = new HashMap<>();
        for (int i = 0; i < n; i++) {
            p[3 * i] = b.getFloat(); p[3 * i + 1] = b.getFloat(); p[3 * i + 2] = b.getFloat(); c[i] = b.getInt(); s[i] = b.getInt();
            cells.computeIfAbsent(key(p[3 * i], p[3 * i + 1], p[3 * i + 2], 0, 0, 0), k -> new ArrayList<>()).add(i);
        }
        SurfaceBuilder.Colors colors = (x, y, z, out) -> {
            int best = -1; float bd = 0.03f * 0.03f;
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
                List<Integer> l = cells.get(key(x, y, z, dx, dy, dz)); if (l == null) continue;
                for (int i : l) { if (s[i] == 0) continue;
                    float ex = p[3*i]-x, ey = p[3*i+1]-y, ez = p[3*i+2]-z, dd = ex*ex+ey*ey+ez*ez;
                    if (dd < bd) { bd = dd; best = i; } }
            }
            if (best < 0) return 0; out[0] = 1; return c[best];
        };
        long t0 = System.nanoTime();
        var mesh = SurfaceBuilder.build(frames, colors, smoothing, fill, minConf, 5f, 300000);
        long ms = (System.nanoTime() - t0) / 1_000_000;
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(
                Paths.get(dir, "mesh_s" + smoothing + "_f" + fill + ".bin").toFile())))) {
            ByteBuffer out = ByteBuffer.allocate(mesh.xyz.length * 4 + mesh.color.length * 4 + mesh.source.length).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : mesh.xyz) out.putFloat(v);
            for (int v : mesh.color) out.putInt(v);
            out.put(mesh.source);
            o.write(out.array());
        }
        System.out.printf(Locale.US, "smoothing=%d fill=%d frames=%d/%d triangles=%d photo=%.1f%% estimated=%.1f%% uncovered=%.1f%% buildMs=%d%n",
            smoothing, fill, mesh.framesUsed, frames.size(), mesh.triangles, 100.0 * mesh.photo / mesh.triangles,
            100.0 * mesh.estimated / mesh.triangles, 100.0 * mesh.uncovered / mesh.triangles, ms);
    }
    static long key(float x, float y, float z, int dx, int dy, int dz) {
        return ((long) Math.floor(x / .02) + dx & 0x1fffffL) << 42 | ((long) Math.floor(y / .02) + dy & 0x1fffffL) << 21 | ((long) Math.floor(z / .02) + dz & 0x1fffffL);
    }
}
