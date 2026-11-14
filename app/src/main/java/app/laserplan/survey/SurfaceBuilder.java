package app.laserplan.survey;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Builds the photo-coloured surface preview from the recorded depth frames.
 *
 * Geometry comes only from measured depth: each frame's depth grid is triangulated where all three
 * corners have depth and no depth jump separates them, so doorways, object edges and scan gaps stay
 * open. "Fill" may close small holes, but only when a hole is surrounded by measured depth on a
 * consistent surface; those vertices are marked ESTIMATED. "Smoothing" is edge-preserving: a cell is
 * averaged only with neighbours on the same surface. Overlapping frames are de-duplicated on a 4 cm
 * grid so each surface patch is drawn once.
 *
 * Pure Java (no Android types) so it runs in JVM unit tests and offline checks.
 */
final class SurfaceBuilder {
    static final int UNCOVERED = 0, PHOTO = 1, ESTIMATED = 2, MODEL = 3;
    static final int NEUTRAL = 0xff90a5b6;
    static final int STRIDE = 2;
    static final float OWNER_CELL_M = 0.04f;

    /** Colour of the fused cloud near a position; returns ARGB and writes the source into source[0]. */
    interface Colors { int near(float x, float y, float z, int[] source); }

    static final class Frame {
        int width, height;
        short[] depthMm;      // tight, row-major
        byte[] confidence;    // tight, row-major, may be null
        double[] pose;        // world_from_camera, column-major 4x4
        double fx, fy, cx, cy;
        int measured;         // valid pixels, used to order frames
        byte[] modelMask;     // optional: non-zero where depth is an AI estimate (GuidedFill)
        int[] pixelColor;     // optional: photo colour per pixel for AI-estimated cells
    }

    static final class Mesh {
        float[] xyz = new float[0];
        int[] color = new int[0];
        byte[] source = new byte[0];
        int triangles, photo, estimated, uncovered, model, framesUsed;
    }

    private SurfaceBuilder() {}

    /** Largest depth difference (metres) still treated as one continuous surface at depth z. */
    static float edgeLimit(float z) { return Math.max(0.04f, 0.035f * z); }

    /** Samples the depth image on a STRIDE grid in metres; 0 = no usable depth. */
    static float[] sample(Frame f, int minConfidence, float maxRange, int[] size) {
        int gw = f.width / STRIDE, gh = f.height / STRIDE;
        size[0] = gw; size[1] = gh;
        float[] z = new float[gw * gh];
        for (int j = 0; j < gh; j++) for (int i = 0; i < gw; i++) {
            int at = (j * STRIDE) * f.width + i * STRIDE;
            int mm = f.depthMm[at] & 0xffff;
            int c = f.confidence == null ? 255 : f.confidence[at] & 255;
            if (mm == 0 || c < minConfidence || mm > maxRange * 1000f) continue;
            z[j * gw + i] = mm * 0.001f;
        }
        return z;
    }

    /**
     * Closes holes that are surrounded by measured depth: an empty cell is filled when at least 5 of its
     * 8 neighbours have depth and they agree within edgeLimit. Each pass grows at most one cell inward.
     */
    static boolean[] fill(float[] z, int gw, int gh, int passes) {
        boolean[] estimated = new boolean[z.length];
        for (int pass = 0; pass < passes; pass++) {
            float[] before = z.clone();
            for (int j = 1; j < gh - 1; j++) for (int i = 1; i < gw - 1; i++) {
                int at = j * gw + i;
                if (before[at] > 0) continue;
                int n = 0; float sum = 0, min = Float.MAX_VALUE, max = 0;
                for (int dj = -1; dj <= 1; dj++) for (int di = -1; di <= 1; di++) {
                    if (di == 0 && dj == 0) continue;
                    float v = before[at + dj * gw + di];
                    if (v <= 0) continue;
                    n++; sum += v; min = Math.min(min, v); max = Math.max(max, v);
                }
                if (n >= 5 && max - min < edgeLimit(min)) { z[at] = sum / n; estimated[at] = true; }
            }
        }
        return estimated;
    }

    /** Edge-preserving smoothing: each pass averages a cell with 4-neighbours on the same surface. */
    static void smooth(float[] z, int gw, int gh, int passes) {
        int[] di = {1, -1, 0, 0}, dj = {0, 0, 1, -1};
        for (int pass = 0; pass < passes; pass++) {
            float[] before = z.clone();
            for (int j = 0; j < gh; j++) for (int i = 0; i < gw; i++) {
                int at = j * gw + i;
                float v = before[at];
                if (v <= 0) continue;
                float sum = 2 * v, weight = 2, limit = edgeLimit(v);
                for (int k = 0; k < 4; k++) {
                    int x = i + di[k], y = j + dj[k];
                    if (x < 0 || y < 0 || x >= gw || y >= gh) continue;
                    float u = before[y * gw + x];
                    if (u > 0 && Math.abs(u - v) < limit) { sum += u; weight += 1; }
                }
                z[at] = sum / weight;
            }
        }
    }

    /** True when three grid depths form one surface (all measured, no jump between them). */
    static boolean connected(float a, float b, float c) {
        if (a <= 0 || b <= 0 || c <= 0) return false;
        float min = Math.min(a, Math.min(b, c)), max = Math.max(a, Math.max(b, c));
        return max - min < edgeLimit(min);
    }

    static Mesh build(List<Frame> frames, Colors colors, int smoothing, int fill, int minConfidence,
                      float maxRange, int maxTriangles) {
        List<Frame> order = new ArrayList<>(frames);
        order.sort((p, q) -> Integer.compare(q.measured, p.measured));
        HashMap<Long, Integer> owner = new HashMap<>();
        FloatList xyz = new FloatList(); IntList color = new IntList(); ByteList source = new ByteList();
        Mesh mesh = new Mesh();
        int[] size = new int[2], src = new int[1];
        for (int fi = 0; fi < order.size() && mesh.triangles < maxTriangles; fi++) {
            Frame f = order.get(fi);
            float[] z = sample(f, minConfidence, maxRange, size);
            int gw = size[0], gh = size[1];
            boolean[] estimated = fill(z, gw, gh, fill);
            smooth(z, gw, gh, smoothing);
            // World position, colour and source of every grid cell with depth.
            float[] world = new float[gw * gh * 3];
            int[] cellColor = new int[gw * gh];
            byte[] cellSource = new byte[gw * gh];
            double[] m = f.pose;
            for (int j = 0; j < gh; j++) for (int i = 0; i < gw; i++) {
                int at = j * gw + i;
                float d = z[at];
                if (d <= 0) continue;
                double u = i * STRIDE, v = j * STRIDE;
                double x = (u - f.cx) * d / f.fx, y = (f.cy - v) * d / f.fy, c = -d;
                float wx = (float) (m[0] * x + m[4] * y + m[8] * c + m[12]);
                float wy = (float) (m[1] * x + m[5] * y + m[9] * c + m[13]);
                float wz = (float) (m[2] * x + m[6] * y + m[10] * c + m[14]);
                world[3 * at] = wx; world[3 * at + 1] = wy; world[3 * at + 2] = wz;
                int argb = colors == null ? 0 : colors.near(wx, wy, wz, src);
                if (argb == 0) { cellColor[at] = NEUTRAL; cellSource[at] = UNCOVERED; }
                else { cellColor[at] = argb; cellSource[at] = (byte) src[0]; }
                if (estimated[at]) cellSource[at] = ESTIMATED;
                int pixel = (j * STRIDE) * f.width + i * STRIDE;
                if (f.modelMask != null && f.modelMask[pixel] != 0) {
                    cellSource[at] = MODEL;
                    if (f.pixelColor != null) cellColor[at] = f.pixelColor[pixel];
                }
            }
            boolean used = false;
            for (int j = 0; j + 1 < gh; j++) for (int i = 0; i + 1 < gw; i++) {
                int a = j * gw + i, b = a + 1, c = a + gw, d = c + 1;
                used |= triangle(a, b, c, z, world, cellColor, cellSource, fi, owner, xyz, color, source, mesh);
                used |= triangle(b, d, c, z, world, cellColor, cellSource, fi, owner, xyz, color, source, mesh);
                if (mesh.triangles >= maxTriangles) break;
            }
            if (used) mesh.framesUsed++;
        }
        mesh.xyz = xyz.toArray(); mesh.color = color.toArray(); mesh.source = source.toArray();
        return mesh;
    }

    private static boolean triangle(int a, int b, int c, float[] z, float[] world, int[] cellColor, byte[] cellSource,
                                    int frame, HashMap<Long, Integer> owner, FloatList xyz, IntList color,
                                    ByteList source, Mesh mesh) {
        if (!connected(z[a], z[b], z[c])) return false;
        float cx = (world[3 * a] + world[3 * b] + world[3 * c]) / 3;
        float cy = (world[3 * a + 1] + world[3 * b + 1] + world[3 * c + 1]) / 3;
        float cz = (world[3 * a + 2] + world[3 * b + 2] + world[3 * c + 2]) / 3;
        long key = cellKey(cx) << 42 | cellKey(cy) << 21 | cellKey(cz);
        Integer first = owner.putIfAbsent(key, frame);
        if (first != null && first != frame) return false;   // another frame already drew this patch
        int photo = 0, estimated = 0, model = 0;
        for (int v : new int[]{a, b, c}) {
            xyz.add(world[3 * v]); xyz.add(world[3 * v + 1]); xyz.add(world[3 * v + 2]);
            color.add(cellColor[v]); source.add(cellSource[v]);
            if (cellSource[v] == PHOTO) photo++;
            if (cellSource[v] == ESTIMATED) estimated++;
            if (cellSource[v] == MODEL) model++;
        }
        mesh.triangles++;
        if (model > 0) mesh.model++;
        else if (estimated > 0) mesh.estimated++;
        else if (photo == 3) mesh.photo++;
        else mesh.uncovered++;
        return true;
    }

    private static long cellKey(float v) { return ((long) Math.floor(v / OWNER_CELL_M)) & 0x1fffffL; }

    static final class FloatList {
        float[] a = new float[1 << 16]; int n;
        void add(float v) { if (n == a.length) a = Arrays.copyOf(a, n * 2); a[n++] = v; }
        float[] toArray() { return Arrays.copyOf(a, n); }
    }
    static final class IntList {
        int[] a = new int[1 << 15]; int n;
        void add(int v) { if (n == a.length) a = Arrays.copyOf(a, n * 2); a[n++] = v; }
        int[] toArray() { return Arrays.copyOf(a, n); }
    }
    static final class ByteList {
        byte[] a = new byte[1 << 15]; int n;
        void add(byte v) { if (n == a.length) a = Arrays.copyOf(a, n * 2); a[n++] = v; }
        byte[] toArray() { return Arrays.copyOf(a, n); }
    }
}
