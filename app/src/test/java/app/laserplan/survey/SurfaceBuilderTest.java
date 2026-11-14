package app.laserplan.survey;

import static org.junit.Assert.*;
import java.util.Collections;
import org.junit.Test;

public class SurfaceBuilderTest {
    /** Identity pose: camera at the origin looking down -Z. */
    private static final double[] IDENTITY = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    private static SurfaceBuilder.Frame frame(int w, int h, int[] mm) {
        SurfaceBuilder.Frame f = new SurfaceBuilder.Frame();
        f.width = w; f.height = h; f.depthMm = new short[w * h];
        for (int i = 0; i < mm.length; i++) f.depthMm[i] = (short) mm[i];
        f.pose = IDENTITY; f.fx = f.fy = 100; f.cx = w / 2.0; f.cy = h / 2.0; f.measured = w * h;
        return f;
    }

    @Test public void depthJumpIsNeverBridged() {
        assertTrue(SurfaceBuilder.connected(2.00f, 2.02f, 2.01f));    // same wall
        assertFalse(SurfaceBuilder.connected(2.00f, 2.00f, 3.10f));   // wall vs doorway behind it
        assertFalse(SurfaceBuilder.connected(2.00f, 0f, 2.00f));      // missing depth
    }

    @Test public void fillClosesOnlyHolesSurroundedByMeasuredDepth() {
        int gw = 5, gh = 5; float[] z = new float[gw * gh];
        java.util.Arrays.fill(z, 2f);
        z[12] = 0;                                                   // one-cell hole in the middle
        z[0] = 0; z[1] = 0; z[5] = 0;                               // corner gap open to the edge
        boolean[] est = SurfaceBuilder.fill(z, gw, gh, 1);
        assertTrue(est[12]); assertEquals(2f, z[12], 1e-6);
        assertFalse(est[0]); assertEquals(0f, z[0], 0f);            // border gaps stay open
    }

    @Test public void fillDoesNotInventSurfaceAcrossAnEdge() {
        int gw = 3, gh = 3;
        float[] z = {1f, 1f, 3f,  1f, 0f, 3f,  1f, 1f, 3f};          // hole between a near and a far surface
        boolean[] est = SurfaceBuilder.fill(z, gw, gh, 2);
        assertFalse(est[4]); assertEquals(0f, z[4], 0f);
    }

    @Test public void smoothingKeepsStepEdgesSharp() {
        int gw = 4, gh = 1;
        float[] z = {1f, 1.01f, 3f, 3.01f};
        SurfaceBuilder.smooth(z, gw, gh, 3);
        assertEquals(1.0f, z[1], .02f);                              // not pulled towards 3 m
        assertEquals(3.0f, z[2], .02f);                              // not pulled towards 1 m
    }

    @Test public void flatWallBecomesOneSurfaceWithoutDuplicates() {
        int w = 20, h = 12; int[] mm = new int[w * h]; java.util.Arrays.fill(mm, 2000);
        SurfaceBuilder.Frame f = frame(w, h, mm);
        SurfaceBuilder.Mesh one = SurfaceBuilder.build(Collections.singletonList(f), null, 0, 0, 0, 5, 100000);
        SurfaceBuilder.Mesh two = SurfaceBuilder.build(java.util.Arrays.asList(f, frame(w, h, mm)), null, 0, 0, 0, 5, 100000);
        assertEquals((w / 2 - 1) * (h / 2 - 1) * 2, one.triangles);
        assertEquals(one.triangles, two.triangles);                  // the second, identical view adds nothing
        assertEquals(one.triangles, one.uncovered);                  // no colours given: all uncovered
        for (int i = 2; i < one.xyz.length; i += 3) assertEquals(-2f, one.xyz[i], 1e-4);
    }

    @Test public void doorwayInAWallStaysOpen() {
        int w = 20, h = 12; int[] mm = new int[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) mm[y * w + x] = (x >= 8 && x < 12) ? 4500 : 2000;
        SurfaceBuilder.Mesh mesh = SurfaceBuilder.build(Collections.singletonList(frame(w, h, mm)), null, 2, 2, 0, 5, 100000);
        for (int t = 0; t < mesh.triangles; t++) {
            float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
            for (int v = 0; v < 3; v++) { float z = mesh.xyz[9 * t + 3 * v + 2]; min = Math.min(min, z); max = Math.max(max, z); }
            assertTrue("triangle bridges wall and doorway", max - min < 1f);
        }
    }

    @Test public void lowConfidenceAndOutOfRangeDepthIsIgnored() {
        int w = 4, h = 4; int[] mm = new int[w * h]; java.util.Arrays.fill(mm, 2000);
        mm[0] = 9000;
        SurfaceBuilder.Frame f = frame(w, h, mm);
        f.confidence = new byte[w * h]; java.util.Arrays.fill(f.confidence, (byte) 200);
        f.confidence[2] = 10;
        int[] size = new int[2];
        float[] z = SurfaceBuilder.sample(f, 64, 5, size);
        assertEquals(0f, z[0], 0f);                                  // 9 m > 5 m range
        assertEquals(0f, z[1], 0f);                                  // pixel (2,0) low confidence
        assertEquals(2f, z[2], 1e-6);
    }

    @Test public void flatSurfaceFillCompletesAWallHole() {
        int w = 40, h = 30; int[] mm = new int[w * h]; java.util.Arrays.fill(mm, 2000);
        for (int y = 10; y < 20; y++) for (int x = 10; x < 30; x++) mm[y * w + x] = 0;   // plain white patch, no depth
        SurfaceBuilder.Frame f = frame(w, h, mm);
        int[] size = new int[2];
        float[] z = SurfaceBuilder.sample(f, 0, 5, size);
        boolean[] est = new boolean[z.length];
        SurfaceBuilder.planeFill(z, est, size[0], size[1], f, 12);
        int filled = 0;
        for (int k = 0; k < z.length; k++) if (est[k]) { filled++; assertEquals(2f, z[k], 0.01f); }
        assertEquals(5 * 10, filled);
    }

    @Test public void flatSurfaceFillLeavesADoorwayOpen() {
        // wall at 2 m above, floor-like surface rising towards the camera below: edges not coplanar
        int w = 40, h = 30; int[] mm = new int[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) mm[y * w + x] = y < 15 ? 2000 : 2000 - (y - 15) * 60;
        for (int y = 8; y < 24; y++) for (int x = 14; x < 26; x++) mm[y * w + x] = 0;
        SurfaceBuilder.Frame f = frame(w, h, mm);
        int[] size = new int[2];
        float[] z = SurfaceBuilder.sample(f, 0, 5, size);
        boolean[] est = new boolean[z.length];
        SurfaceBuilder.planeFill(z, est, size[0], size[1], f, 12);
        for (boolean e : est) assertFalse(e);
    }

    @Test public void smoothedDepthFillsOnlyWhereRawIsMissing() {
        int w = 4, h = 4; int[] mm = new int[w * h]; java.util.Arrays.fill(mm, 2000); mm[0] = 0;
        SurfaceBuilder.Frame f = frame(w, h, mm);
        f.smoothMm = new short[w * h]; java.util.Arrays.fill(f.smoothMm, (short) 2100);
        int[] size = new int[2]; boolean[] smoothed = new boolean[4];
        float[] z = SurfaceBuilder.sample(f, 0, 5, size, smoothed);
        assertTrue(smoothed[0]); assertEquals(2.1f, z[0], 1e-6);
        assertFalse(smoothed[1]); assertEquals(2.0f, z[1], 1e-6);
        float[] measuredOnly = SurfaceBuilder.sample(f, 0, 5, size);
        assertEquals(0f, measuredOnly[0], 0f);
    }
}
