package app.laserplan.survey;

import static org.junit.Assert.*;
import org.junit.Test;

public class GuidedFillTest {
    static final int GW = 40, GH = 30;

    /** A wall tilted away to the right: depth 1.5 m at the left edge to 2.5 m at the right edge. */
    static float wall(int i) { return 1.5f + i / (float) (GW - 1); }

    /** Model with the right shape but arbitrary scale and offset: m = 7 / z + 3. */
    static float[] model(float[] truth) {
        float[] m = new float[truth.length];
        for (int k = 0; k < m.length; k++) m[k] = 7f / truth[k] + 3f;
        return m;
    }

    static float[] truth() {
        float[] t = new float[GW * GH];
        for (int j = 0; j < GH; j++) for (int i = 0; i < GW; i++) t[j * GW + i] = wall(i);
        return t;
    }

    static float[] withHole(float[] t, int x0, int y0, int size) {
        float[] d = t.clone();
        for (int j = y0; j < y0 + size; j++) for (int i = x0; i < x0 + size; i++) d[j * GW + i] = 0;
        return d;
    }

    @Test public void enclosedHoleOnTheSameSurfaceIsFilledAccurately() {
        float[] t = truth(), d = withHole(t, 12, 8, 12);
        boolean[] filled = new boolean[GW * GH];
        float[] z = GuidedFill.fill(d, model(t), GW, GH, 5, filled);
        assertNotNull(z);
        int n = 0;
        for (int k = 0; k < z.length; k++) if (filled[k]) { n++; assertEquals(t[k], z[k], 0.005); }
        assertEquals(144, n);
    }

    @Test public void doorwayTheModelSeesFartherAwayStaysOpen() {
        float[] t = truth(), d = withHole(t, 12, 8, 12);
        float[] seen = t.clone();
        for (int j = 8; j < 20; j++) for (int i = 12; i < 24; i++) seen[j * GW + i] = t[j * GW + i] + 2.5f;  // room beyond the door
        boolean[] filled = new boolean[GW * GH];
        GuidedFill.fill(d, model(seen), GW, GH, 5, filled);
        for (boolean f : filled) assertFalse(f);
    }

    @Test public void gapOpenToTheEdgeIsNotFilled() {
        float[] t = truth(), d = t.clone();
        for (int j = 0; j < GH; j++) for (int i = 30; i < GW; i++) d[j * GW + i] = 0;   // right side never scanned
        boolean[] filled = new boolean[GW * GH];
        GuidedFill.fill(d, model(t), GW, GH, 5, filled);
        for (boolean f : filled) assertFalse(f);
    }

    @Test public void tooFewMeasuredCellsGivesNoEstimate() {
        float[] d = new float[GW * GH]; d[0] = 2f; d[1] = 2f;
        assertNull(GuidedFill.fill(d, model(truth()), GW, GH, 5, new boolean[GW * GH]));
    }

    @Test public void modelThatGetsNearAndFarBackwardsIsRejected() {
        float[] t = truth(), m = new float[t.length];
        for (int k = 0; k < m.length; k++) m[k] = t[k];                 // grows with depth: wrong order
        assertNull(GuidedFill.fill(withHole(t, 12, 8, 6), m, GW, GH, 5, new boolean[GW * GH]));
    }

    @Test public void fitRecoversScaleAndOffsetDespiteOutliers() {
        float[] t = truth(), m = model(t);
        for (int k = 0; k < 60; k++) m[k * 13] += 5f;                    // 5 % wild model values
        double[] ab = GuidedFill.fit(t, m);
        assertEquals(1 / 7.0, ab[0], 0.002);
        assertEquals(-3 / 7.0, ab[1], 0.01);
    }
}
