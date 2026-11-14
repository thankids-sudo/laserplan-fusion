package app.laserplan.survey;

/**
 * Model-guided depth fill for large holes, in a photo's cell grid (pure Java, unit-tested).
 *
 * The monocular model gives relative inverse depth (disparity) with a correct surface SHAPE but an
 * unknown scale. It is fitted to the measured cells (1/z = a*m + b, trimmed least squares); the
 * remaining measured-minus-model residual is spread across each hole from its measured edge
 * (harmonic interpolation) and added back. A cell is filled only when:
 *  - it is enclosed: measured depth exists within MAX_REACH cells in all four directions, and
 *  - the model sees it as the same surface as that edge (model disparity within 15 % of the edge),
 *    so a doorway or gap the model sees as farther away stays open.
 * Hold-out test on an S24 Ultra scan (30 photos, 15-cell holes, 17,225 cells this method filled):
 * mean error 5.35 cm vs 5.43 cm for plain harmonic interpolation of the same cells. The depth gain
 * is therefore small; the model's real contribution is the same-surface check (a hole it sees as a
 * different surface, such as a doorway, is left open). Off by default; results labelled AI estimate.
 */
final class GuidedFill {
    static final int MAX_REACH = 15;
    static final float CONSISTENCY = 0.15f;
    static final int MIN_KNOWN = 200;

    private GuidedFill() {}

    /**
     * @param depth  measured depth per cell in metres, 0 = unknown (gw*gh)
     * @param model  model disparity per cell (gw*gh), larger = nearer
     * @param filled output: true where this method estimated depth
     * @return depth with estimated cells filled, or null when the model could not be fitted
     */
    static float[] fill(float[] depth, float[] model, int gw, int gh, float maxRange, boolean[] filled) {
        int n = gw * gh, known = 0;
        for (float d : depth) if (d > 0) known++;
        if (known < MIN_KNOWN) return null;
        double[] ab = fit(depth, model);
        if (ab == null || !(ab[0] > 0)) return null;               // model must order depth correctly
        float[] base = new float[n], residual = new float[n];
        boolean[] isKnown = new boolean[n];
        for (int i = 0; i < n; i++) {
            base[i] = (float) (ab[0] * model[i] + ab[1]);
            if (depth[i] > 0) { isKnown[i] = true; residual[i] = 1f / depth[i] - base[i]; }
        }
        diffuse(residual, isKnown, gw, gh, 300);
        float[] out = depth.clone();
        int[] di = {1, -1, 0, 0}, dj = {0, 0, 1, -1};
        for (int j = 0; j < gh; j++) for (int i = 0; i < gw; i++) {
            int at = j * gw + i;
            if (isKnown[at]) continue;
            float edge = 0; int hits = 0;
            for (int k = 0; k < 4; k++) {
                for (int s = 1; s <= MAX_REACH; s++) {
                    int x = i + di[k] * s, y = j + dj[k] * s;
                    if (x < 0 || y < 0 || x >= gw || y >= gh) break;
                    if (isKnown[y * gw + x]) { edge += base[y * gw + x]; hits++; break; }
                }
            }
            if (hits < 4) continue;                                   // not enclosed by measured depth
            edge /= 4;
            if (edge <= 0 || Math.abs(base[at] - edge) > CONSISTENCY * edge) continue;  // model: different surface
            float disparity = base[at] + residual[at];
            if (disparity <= 0) continue;
            float z = 1f / disparity;
            if (z < 0.1f || z > maxRange) continue;
            out[at] = z; filled[at] = true;
        }
        return out;
    }

    /** 1/z = a*m + b over measured cells, refitted twice without the worst 20 % of residuals. */
    static double[] fit(float[] depth, float[] model) {
        int n = depth.length;
        boolean[] keep = new boolean[n];
        for (int i = 0; i < n; i++) keep[i] = depth[i] > 0 && Float.isFinite(model[i]);
        double a = 0, b = 0;
        for (int pass = 0; pass < 3; pass++) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0; int c = 0;
            for (int i = 0; i < n; i++) if (keep[i]) {
                double x = model[i], y = 1.0 / depth[i];
                sx += x; sy += y; sxx += x * x; sxy += x * y; c++;
            }
            double den = c * sxx - sx * sx;
            if (c < 2 || Math.abs(den) < 1e-12) return null;
            a = (c * sxy - sx * sy) / den; b = (sy - a * sx) / c;
            if (pass == 2) break;
            double[] r = new double[c]; int k = 0;
            for (int i = 0; i < n; i++) if (keep[i]) r[k++] = Math.abs(a * model[i] + b - 1.0 / depth[i]);
            java.util.Arrays.sort(r);
            double limit = Math.max(r[(int) (0.8 * (c - 1))], 1e-9) * 1.5;
            for (int i = 0; i < n; i++) if (keep[i] && Math.abs(a * model[i] + b - 1.0 / depth[i]) >= limit) keep[i] = false;
        }
        return new double[]{a, b};
    }

    /** Harmonic interpolation: unknown cells become the average of their 4 neighbours, known cells fixed. */
    static void diffuse(float[] v, boolean[] known, int gw, int gh, int iterations) {
        float sum = 0; int c = 0;
        for (int i = 0; i < v.length; i++) if (known[i]) { sum += v[i]; c++; }
        float mean = c == 0 ? 0 : sum / c;
        for (int i = 0; i < v.length; i++) if (!known[i]) v[i] = mean;
        float[] next = new float[v.length];
        for (int it = 0; it < iterations; it++) {
            for (int j = 0; j < gh; j++) for (int i = 0; i < gw; i++) {
                int at = j * gw + i;
                if (known[at]) { next[at] = v[at]; continue; }
                float up = v[(j > 0 ? j - 1 : j) * gw + i], down = v[(j < gh - 1 ? j + 1 : j) * gw + i];
                float left = v[j * gw + (i > 0 ? i - 1 : i)], right = v[j * gw + (i < gw - 1 ? i + 1 : i)];
                next[at] = (up + down + left + right) / 4;
            }
            System.arraycopy(next, 0, v, 0, v.length);
        }
    }
}
