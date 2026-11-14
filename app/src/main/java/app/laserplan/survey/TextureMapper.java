package app.laserplan.survey;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Projective texturing: each surface triangle shows the real pixels of the one photo that sees it
 * best, instead of one blended colour per vertex (which blurs patterns, grain and lettering).
 *
 * For every photo: the triangle must be fully inside the image and in front of the camera, and visible
 * (its centre within 4 cm + 1.5 % of the nearest triangle centre in a 4 px depth buffer of the mesh
 * seen from that photo, so a cushion never paints the wall behind it). Among those photos the one with
 * the most head-on, closest view away from the image border wins. To keep the texture atlas small, the
 * photos are limited to the maxPhotos that win the most triangles, and every triangle is then
 * reassigned among those. Triangles no kept photo sees keep their vertex colours.
 *
 * Pure Java (no Android types) so it can be unit-tested and run offline.
 */
final class TextureMapper {
    static final class View {
        int width, height;
        double fx, fy, cx, cy;
        double[] pose;          // world_from_camera, column-major
    }

    static final class Result {
        int[] photo;            // per triangle: index into views, or -1
        float[] uv;             // per vertex: pixel u, v in its triangle's photo (2 per vertex)
        int[] used;             // views used, most triangles first
        int textured;
    }

    private TextureMapper() {}

    static Result map(float[] xyz, int triangles, List<View> views, int maxPhotos) {
        int t = triangles;
        float[] cen = new float[3 * t], nrm = new float[3 * t];
        for (int i = 0; i < t; i++) {
            int a = 9 * i;
            for (int k = 0; k < 3; k++) cen[3 * i + k] = (xyz[a + k] + xyz[a + 3 + k] + xyz[a + 6 + k]) / 3;
            float ux = xyz[a + 3] - xyz[a], uy = xyz[a + 4] - xyz[a + 1], uz = xyz[a + 5] - xyz[a + 2];
            float vx = xyz[a + 6] - xyz[a], vy = xyz[a + 7] - xyz[a + 1], vz = xyz[a + 8] - xyz[a + 2];
            float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12f;
            nrm[3 * i] = nx / len; nrm[3 * i + 1] = ny / len; nrm[3 * i + 2] = nz / len;
        }
        // Pass 1: best photo per triangle among all photos, to rank photos by usefulness.
        int[] best = new int[t];
        assign(xyz, cen, nrm, t, views, null, best, null);
        int[] wins = new int[views.size()];
        for (int b : best) if (b >= 0) wins[b]++;
        Integer[] order = new Integer[views.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (p, q) -> Integer.compare(wins[q], wins[p]));
        boolean[] keep = new boolean[views.size()];
        List<Integer> used = new ArrayList<>();
        for (int i = 0; i < order.length && used.size() < maxPhotos; i++) {
            if (wins[order[i]] == 0) break;
            keep[order[i]] = true; used.add(order[i]);
        }
        // Pass 2: reassign every triangle among the kept photos, recording the projected pixels.
        Result r = new Result();
        r.photo = new int[t];
        r.uv = new float[6 * t];
        assign(xyz, cen, nrm, t, views, keep, r.photo, r.uv);
        r.used = new int[used.size()];
        for (int i = 0; i < r.used.length; i++) r.used[i] = used.get(i);
        for (int p : r.photo) if (p >= 0) r.textured++;
        return r;
    }

    private static void assign(float[] xyz, float[] cen, float[] nrm, int t, List<View> views, boolean[] keep,
                               int[] best, float[] uvOut) {
        Arrays.fill(best, -1);
        float[] score = new float[t];
        float[] pu = new float[3 * t], pv = new float[3 * t], pz = new float[3 * t];
        for (int pi = 0; pi < views.size(); pi++) {
            if (keep != null && !keep[pi]) continue;
            View p = views.get(pi);
            double[] m = p.pose;
            // project every vertex once
            for (int q = 0; q < 3 * t; q++) {
                double x = xyz[3 * q] - m[12], y = xyz[3 * q + 1] - m[13], z = xyz[3 * q + 2] - m[14];
                double a = m[0] * x + m[1] * y + m[2] * z, b = m[4] * x + m[5] * y + m[6] * z, c = -(m[8] * x + m[9] * y + m[10] * z);
                pz[q] = (float) c;
                if (c > 0.15) { pu[q] = (float) (p.fx * a / c + p.cx); pv[q] = (float) (p.cy - p.fy * b / c); }
            }
            // depth buffer (4 px cells) with every triangle rasterised, so large near triangles hide what is behind
            int gw = (p.width + 3) / 4, gh = (p.height + 3) / 4;
            float[] zbuf = new float[gw * gh];
            Arrays.fill(zbuf, Float.MAX_VALUE);
            for (int i = 0; i < t; i++) {
                int q = 3 * i;
                if (pz[q] <= 0.15f || pz[q + 1] <= 0.15f || pz[q + 2] <= 0.15f) continue;
                float x0 = pu[q], y0 = pv[q], x1 = pu[q + 1], y1 = pv[q + 1], x2 = pu[q + 2], y2 = pv[q + 2];
                int cx0 = Math.max(0, (int) Math.floor(Math.min(x0, Math.min(x1, x2)) / 4));
                int cx1 = Math.min(gw - 1, (int) Math.floor(Math.max(x0, Math.max(x1, x2)) / 4));
                int cy0 = Math.max(0, (int) Math.floor(Math.min(y0, Math.min(y1, y2)) / 4));
                int cy1 = Math.min(gh - 1, (int) Math.floor(Math.max(y0, Math.max(y1, y2)) / 4));
                if (cx0 > cx1 || cy0 > cy1 || (cx1 - cx0 + 1) * (cy1 - cy0 + 1) > 4096) continue;
                float den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
                if (Math.abs(den) < 1e-6f) continue;
                for (int gy = cy0; gy <= cy1; gy++) for (int gx = cx0; gx <= cx1; gx++) {
                    float sx = gx * 4 + 2, sy = gy * 4 + 2;
                    float l0 = ((y1 - y2) * (sx - x2) + (x2 - x1) * (sy - y2)) / den;
                    float l1 = ((y2 - y0) * (sx - x2) + (x0 - x2) * (sy - y2)) / den;
                    float l2 = 1 - l0 - l1;
                    if (l0 < -0.01f || l1 < -0.01f || l2 < -0.01f) continue;
                    float z = l0 * pz[q] + l1 * pz[q + 1] + l2 * pz[q + 2];
                    int g = gy * gw + gx;
                    if (z < zbuf[g]) zbuf[g] = z;
                }
            }
            for (int i = 0; i < t; i++) {
                int q = 3 * i;
                boolean inside = true;
                float edge = Float.MAX_VALUE;
                for (int k = 0; k < 3; k++) {
                    float u = pu[q + k], v = pv[q + k];
                    if (pz[q + k] <= 0.15f || u < 0 || v < 0 || u > p.width - 1 || v > p.height - 1) { inside = false; break; }
                    edge = Math.min(edge, Math.min(Math.min(u, p.width - u), Math.min(v, p.height - v)));
                }
                if (!inside) continue;
                float cu = (pu[q] + pu[q + 1] + pu[q + 2]) / 3, cv = (pv[q] + pv[q + 1] + pv[q + 2]) / 3;
                float cz = (pz[q] + pz[q + 1] + pz[q + 2]) / 3;
                int g = ((int) cv >> 2) * gw + ((int) cu >> 2);
                if (cz > zbuf[g] + 0.04f + 0.015f * cz) continue;                     // hidden behind a nearer surface
                double vx = m[12] - cen[3 * i], vy = m[13] - cen[3 * i + 1], vz = m[14] - cen[3 * i + 2];
                double dist = Math.sqrt(vx * vx + vy * vy + vz * vz);
                double cos = Math.abs(nrm[3 * i] * vx + nrm[3 * i + 1] * vy + nrm[3 * i + 2] * vz) / dist;
                float s = (float) (cos * cos / Math.max(dist, 0.3) * Math.max(0.2, Math.min(1, edge / 80.0)));
                if (s > score[i]) {
                    score[i] = s; best[i] = pi;
                    if (uvOut != null) for (int k = 0; k < 3; k++) { uvOut[6 * i + 2 * k] = pu[q + k]; uvOut[6 * i + 2 * k + 1] = pv[q + k]; }
                }
            }
        }
    }
}
