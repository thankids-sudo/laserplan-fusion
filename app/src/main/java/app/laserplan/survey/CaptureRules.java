package app.laserplan.survey;

/** Pure capture decisions, kept free of Android types so they run as JVM unit tests in CI. */
final class CaptureRules {
    private CaptureRules() {}

    /**
     * Index of the remembered frame closest to [time] (ARCore or Android camera clock), or -1 when
     * nothing is within [toleranceNs]. out[0] receives the closest error in ns (or -1 when empty).
     */
    static int findPose(long[] arcoreTime, long[] androidTime, long time, long toleranceNs, long[] out) {
        int best = -1; long bestError = Long.MAX_VALUE;
        for (int i = 0; i < arcoreTime.length; i++) {
            if (arcoreTime[i] == 0) continue;
            long error = Math.abs(arcoreTime[i] - time);
            if (androidTime != null && androidTime[i] != 0) error = Math.min(error, Math.abs(androidTime[i] - time));
            if (error < bestError) { bestError = error; best = i; }
        }
        if (out != null) out[0] = best < 0 ? -1 : bestError;
        return best >= 0 && bestError <= toleranceNs ? best : -1;
    }

    /** True when a raw depth image has too few measured pixels to be worth a frame (under 0.5 %). */
    static boolean isEmpty(int nonZeroPixels, int width, int height) {
        return nonZeroPixels < Math.max(1, width * height / 200);
    }

    /** Non-zero 16-bit little-endian pixels in a tightly packed or strided buffer. */
    static int nonZero16(byte[] data, int origin, int width, int height, int rowStride, int pixelStride) {
        int n = 0;
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) {
            int i = origin + y * rowStride + x * pixelStride;
            if (data[i] != 0 || data[i + 1] != 0) n++;
        }
        return n;
    }

    /** Millimetres stored little-endian at pixel [index] of a tightly packed 16-bit image. */
    static int millimetres(byte[] depth, int index) {
        return (depth[index * 2] & 255) | ((depth[index * 2 + 1] & 255) << 8);
    }
}
