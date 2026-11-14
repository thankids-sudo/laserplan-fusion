package app.laserplan.survey;

import android.graphics.Bitmap;

/** Optional on-device monocular depth model (relative inverse depth). Present only in the "ai" build. */
interface DepthModel extends AutoCloseable {
    /** Model name, version and licence for labels and reports. */
    String label();
    /** Square input side in pixels. */
    int side();
    /** Relative inverse depth, side×side row-major (larger = nearer). */
    float[] predict(Bitmap photo) throws Exception;
    @Override void close();
}
