package app.laserplan.survey;

import android.content.Context;

/** Standard build: no AI model is packaged, so AI depth fill is unavailable and the local fill is used. */
final class DepthModels {
    private DepthModels() {}
    static DepthModel open(Context context) { return null; }
}
