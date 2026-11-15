package app.laserplan.survey;

import static org.junit.Assert.*;
import java.util.Arrays;
import org.junit.Test;

public class TextureMapperTest {
    static final double[] IDENTITY = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    static TextureMapper.View view(double x) {
        TextureMapper.View v = new TextureMapper.View();
        v.width = 640; v.height = 480; v.fx = v.fy = 400; v.cx = 320; v.cy = 240;
        v.pose = IDENTITY.clone(); v.pose[12] = x;                 // camera at (x, 0, 0) looking down -Z
        return v;
    }

    /** One triangle facing the cameras at depth z, centred on x. */
    static float[] triangle(float x, float z) {
        return new float[]{x - .1f, -.1f, -z, x + .1f, -.1f, -z, x, .1f, -z};
    }

    @Test public void projectsVerticesToPhotoPixels() {
        TextureMapper.Result r = TextureMapper.map(triangle(0, 2), 1, Arrays.asList(view(0)), 4);
        assertEquals(0, r.photo[0]);
        assertEquals(320 - 20, r.uv[0], 1e-3);   // x -0.1 m at 2 m: 400 * -0.05 = -20 px
        assertEquals(240 + 20, r.uv[1], 1e-3);   // y -0.1 m is BELOW centre: image v grows downwards
        assertEquals(240 - 20, r.uv[5], 1e-3);
    }

    @Test public void closerHeadOnPhotoWins() {
        // camera 0 sees the triangle head-on; camera 1 is 1.5 m to the side (oblique, farther)
        TextureMapper.Result r = TextureMapper.map(triangle(0, 2), 1, Arrays.asList(view(1.5), view(0)), 4);
        assertEquals(1, r.photo[0]);
    }

    @Test public void hiddenTriangleIsNotPaintedFromThatPhoto() {
        // a near triangle at 1 m in front of a far one at 3 m, same line of sight
        float[] xyz = new float[18];
        System.arraycopy(triangle(0, 1), 0, xyz, 0, 9);
        System.arraycopy(triangle(0, 3), 0, xyz, 9, 9);
        TextureMapper.Result r = TextureMapper.map(xyz, 2, Arrays.asList(view(0)), 4);
        assertEquals(0, r.photo[0]);
        assertEquals(-1, r.photo[1]);            // the wall behind the cushion keeps its own colour
    }

    @Test public void outsideThePhotoStaysUntextured() {
        TextureMapper.Result r = TextureMapper.map(triangle(5, 2), 1, Arrays.asList(view(0)), 4);
        assertEquals(-1, r.photo[0]);
        assertEquals(0, r.used.length);
    }

    @Test public void atlasIsLimitedToTheMostUsefulPhotos() {
        int n = 6;
        float[] xyz = new float[9 * n];
        TextureMapper.View[] views = new TextureMapper.View[n];
        for (int i = 0; i < n; i++) { System.arraycopy(triangle(3 * i, 2), 0, xyz, 9 * i, 9); views[i] = view(3 * i); }
        TextureMapper.Result r = TextureMapper.map(xyz, n, Arrays.asList(views), 4);
        assertEquals(4, r.used.length);
        assertEquals(4, r.textured);
    }
}
