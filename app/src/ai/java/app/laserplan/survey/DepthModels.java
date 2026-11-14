package app.laserplan.survey;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import org.tensorflow.lite.Interpreter;

/**
 * "ai" build: MiDaS v2.1 Small (MIT, Intel ISL) through TensorFlow Lite (Apache-2.0) on the CPU.
 * The model file is fetched by CI from the official MiDaS release and checked by SHA-256; when it is
 * missing (for example a local build without it) open() returns null and the app keeps the local fill.
 */
final class DepthModels {
    static final String ASSET = "midas_v21_small_256.tflite";
    private DepthModels() {}

    static DepthModel open(Context context) {
        try (AssetFileDescriptor fd = context.getAssets().openFd(ASSET);
             FileInputStream in = new FileInputStream(fd.getFileDescriptor())) {
            MappedByteBuffer model = in.getChannel().map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
            Interpreter.Options options = new Interpreter.Options().setNumThreads(4);
            return new Midas(new Interpreter(model, options));
        } catch (Exception e) {
            return null;
        }
    }

    private static final class Midas implements DepthModel {
        private static final int SIDE = 256;
        private static final float[] MEAN = {0.485f, 0.456f, 0.406f}, STD = {0.229f, 0.224f, 0.225f};
        private final Interpreter interpreter;
        private final ByteBuffer input = ByteBuffer.allocateDirect(SIDE * SIDE * 3 * 4).order(ByteOrder.nativeOrder());
        private final float[][][][] output = new float[1][SIDE][SIDE][1];

        Midas(Interpreter interpreter) { this.interpreter = interpreter; }

        @Override public String label() { return "MiDaS v2.1 Small (MIT, Intel ISL) · TensorFlow Lite CPU"; }
        @Override public int side() { return SIDE; }

        @Override public float[] predict(Bitmap photo) {
            Bitmap scaled = Bitmap.createScaledBitmap(photo, SIDE, SIDE, true);
            int[] pixels = new int[SIDE * SIDE];
            scaled.getPixels(pixels, 0, SIDE, 0, 0, SIDE, SIDE);
            if (scaled != photo) scaled.recycle();
            input.rewind();
            for (int p : pixels) {
                input.putFloat((((p >> 16) & 255) / 255f - MEAN[0]) / STD[0]);
                input.putFloat((((p >> 8) & 255) / 255f - MEAN[1]) / STD[1]);
                input.putFloat(((p & 255) / 255f - MEAN[2]) / STD[2]);
            }
            input.rewind();
            interpreter.run(input, output);
            float[] out = new float[SIDE * SIDE];
            for (int y = 0; y < SIDE; y++) for (int x = 0; x < SIDE; x++) out[y * SIDE + x] = output[0][y][x][0];
            return out;
        }

        @Override public void close() { interpreter.close(); }
    }
}
