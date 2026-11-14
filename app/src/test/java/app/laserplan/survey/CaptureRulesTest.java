package app.laserplan.survey;

import static org.junit.Assert.*;
import org.junit.Test;

public class CaptureRulesTest {
    private static final long MS = 1_000_000L;

    @Test public void depthFromAnEarlierFrameFindsThatFramesPose() {
        long[] t = {1000 * MS, 1033 * MS, 1066 * MS, 1100 * MS};
        long[] err = new long[1];
        // Depth 64 ms older than the live frame (as on the S24 Ultra) still matches its own frame.
        assertEquals(1, CaptureRules.findPose(t, null, 1034 * MS, 34 * MS, err));
        assertEquals(1 * MS, err[0]);
    }

    @Test public void androidCameraClockAlsoMatches() {
        long[] arcore = {5 * MS, 0, 0};
        long[] android = {900 * MS, 0, 0};
        assertEquals(0, CaptureRules.findPose(arcore, android, 902 * MS, 34 * MS, null));
    }

    @Test public void tooFarFromEveryFrameReportsNoMatchButKeepsError() {
        long[] t = {1000 * MS, 1033 * MS};
        long[] err = new long[1];
        assertEquals(-1, CaptureRules.findPose(t, null, 1200 * MS, 34 * MS, err));
        assertEquals(167 * MS, err[0]);
    }

    @Test public void emptyHistoryReportsMinusOne() {
        long[] err = new long[1];
        assertEquals(-1, CaptureRules.findPose(new long[4], null, 5 * MS, 34 * MS, err));
        assertEquals(-1, err[0]);
    }

    @Test public void emptyDepthThresholdIsHalfAPercent() {
        assertTrue(CaptureRules.isEmpty(0, 160, 90));
        assertTrue(CaptureRules.isEmpty(71, 160, 90));
        assertFalse(CaptureRules.isEmpty(72, 160, 90));
    }

    @Test public void stridedDepthIsCountedAndDecodedLittleEndian() {
        // 2x2 image, row stride 6 bytes (2 px + 2 padding bytes), pixel stride 2.
        byte[] d = {(byte) 0xE8, 0x03, 0, 0, 9, 9, 0, 0, 0x10, 0x27, 9, 9};
        assertEquals(2, CaptureRules.nonZero16(d, 0, 2, 2, 6, 2));
        byte[] packed = {(byte) 0xE8, 0x03, 0x10, 0x27};
        assertEquals(1000, CaptureRules.millimetres(packed, 0));
        assertEquals(10000, CaptureRules.millimetres(packed, 1));
    }
}
