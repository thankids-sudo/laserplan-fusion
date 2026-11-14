# LaserPlan Fusion

Standalone Android test project. The app uses native ARCore for capture and native OpenGL ES for viewing. Its application ID is `app.laserplan.fusion`.

## Capture and view

Tap **Scan**, choose confidence, range and interval, then tap **Start**. Move slowly through overlapping views. **Pause** and **Resume** control capture. **Finish & view** saves the scan privately and opens it. **Saved** reopens local captures. **Open ZIP** loads an existing LaserPlan schema-version-1 archive. **Export** copies the selected original survey ZIP.

Use one finger to rotate, two fingers to pan or pinch to zoom. **Fit** resets the view, and **Top** shows the plan view. The slider adjusts point size.

| View | Display |
|---|---|
| Raw geometry | Original scan positions |
| Photo projection | Color sampled from matching photos and blended where several views agree with the depth check |
| Smooth + estimate | Mild local position smoothing and color estimates for existing points surrounded by similarly colored neighbors |
| Coverage map | Mint for photo-sampled points, amber for estimated color, gray where color is unavailable |

Photo projection uses recorded RGB calibration when present. Older exports use an approximate centered crop. Visual registration and reflections can still cause misalignment. Color estimates are local interpolation. They do not use an AI model or create geometry across missing areas. The original scan is available in Raw geometry and ZIP export.

## Build the APK

Use JDK 17, Gradle 8.9, and Android SDK platform 35. Extract the source and run `gradle assembleDebug`, or open the extracted folder in Android Studio and build a debug APK. Output: `app/build/outputs/apk/debug/app-debug.apk`.

The included **Build Android APK** GitHub Actions workflow publishes the APK in the **LaserPlanFusion-debug** artifact when the extracted contents are at the root of a GitHub repository.

## Validation status

A separate Python check evaluated the Java projection logic against the original TV-wall scan. All eight eligible RGB frames contributed to 38,097 of 73,192 points. This measures projection coverage using depth checks, not image-registration accuracy. Android compilation, installation, camera capture and interactive rendering remain unverified.
