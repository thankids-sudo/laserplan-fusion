# Hold-out evaluation of monocular depth models for hole filling on a LaserPlan survey.
# Usage: python3 tools/ai/eval_depth_models.py <extracted-survey> <model.tflite|model.onnx> <out.json>
# Needs numpy, pillow, ai-edge-litert (tflite) or onnxruntime (onnx). Models are not in the repo.
# Hold-out evaluation: does MiDaS v2.1 Small, fitted to the measured depth around a hole, fill
# hidden depth better than local interpolation? Ground truth = visible fused-cloud depth per photo.
import json, glob, os, sys, time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
root, model_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
rng = np.random.default_rng(7)
cloud = np.loadtxt(os.path.join(root, 'cloud.xyzc'), comments='#')[:, :3]
frames = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, 'frames', '*.json')))]
photos = [f for f in frames if f.get('rgb', {}).get('status') == 'saved']
if model_path.endswith('.onnx'):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_path, so); IN = sess.get_inputs()[0]
    SIDE = IN.shape[2] if isinstance(IN.shape[2], int) else 518
else:
    it = Interpreter(model_path=model_path, num_threads=4); it.allocate_tensors()
    inp, outp = it.get_input_details()[0]['index'], it.get_output_details()[0]['index']; SIDE = 256
MEAN, STD = np.array([.485, .456, .406]), np.array([.229, .224, .225])
GW, GH, CELL = 80, 60, 8
def midas(img):
    x = (np.asarray(img.resize((SIDE, SIDE), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
    t = time.perf_counter()
    if model_path.endswith('.onnx'):
        d = sess.run(None, {IN.name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0].reshape(SIDE, SIDE)
    else:
        it.set_tensor(inp, x[None].astype(np.float32)); it.invoke(); d = it.get_tensor(outp)[0, :, :, 0]
    ms = (time.perf_counter() - t) * 1000
    return np.asarray(Image.fromarray(d).resize((GW, GH), Image.BILINEAR)), ms
def gt_depth(f):
    r = f['rgb']; k = r['intrinsics']; M = np.array(r['pose']['matrixColumnMajor']).reshape(4, 4).T
    c = (cloud - M[:3, 3]) @ M[:3, :3]; z = -c[:, 2]; ok = z > .1
    u = k['fx'] * c[ok, 0] / z[ok] + k['cx']; v = k['cy'] - k['fy'] * c[ok, 1] / z[ok]; z = z[ok]
    g = (u >= 0) & (u < GW * CELL) & (v >= 0) & (v < GH * CELL)
    gi = (v[g] // CELL).astype(int) * GW + (u[g] // CELL).astype(int)
    D = np.full(GW * GH, np.inf); np.minimum.at(D, gi, z[g]); D[np.isinf(D)] = 0
    return D.reshape(GH, GW)
def fit(m, D, mask):                          # 1/z = a*m + b, trimmed least squares
    x, y = m[mask], 1 / D[mask]
    keep = np.ones(len(x), bool)
    for _ in range(3):
        A = np.c_[x[keep], np.ones(keep.sum())]; (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        res = np.abs(a * x + b - y); keep = res < max(np.percentile(res, 80), 1e-6) * 1.5
    return a, b
def diffuse(D, known, iters=300):              # local interpolation: harmonic fill from the hole boundary
    Z = np.where(known, D, 0.).astype(float); w = known.astype(float)
    est = np.where(known, D, np.nan)
    fillv = np.nanmean(D[known]) if known.any() else 0
    Z2 = np.where(known, D, fillv)
    for _ in range(iters):
        p = np.pad(Z2, 1, mode='edge')
        avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
        Z2 = np.where(known, D, avg)
    return Z2
rows = []; times = []
for f in photos:
    D = gt_depth(f); measured = D > 0
    if measured.sum() < 800: continue
    img = Image.open(os.path.join(root, f['rgb']['file'])).convert('RGB')
    m, ms = midas(img); times.append(ms)
    for size in (5, 9, 15):                      # hole side in cells (8 px): ~4-10 cm, ~8-20 cm, ~15-35 cm at 1-2.5 m
        hidden = np.zeros_like(measured)
        for _ in range(400):
            y, x = rng.integers(1, GH - size - 1), rng.integers(1, GW - size - 1)
            blk = measured[y:y + size, x:x + size]
            # bounded: the hole's ring must be mostly measured (as the app requires)
            ring = np.r_[measured[y - 1, x:x + size], measured[y + size, x:x + size], measured[y:y + size, x - 1], measured[y:y + size, x + size]]
            if blk.mean() > .9 and ring.mean() > .9 and not hidden[y - 1:y + size + 1, x - 1:x + size + 1].any():
                hidden[y:y + size, x:x + size] = True
            if hidden.sum() > .15 * measured.sum(): break
        known = measured & ~hidden; test = hidden & measured
        if test.sum() < 20: continue
        a, b = fit(m, D, known)
        glob_pred = 1 / np.clip(a * m + b, 1e-3, None)
        # local alignment: shift by the median disparity residual of measured cells around the hole
        resid = np.where(known, 1 / np.where(known, D, 1) - (a * m + b), np.nan)
        loc = glob_pred.copy()
        ys, xs = np.where(test)
        for yy, xx in zip(ys, xs):
            win = resid[max(0, yy - size): yy + size + 1, max(0, xx - size): xx + size + 1]
            r = np.nanmedian(win) if np.isfinite(win).any() else 0
            loc[yy, xx] = 1 / max(a * m[yy, xx] + b + r, 1e-3)
        interp = diffuse(D, known)
        # model shape + measured edge: interpolate the disparity residual across the hole, add the model back
        base = a * m + b
        rfield = diffuse(np.where(known, 1 / np.where(known, D, 1) - base, 0), known)
        guided = 1 / np.clip(base + rfield, 1e-3, None)
        for name, P in (('local interpolation', interp), ('model global fit', glob_pred), ('model + local alignment', loc), ('model-guided residual fill', guided)):
            e = np.abs(P[test] - D[test]) * 100
            rows.append((size, name, e.mean(), np.median(e), (e < 5).mean() * 100, (e < 2).mean() * 100, test.sum()))
import collections
agg = collections.defaultdict(list)
for r in rows: agg[(r[0], r[1])].append(r)
lines = ['hole (cells)  method                      MAE cm  median cm  <5 cm %  <2 cm %  cells']
summary = {}
for (size, name), rs in sorted(agg.items()):
    w = np.array([r[6] for r in rs], float)
    vals = [np.average([r[i] for r in rs], weights=w) for i in (2, 3, 4, 5)]
    summary[f'{size}:{name}'] = [round(v, 2) for v in vals] + [int(w.sum())]
    lines.append(f'{size:>4}          {name:<26} {vals[0]:6.2f} {vals[1]:9.2f} {vals[2]:8.1f} {vals[3]:8.1f}  {int(w.sum())}')
lines.append(f'photos evaluated: {len(times)}; model CPU inference (x86 desktop, 4 threads): median {np.median(times):.0f} ms')
print('\n'.join(lines)); json.dump({'summary': summary, 'cpuMs': float(np.median(times))}, open(out, 'w'), indent=1)
