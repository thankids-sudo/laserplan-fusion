# Offline surface check. Usage (from repo root):
#   python3 tools/surface/prepare.py <extracted-survey> <work>
#   javac -d <classes> app/src/main/java/app/laserplan/survey/SurfaceBuilder.java tools/surface/SurfaceCheck.java
#   java -cp <classes> app.laserplan.survey.SurfaceCheck <work> <smoothing 0-4> <fill 0-4> <minConfidence>
#   python3 tools/surface/render.py <extracted-survey> <work>/mesh_s2_f1.bin <survey>/frames/0067.json out.png
# Writes frames.txt and colors.bin for SurfaceCheck.java from an extracted survey.
# colors.bin: per cloud point x,y,z (float32), argb (int32), source (int32: 0 none, 1 photo)
import json, glob, os, sys
import numpy as np
from PIL import Image
root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cloud = np.loadtxt(os.path.join(root, 'cloud.xyzc'), comments='#'); P = cloud[:, :3]; N = len(P)
frames = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, 'frames', '*.json')))]
def mat(a): return np.array(a, dtype=np.float64).reshape(4, 4).T
def rgb_pose(f):
    r = f['rgb']
    if 'pose' in r: return r['pose']['matrixColumnMajor']
    if 'poseMatrixColumnMajor' in r: return r['poseMatrixColumnMajor']
    return f['pose']['matrixColumnMajor']
cs = np.zeros((N, 3)); cw = np.zeros(N)
for f in frames:
    r = f.get('rgb', {})
    if r.get('status') != 'saved' or 'intrinsics' not in r: continue
    k = r['intrinsics']; img = np.asarray(Image.open(os.path.join(root, r['file'])).convert('RGB'), float)
    H, W = img.shape[:2]; M = mat(rgb_pose(f)); c = (P - M[:3, 3]) @ M[:3, :3]; z = -c[:, 2]
    ok = z > .1; u = np.full(N, -1.); v = np.full(N, -1.)
    u[ok] = k['fx'] * c[ok, 0] / z[ok] + k['cx']; v[ok] = k['cy'] - k['fy'] * c[ok, 1] / z[ok]
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    g = (v[ok] // 8).astype(int) * ((W + 7) // 8) + (u[ok] // 8).astype(int)
    zb = np.full(((H + 7) // 8) * ((W + 7) // 8), np.inf); np.minimum.at(zb, g, z[ok])
    idx = np.where(ok)[0]; idx = idx[z[idx] <= zb[g] + .05 + .02 * z[idx]]
    w = 1 / np.maximum(z[idx], .3) ** 2
    cs[idx] += img[v[idx].astype(int), u[idx].astype(int)] * w[:, None]; cw[idx] += w
col = np.clip(np.round(cs / np.maximum(cw, 1e-9)[:, None]), 0, 255).astype(np.int64)
argb = (0xff << 24) | (col[:, 0] << 16) | (col[:, 1] << 8) | col[:, 2]
rec = np.zeros(N, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('c', '<i4'), ('s', '<i4')])
rec['x'], rec['y'], rec['z'] = P[:, 0], P[:, 1], P[:, 2]
rec['c'] = np.where(cw > 0, argb, 0).astype(np.int64).astype(np.uint32).view(np.int32)
rec['s'] = (cw > 0).astype(np.int32)
rec.tofile(os.path.join(out, 'colors.bin'))
with open(os.path.join(out, 'frames.txt'), 'w') as o:
    for f in frames:
        if not f.get('valid'): continue
        d = f['intrinsics']['depth']; conf = f.get('confidence')
        o.write(' '.join(map(str, [d['width'], d['height'], d['fx'], d['fy'], d['cx'], d['cy'],
            os.path.join(root, f['depth']['file']), os.path.join(root, conf['file']) if conf else '-',
            f['pixels'].get('accepted', 0)] + list(f['pose']['matrixColumnMajor']))) + '\n')
with open(os.path.join(out, 'frames_smooth.txt'), 'w') as o:
    for f in frames:
        sd = f.get('smoothedDepth')
        if not sd or 'pose' not in sd: continue
        k = sd['intrinsics']
        o.write(' '.join(map(str, [sd['width'], sd['height'], k['fx'], k['fy'], k['cx'], k['cy'],
            os.path.join(root, sd['file']), '-', 0] + list(sd['pose']['matrixColumnMajor']))) + '\n')
print('points', N, 'photo-coloured', int((cw > 0).sum()))
