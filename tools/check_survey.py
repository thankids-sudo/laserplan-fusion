# Offline check of a LaserPlan survey ZIP (extracted): coverage stats, photo-coloured plan view, and
# alignment sheets (real photo | cloud rendered from that photo pose | overlay).
# Usage: python3 tools/check_survey.py <extracted-survey-dir> <output-dir>   (needs numpy, pillow)
import json, glob, os, sys
import numpy as np
from PIL import Image, ImageDraw

root, out = sys.argv[1], sys.argv[2]
cloud = np.loadtxt(os.path.join(root, 'cloud.xyzc'), comments='#')
P = cloud[:, :3]; N = len(P)

def mat(a): return np.array(a, dtype=np.float64).reshape(4, 4).T   # column-major -> world_from_camera

def to_cam(Mw, pts):
    R, t = Mw[:3, :3], Mw[:3, 3]
    return (pts - t) @ R            # = R^T (p - t)

frames = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, 'frames', '*.json')))]
photos = [f for f in frames if f.get('rgb', {}).get('status') == 'saved']
col_sum = np.zeros((N, 3)); col_w = np.zeros(N)
per_photo = []
for f in photos:
    rgb = f['rgb']; k = rgb['intrinsics']
    img = np.asarray(Image.open(os.path.join(root, rgb['file'])).convert('RGB'), dtype=np.float64)
    H, W = img.shape[:2]
    M = mat(rgb['pose']['matrixColumnMajor'] if 'pose' in rgb else f['pose']['matrixColumnMajor'])
    c = to_cam(M, P); z = -c[:, 2]
    ok = z > 0.1
    u = np.full(N, -1.0); v = np.full(N, -1.0)
    u[ok] = k['fx'] * c[ok, 0] / z[ok] + k['cx']
    v[ok] = k['cy'] - k['fy'] * c[ok, 1] / z[ok]
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    # visibility: z-buffer on a coarse 8 px grid; keep points within 5 cm (+2 %) of the nearest surface
    gi = (v[ok] // 8).astype(int) * ((W + 7) // 8) + (u[ok] // 8).astype(int)
    zb = np.full(((H + 7) // 8) * ((W + 7) // 8), np.inf)
    np.minimum.at(zb, gi, z[ok])
    idx = np.where(ok)[0]
    vis = z[idx] <= zb[gi] + 0.05 + 0.02 * z[idx]
    idx = idx[vis]
    colours = img[v[idx].astype(int), u[idx].astype(int)]
    w = 1.0 / np.maximum(z[idx], 0.3) ** 2
    col_sum[idx] += colours * w[:, None]; col_w[idx] += w
    per_photo.append((rgb['file'], int(ok.sum()), int(len(idx))))

coloured = col_w > 0
col = np.where(coloured[:, None], col_sum / np.maximum(col_w, 1e-9)[:, None], 160)
stats = {
  'points': N, 'framesInZip': len(frames), 'validFrames': sum(1 for f in frames if f.get('valid')),
  'framesWithPoints': sum(1 for f in frames if f['pixels']['accepted'] > 0),
  'photos': len(photos),
  'photosWithOwnPose': sum(1 for f in photos if 'pose' in f['rgb']),
  'photosPairedWithFramesThatHavePoints': sum(1 for f in photos if f['pixels']['accepted'] > 0),
  'pointsPhotoColouredPct': round(100 * coloured.mean(), 1),
  'pointsUncoveredPct': round(100 * (1 - coloured.mean()), 1),
  'bboxMinM': P.min(0).round(2).tolist(), 'bboxMaxM': P.max(0).round(2).tolist(),
  'poseSources': {s: sum(1 for f in frames if f['pose'].get('source') == s) for s in set(f['pose'].get('source') for f in frames)},
  'depthToCameraLagMs': {'min': round(min(-f['depthToCameraDeltaNs'] for f in frames)/1e6,1),
                         'median': round(float(np.median([-f['depthToCameraDeltaNs'] for f in frames]))/1e6,1),
                         'max': round(max(-f['depthToCameraDeltaNs'] for f in frames)/1e6,1)},
}
# Floor: the lowest dense horizontal layer
hist, edges = np.histogram(P[:, 1], bins=np.arange(P[:, 1].min(), P[:, 1].max() + 0.02, 0.02))
low = hist[: len(hist) // 3]; fi = int(np.argmax(low))
stats['floorYM'] = round(float(edges[fi] + 0.01), 2)
stats['floorBandPoints'] = int(low[fi])
json.dump(stats, open(os.path.join(out, 'stats.json'), 'w'), indent=1)
print(json.dumps(stats, indent=1))

# 1) top (plan) view coloured by photo
def splat(xs, ys, cs, size=900, pad=20):
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    s = (size - 2 * pad) / max(x1 - x0, y1 - y0)
    im = np.full((size, size, 3), 255, np.uint8); zb = np.full((size, size), -np.inf)
    return im, s, x0, y0, pad
def render_plan(fname):
    order = np.argsort(P[:, 1])               # draw low to high so tops win
    xs, zs = P[order, 0], P[order, 2]
    size, pad = 1000, 30
    x0, x1, z0, z1 = np.percentile(xs, 0.5), np.percentile(xs, 99.5), np.percentile(zs, 0.5), np.percentile(zs, 99.5)
    s = (size - 2 * pad) / max(x1 - x0, z1 - z0)
    im = np.full((size, size, 3), 255, np.uint8)
    px = ((xs - x0) * s + pad).astype(int); py = ((zs - z0) * s + pad).astype(int)
    keep = (px >= 0) & (px < size - 1) & (py >= 0) & (py < size - 1)
    c = col[order][keep].astype(np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            im[py[keep] + dy, px[keep] + dx] = c
    img = Image.fromarray(im); d = ImageDraw.Draw(img)
    d.line([(pad, size - 12), (pad + s, size - 12)], fill=(0, 0, 0), width=3); d.text((pad + s + 6, size - 20), '1 m', fill=(0, 0, 0))
    d.text((pad, 6), 'Top view (plan), points coloured from photos; grey = no photo', fill=(0, 0, 0))
    img.save(os.path.join(out, fname))
render_plan('plan_photo_coloured.png')

# 2) alignment: render the coloured cloud from a photo's own pose next to the real photo
def render_from(f, fname):
    rgb = f['rgb']; k = rgb['intrinsics']; W, H = rgb['width'], rgb['height']
    M = mat(rgb['pose']['matrixColumnMajor'] if 'pose' in rgb else f['pose']['matrixColumnMajor'])
    c = to_cam(M, P); z = -c[:, 2]; ok = z > 0.1
    u = k['fx'] * c[ok, 0] / z[ok] + k['cx']; v = k['cy'] - k['fy'] * c[ok, 1] / z[ok]
    cc = col[ok]; zz = z[ok]; uncol = ~coloured[ok]
    order = np.argsort(-zz)
    im = np.full((H, W, 3), 255, np.uint8)
    ui = u[order].astype(int); vi = v[order].astype(int); keep = (ui >= 0) & (ui < W - 2) & (vi >= 0) & (vi < H - 2)
    cs = cc[order][keep].astype(np.uint8)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            im[vi[keep] + dy, ui[keep] + dx] = cs
    photo = Image.open(os.path.join(root, rgb['file'])).convert('RGB')
    # edge overlay: real photo with rendered points tinted on top at 50 %
    render = Image.fromarray(im)
    mask = Image.fromarray(((im.sum(2) < 765) * 140).astype(np.uint8))
    overlay = photo.copy(); overlay.paste(render, (0, 0), mask)
    # rotate 90° clockwise: sensor landscape -> phone portrait
    tiles = [x.rotate(-90, expand=True) for x in (photo, render, overlay)]
    w, h = tiles[0].size
    sheet = Image.new('RGB', (w * 3 + 20, h + 28), 'white')
    for i, t in enumerate(tiles): sheet.paste(t, (i * (w + 10), 28))
    d = ImageDraw.Draw(sheet)
    for i, label in enumerate(['Real photo ' + rgb['file'], 'Cloud rendered from that photo pose', 'Overlay (cloud at 55 %)']):
        d.text((i * (w + 10) + 4, 8), label, fill=(0, 0, 0))
    sheet.save(os.path.join(out, fname))

ranked = sorted(photos, key=lambda f: -f['pixels']['accepted'])
for i, f in enumerate(ranked[:3]):
    render_from(f, 'alignment_%d_%s.png' % (i + 1, os.path.basename(f['rgb']['file']).split('.')[0]))
print(sorted(os.listdir(out)))
