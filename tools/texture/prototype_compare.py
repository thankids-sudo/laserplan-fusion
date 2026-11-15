# Offline: render a SurfaceCheck mesh from a novel view, vertex colour vs per-triangle photo texture.
# Usage: python3 tools/texture/prototype_compare.py <survey> <mesh.bin> <out.png> rgb/NNNN.jpg
# Prototype: per-triangle best-photo projective texturing vs per-vertex colour, rendered from a novel view.
import json, glob, os, sys, numpy as np
from PIL import Image, ImageDraw
root, mesh_path, out = sys.argv[1:4]
raw = open(mesh_path, 'rb').read(); T = len(raw) // 51
X = np.frombuffer(raw, '<f4', 9 * T).reshape(T, 3, 3).astype(np.float64)
COL = np.frombuffer(raw, '<u4', 3 * T, 36 * T).reshape(T, 3)
frames = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, 'frames', '*.json')))]
photos = []
for f in frames:
    r = f.get('rgb', {})
    if r.get('status') != 'saved' or 'intrinsics' not in r: continue
    photos.append(dict(k=r['intrinsics'], M=np.array(r['pose']['matrixColumnMajor']).reshape(4, 4).T,
                       img=np.asarray(Image.open(os.path.join(root, r['file'])).convert('RGB')), name=r['file']))
W, H = 640, 480
def project(M, k, P):
    c = (P - M[:3, 3]) @ M[:3, :3]; z = -c[..., 2]
    return k['fx'] * c[..., 0] / np.maximum(z, 1e-6) + k['cx'], k['cy'] - k['fy'] * c[..., 1] / np.maximum(z, 1e-6), z
cen = X.mean(1); nrm = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
best = np.full(T, -1); bscore = np.zeros(T); UV = np.zeros((T, 3, 2))
for pi, p in enumerate(photos):
    u, v, z = project(p['M'], p['k'], X)                      # (T,3)
    inside = (z.min(1) > .15) & (u.min(1) >= 0) & (u.max(1) < W - 1) & (v.min(1) >= 0) & (v.max(1) < H - 1)
    # visibility: coarse z-buffer of triangle centroids (4 px cells)
    cu, cv, cz = project(p['M'], p['k'], cen)
    ok = inside.copy(); gi = (cv[ok] // 4).astype(int) * (W // 4) + (cu[ok] // 4).astype(int)
    zb = np.full((H // 4) * (W // 4), np.inf); np.minimum.at(zb, gi, cz[ok])
    vis = np.zeros(T, bool); idx = np.where(ok)[0]; vis[idx] = cz[idx] <= zb[gi] + .04 + .015 * cz[idx]
    view = p['M'][:3, 3] - cen; dist = np.linalg.norm(view, axis=1); cosang = np.abs((nrm * view).sum(1)) / dist
    # prefer head-on, close, and away from the photo border (less blur and distortion)
    edge = np.minimum.reduce([u.min(1), W - u.max(1), v.min(1), H - v.max(1)]) / 80
    score = np.where(vis, cosang ** 2 / np.maximum(dist, .3) * np.clip(edge, .2, 1), 0)
    better = score > bscore
    best[better] = pi; bscore[better] = score[better]; UV[better] = np.stack([u, v], -1)[better]
print('triangles textured from a photo: %.1f%%' % (100 * (best >= 0).mean()))
def render(Mv, k, mode):
    u, v, z = project(Mv, k, X); keep = (z.min(1) > .15)
    order = np.where(keep)[0]; order = order[np.argsort(-z[order].mean(1))]
    idimg = Image.new('I', (W, H), -1); d = ImageDraw.Draw(idimg)
    for t in order: d.polygon([(u[t, 0], v[t, 0]), (u[t, 1], v[t, 1]), (u[t, 2], v[t, 2])], fill=int(t))
    ids = np.asarray(idimg); outim = np.full((H, W, 3), 255, np.uint8)
    ys, xs = np.where(ids >= 0); t = ids[ys, xs]
    # barycentric coordinates of each pixel in its triangle
    a = np.stack([u[t, 0], v[t, 0]], 1); b = np.stack([u[t, 1], v[t, 1]], 1); c = np.stack([u[t, 2], v[t, 2]], 1); pnt = np.stack([xs + .5, ys + .5], 1)
    v0, v1, v2 = b - a, c - a, pnt - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]; den[np.abs(den) < 1e-9] = 1e-9
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den; l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den; l0 = 1 - l1 - l2
    L = np.clip(np.stack([l0, l1, l2], 1), 0, 1)
    if mode == 'vertex':
        cc = COL[t]; rgb = np.stack([(cc >> 16) & 255, (cc >> 8) & 255, cc & 255], -1).astype(float)   # (N,3,3)
        outim[ys, xs] = (L[:, :, None] * rgb).sum(1).astype(np.uint8)
    else:
        tex = best[t] >= 0
        cc = COL[t]; rgb = np.stack([(cc >> 16) & 255, (cc >> 8) & 255, cc & 255], -1).astype(float)
        outim[ys, xs] = (L[:, :, None] * rgb).sum(1).astype(np.uint8)
        tt = t[tex]; uv = (L[tex][:, :, None] * UV[tt]).sum(1)
        for pi in np.unique(best[tt]):
            sel = best[tt] == pi; img = photos[pi]['img']
            px = np.clip(uv[sel, 0].astype(int), 0, W - 1); py = np.clip(uv[sel, 1].astype(int), 0, H - 1)
            outim[ys[tex][sel], xs[tex][sel]] = img[py, px]
    return Image.fromarray(outim)
# novel view: take a photo pose, step 0.6 m back along its viewing axis and 0.4 m to the side, turn 12° towards the centre
src = [p for p in photos if p['name'] == sys.argv[4]][0]
Mv = src['M'].copy(); R = Mv[:3, :3]
ang = np.radians(-12); Ry = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
Mv[:3, :3] = R @ Ry; Mv[:3, 3] = Mv[:3, 3] + R @ np.array([0.4, 0, 0.6])
tiles = [render(Mv, src['k'], 'vertex').rotate(-90, expand=True), render(Mv, src['k'], 'texture').rotate(-90, expand=True),
         Image.fromarray(src['img']).rotate(-90, expand=True)]
w, h = tiles[0].size; sheet = Image.new('RGB', (3 * w + 20, h + 28), 'white'); dd = ImageDraw.Draw(sheet)
for i, (tile, lab) in enumerate(zip(tiles, ['Current (0.7/0.8): colour per vertex', 'Proposed: photo texture per triangle', 'Nearest real photo ' + src['name'] + ' (different position)'])):
    sheet.paste(tile, (i * (w + 10), 28)); dd.text((i * (w + 10) + 4, 8), lab, fill=(0, 0, 0))
sheet.save(out); print('saved', out)
