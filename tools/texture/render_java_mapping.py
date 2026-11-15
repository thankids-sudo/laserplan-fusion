# Offline: render using the app TextureMapper output written by TextureCheck.java.
# Usage: python3 tools/texture/render_java_mapping.py <survey> <mesh.bin> <out.png> rgb/NNNN.jpg <mapping.bin> (needs <work>/views.txt; see TextureCheck)
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
names=[l.split()[0] for l in open('/home/claude/surf2/views.txt')]
order_idx={p['name']:i for i,p in enumerate(photos)}
rawm=open(sys.argv[5],'rb').read()
jb=np.frombuffer(rawm,'<i4',T); juv=np.frombuffer(rawm,'<f4',6*T,4*T).reshape(T,3,2)
best=np.array([order_idx[names[j]] if j>=0 else -1 for j in jb]); UV=juv.astype(float)
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
for i, (tile, lab) in enumerate(zip(tiles, ['Before (0.7/0.8): colour per vertex, confidence 128', 'After (0.9): Java photo texture + confidence 32 + flat fill', 'Nearest real photo ' + src['name'] + ' (different position)'])):
    sheet.paste(tile, (i * (w + 10), 28)); dd.text((i * (w + 10) + 4, 8), lab, fill=(0, 0, 0))
sheet.save(out); print('saved', out)
