# Renders a SurfaceCheck mesh from a photo's pose: real photo | surface | coverage.
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
root, mesh_path, frame_json, out = sys.argv[1:5]
raw = open(mesh_path, 'rb').read(); T = len(raw) // (9 * 4 + 3 * 4 + 3)
xyz = np.frombuffer(raw, '<f4', 9 * T).reshape(T, 3, 3)
col = np.frombuffer(raw, '<u4', 3 * T, 9 * T * 4).reshape(T, 3)
src = np.frombuffer(raw, 'u1', 3 * T, 12 * T * 4).reshape(T, 3)
f = json.load(open(frame_json)); r = f['rgb']; k = r['intrinsics']; W, H = r['width'], r['height']
M = np.array(r['pose']['matrixColumnMajor'], float).reshape(4, 4).T
c = (xyz.reshape(-1, 3) - M[:3, 3]) @ M[:3, :3]; z = -c[:, 2]
u = k['fx'] * c[:, 0] / np.maximum(z, 1e-3) + k['cx']; v = k['cy'] - k['fy'] * c[:, 1] / np.maximum(z, 1e-3)
u, v, z = u.reshape(T, 3), v.reshape(T, 3), z.reshape(T, 3)
keep = (z.min(1) > .1) & (u.max(1) > 0) & (u.min(1) < W) & (v.max(1) > 0) & (v.min(1) < H)
order = np.where(keep)[0][np.argsort(-z[keep].mean(1))]
def draw(colour_of):
    im = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(im)
    for t in order:
        d.polygon([(u[t, 0], v[t, 0]), (u[t, 1], v[t, 1]), (u[t, 2], v[t, 2])], fill=colour_of(t))
    return im
def photo_colour(t):
    cc = col[t]; return (int(np.mean((cc >> 16) & 255)), int(np.mean((cc >> 8) & 255)), int(np.mean(cc & 255)))
def coverage(t):
    s = src[t]
    if (s == 2).any(): return (255, 176, 94)
    if (s == 1).all(): return (114, 217, 178)
    return (119, 136, 152)
photo = Image.open(os.path.join(root, r['file'])).convert('RGB')
tiles = [x.rotate(-90, expand=True) for x in (photo, draw(photo_colour), draw(coverage))]
w, h = tiles[0].size; sheet = Image.new('RGB', (3 * w + 20, h + 28), 'white'); d = ImageDraw.Draw(sheet)
for i, (t, label) in enumerate(zip(tiles, ['Real photo ' + r['file'], 'Photo surface (' + os.path.basename(mesh_path) + ')',
                                          'Coverage: mint photo · amber estimated · grey none'])):
    sheet.paste(t, (i * (w + 10), 28)); d.text((i * (w + 10) + 4, 8), label, fill=(0, 0, 0))
sheet.save(out); print('saved', out, 'triangles drawn', len(order))
