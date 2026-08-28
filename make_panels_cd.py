# Картинки-результаты для схемы, все заливкой по клеточным территориям:
#   c_layer1_fill.png  — слой 1, опухоль/строма (обучающий срез)
#   c3_layer2_pred.png — слой 2, 5 зон (тот же обучающий срез, то же окно)
#   graph_stardist.png — StarDist-ядра + граф, узлы окрашены слоем 1 (малый регион TCGA)
#   d1_tcga_he.png     — фрагмент H&E TCGA (вход)
#   d2_zonemap.png     — тот же фрагмент, карта зон (выход)
#
# Тот же инференс, что в боевом graph_infer.
#   заливки слоёв 1 и 2 (обучающий срез, граф уже построен — быстро):
#     python make_panels_cd.py --he data/raw/ovary2/..._he_image.ome.tif \
#         --slide ovary2_he --layer1 runs/graph/l1_final.pth --layer2 runs/graph/l2_final.pth
#   граф+StarDist и панель d (TCGA):
#     python make_panels_cd.py --tcga data/tcga_ov_flat/TCGA-57-1585-...svs \
#         --layer1 runs/graph/l1_final.pth --layer2 runs/graph/l2_final.pth

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from graph_infer import load_gnn, classify, scan_block, fill_cells, fill_zones, L2_TO_CLASS, COL
from cell_graph_train import adjacency
from make_figure_assets import pick_zoom, thumb

Image.MAX_IMAGE_PIXELS = None


class NoNorm:
    """Пустой нормализатор: обучение шло на сырых H&E, поэтому на обучающем срезе
    Macenko не применяем — иначе модель видит не тот цвет и путает классы."""

    def transform(self, x):
        return x


class ArrayReader:
    """Мини-ридер над куском H&E в памяти — тот же интерфейс, что нужен scan_block."""

    def __init__(self, arr, mpp):
        self.img = arr
        self.H, self.W = arr.shape[:2]
        self.mpp = mpp

    def read(self, x, y, s):
        return self.img[y:y + s, x:x + s]


def combine(p1, p2):
    """Слой 1 (0 опухоль / 1 строма) + канал слоя 2 -> итоговый класс зоны (5 зон)."""
    cls = np.empty(len(p1), np.uint8)
    cls[p1 == 0] = TUMOR
    sub = np.array([L2_TO_CLASS[int(c)] for c in p2], np.uint8)
    cls[p1 == 1] = sub[p1 == 1]
    return cls


def l1_class(p1):
    return np.where(p1 == 0, TUMOR, STROMA).astype(np.uint8)


def fill_zones_solid(he, xy, cls, ox, oy, alpha=0.55, boundary=True):
    """Сплошная заливка зон: пиксель = класс ближайшей клетки, границы только на
    смене класса (клетки одного класса сливаются в сплошное поле, без мозаики)."""
    h, w = he.shape[:2]
    step = 2 if max(h, w) > 2200 else 1
    hs, ws = (h + step - 1) // step, (w + step - 1) // step
    gx, gy = np.meshgrid(np.arange(ws) * step, np.arange(hs) * step)
    _, idx = cKDTree(xy - [ox, oy]).query(np.c_[gx.ravel(), gy.ravel()])
    clsmap = cls[idx].reshape(hs, ws).astype(np.uint8)
    if step > 1:
        clsmap = np.asarray(Image.fromarray(clsmap).resize((w, h), Image.NEAREST))
    cmap = np.zeros((h, w, 3), np.uint8)
    for c, col in COL.items():
        cmap[clsmap == c] = col
    ov = he.copy()
    m = clsmap > 0
    ov[m] = (alpha * cmap[m] + (1 - alpha) * he[m]).astype(np.uint8)
    if boundary:
        b = np.zeros((h, w), bool)
        b[:, 1:] |= clsmap[:, 1:] != clsmap[:, :-1]
        b[1:, :] |= clsmap[1:, :] != clsmap[:-1, :]
        ov[b] = (55, 55, 55)
    return ov


def edges_for(xy, mpp, knn, max_edge_um):
    tree = cKDTree(xy)
    d, nb = tree.query(xy, k=min(knn + 1, len(xy)))
    if nb.ndim == 1:
        nb, d = nb[:, None], d[:, None]
    r_px = max_edge_um / mpp
    src = np.repeat(np.arange(len(xy)), nb.shape[1] - 1)
    dst = nb[:, 1:].ravel()
    ok = d[:, 1:].ravel() <= r_px
    if ok.sum() == 0:
        return np.zeros((2, 0), np.int64), nb, d
    e = np.vstack([np.r_[src[ok], dst[ok]], np.r_[dst[ok], src[ok]]]).astype(np.int64)
    return e, nb, d


def classify_l1(xy, Fe, l1, device, mpp, knn, max_edge_um):
    e, _, _ = edges_for(xy, mpp, knn, max_edge_um)
    X = torch.from_numpy(Fe.astype(np.float32)).to(device)
    A = adjacency(e, len(xy), device)
    with torch.no_grad():
        return l1(X, A).argmax(1).cpu().numpy()


def draw_graph(he, xy, ox, oy, cls, mpp, max_edge_um, scale=2, r=5):
    """H&E с ядрами (узлы, окрашены классом) и рёбрами графа ≤ max_edge_um."""
    im = Image.fromarray(he.copy())
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(im, "RGBA")
    p = (xy - [ox, oy]) * scale
    if len(p) > 2:
        _, nb, d = edges_for(xy, mpp, 8, max_edge_um)
        r_px = max_edge_um / mpp * scale
        for i in range(len(p)):
            for j, dist in zip(nb[i][1:], d[i][1:] * scale):
                if dist <= r_px:
                    dr.line([tuple(p[i]), tuple(p[j])], fill=(35, 35, 35, 150), width=2)
    for (x, y), c in zip(p, cls):
        col = COL.get(int(c), (120, 120, 120))
        dr.ellipse([x - r, y - r, x + r, y + r], fill=col + (235,), outline=(20, 20, 20), width=2)
    return np.asarray(im)


def panel_c(args, device, l1, l2, sd, normalize, uni, norm, out):
    """Заливки слоёв 1 и 2 на обучающем срезе. Окно выбираем по разметке графа
    (чтобы были все зоны), но ядра детектим StarDist-ом по H&E — тогда закрашены
    все клетки, а не подвыборка патчей."""
    g = np.load(Path(args.graph_dir) / f"{args.slide}_graph.npz")
    pos, label = g["pos"], g["label"]
    mpp = float(g["mpp"]) if "mpp" in g else 0.2738

    from src.data.patching import load_image
    print("грузим H&E обучающего среза...")
    img = load_image(args.he)
    H, W = img.shape[:2]
    z = min(args.zoom, H, W)
    x0, y0 = pick_zoom(pos, label, W, H, z)
    crop = np.ascontiguousarray(img[y0:y0 + z, x0:x0 + z])
    del img
    print(f"окно {x0},{y0} {z}x{z} @ {mpp:.4f} мкм/px — StarDist по всем ядрам...")
    sl = ArrayReader(crop, mpp)
    xy, Fe, _ = scan_block(sl, sd, normalize, uni, norm, device, 0, 0, z, args.patch_size, False)
    if len(xy) < 40:
        print("мало ядер в окне — попробуй другое окно"); return
    e, _, _ = edges_for(xy, mpp, args.knn, args.max_edge_um)
    X = torch.from_numpy(Fe.astype(np.float32)).to(device)
    A = adjacency(e, len(xy), device)
    with torch.no_grad():
        p1 = l1(X, A).argmax(1).cpu().numpy()
        p2 = l2(X, A).argmax(1).cpu().numpy()
    cls5, cls1 = combine(p1, p2), l1_class(p1)
    med = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    print(f"ядер {len(xy)}: опухоль {int((cls1==TUMOR).sum())}, строма {int((cls1==STROMA).sum())}")
    for name, cls in (("c_layer1_fill", cls1), ("c3_layer2_pred", cls5)):
        ov = fill_zones_solid(crop, xy, cls, 0, 0, alpha=0.55)
        ov, _ = thumb(ov, args.fig_dim)
        Image.fromarray(ov).save(out / f"{name}.png")
    print("c_layer1_fill.png, c3_layer2_pred.png")


def panel_c_whole(args, device, l1, l2, out):
    """Весь срез: зоны по готовому графу — точная классификация (сырые признаки
    обучения, без Macenko), сплошная заливка по всему срезу."""
    from src.data.patching import load_image
    g = np.load(Path(args.graph_dir) / f"{args.slide}_graph.npz")
    pos, edges = g["pos"].astype(np.float32), g["edges"]
    mpp = float(g["mpp"]) if "mpp" in g else 0.2738
    X = torch.from_numpy(g["feat"].astype(np.float32)).to(device)
    A = adjacency(edges, len(pos), device)
    with torch.no_grad():
        p1 = l1(X, A).argmax(1).cpu().numpy()
        p2 = l2(X, A).argmax(1).cpu().numpy()
    cls5, cls1 = combine(p1, p2), l1_class(p1)
    print(f"{args.slide}: узлов {len(pos)}, опухоль {int((cls1==TUMOR).sum())}, "
          f"строма {int((cls1==STROMA).sum())}")

    print("грузим H&E (весь срез)...")
    img = load_image(args.he)
    H, W = img.shape[:2]
    md = min(args.whole_dim, max(H, W))
    tw, th = int(W * md / max(H, W)), int(H * md / max(H, W))
    bg = np.asarray(Image.fromarray(img).resize((tw, th), Image.LANCZOS))
    del img
    sc = W / tw
    for name, cls in (("c_layer1_whole", cls1), ("c3_layer2_whole", cls5)):
        ov = smooth_zones(bg, pos, cls, sc, mpp, alpha=0.55,
                          bin_um=args.bin_um, fill_um=args.fill_um)
        Image.fromarray(ov).save(out / f"{name}.png")
    print(f"c_layer1_whole.png, c3_layer2_whole.png  ({tw}x{th})")


def smooth_zones(bg, xy, cls, sc, mpp, alpha=0.55, bin_um=30.0, fill_um=110.0):
    """Сглаженные зоны для обзора всего среза: ячейки ~bin_um, в каждой — преобладающий
    класс, пустые достраиваются от ближайшей населённой в пределах fill_um. Убирает
    крап от подвыборки патчей, оставляя сплошные поля зон."""
    h, w = bg.shape[:2]
    p = xy / sc
    bpx = max(2, int(round(bin_um / mpp / sc)))
    gw, gh = w // bpx + 1, h // bpx + 1
    uc = sorted({int(c) for c in cls})
    ci = {c: i for i, c in enumerate(uc)}
    counts = np.zeros((gh, gw, len(uc)), np.int32)
    bx = np.clip((p[:, 0] / bpx).astype(int), 0, gw - 1)
    by = np.clip((p[:, 1] / bpx).astype(int), 0, gh - 1)
    np.add.at(counts, (by, bx, np.array([ci[int(c)] for c in cls])), 1)
    tot = counts.sum(-1)
    maj = np.array(uc, np.uint8)[counts.argmax(-1)]
    ys, xs = np.where(tot > 0)
    if len(xs) == 0:
        return bg
    tree = cKDTree(np.c_[xs, ys])
    ally, allx = np.mgrid[0:gh, 0:gw]
    d, idx = tree.query(np.c_[allx.ravel(), ally.ravel()])
    binc = maj[ys, xs]
    maxbin = fill_um / mpp / sc / bpx
    zb = np.where(d <= maxbin, binc[idx], 0).reshape(gh, gw).astype(np.uint8)
    zone = np.asarray(Image.fromarray(zb).resize((w, h), Image.NEAREST))
    cmap = np.zeros((h, w, 3), np.uint8)
    for c, col in COL.items():
        cmap[zone == c] = col
    ov = bg.copy()
    m = zone > 0
    ov[m] = (alpha * cmap[m] + (1 - alpha) * bg[m]).astype(np.uint8)
    return ov


def open_tcga(args):
    from seg_infer import SlideReader
    return SlideReader(args.tcga, args.target_mpp, None)


def pick_region(sl, S):
    """Участок с максимумом ткани и цветового контраста (смесь опухоль+строма),
    по миниатюре среза. Возвращает (rx, ry) в рабочих координатах."""
    md = 1500
    tw = int(sl.W * md / max(sl.W, sl.H))
    th = int(sl.H * md / max(sl.W, sl.H))
    thumb_img = sl.thumbnail(tw, th)
    sct = sl.W / tw
    st = max(8, int(S / sct))
    best, bxy = -1.0, ((sl.W - S) // 2, (sl.H - S) // 2)
    xs = np.linspace(0, max(1, tw - st), 7).astype(int)
    ys = np.linspace(0, max(1, th - st), 7).astype(int)
    for ty in ys:
        for tx in xs:
            sub = thumb_img[ty:ty + st, tx:tx + st]
            if sub.size == 0:
                continue
            tissue = float((sub.mean(-1) < 220).mean())
            if tissue < 0.5:
                continue
            var = float(sub.reshape(-1, 3).std(0).mean())
            score = tissue * var
            if score > best:
                best, bxy = score, (int(tx * sct), int(ty * sct))
    return bxy


def panel_graph(args, device, l1, sd, normalize, uni, norm, out):
    """StarDist-ядра + граф на малом регионе TCGA, узлы окрашены слоем 1."""
    sl = open_tcga(args)
    S = min(args.graph_size, sl.W, sl.H)
    rx, ry = args.graph_region if args.graph_region else pick_region(sl, S)
    print(f"граф: регион x{rx} y{ry} {S}x{S}")
    xy, Fe, _ = scan_block(sl, sd, normalize, uni, norm, device, rx, ry, S, args.patch_size, False)
    if len(xy) < 20:
        print("мало ядер для графа — попробуй другой --graph-region"); return
    cls1 = l1_class(classify_l1(xy, Fe, l1, device, sl.mpp, args.knn, args.max_edge_um))
    he = sl.read(rx, ry, S)
    img = draw_graph(he, xy, rx, ry, cls1, sl.mpp, args.max_edge_um, scale=2)
    img, _ = thumb(img, args.fig_dim)
    Image.fromarray(img).save(out / "graph_stardist.png")
    print(f"graph_stardist.png  (ядер {len(xy)})")


def panel_d(args, device, l1, l2, sd, normalize, uni, norm, out):
    """Фрагмент TCGA: чистый H&E + карта зон."""
    sl = open_tcga(args)
    S = min(args.size, sl.W, sl.H)
    rx, ry = args.region if args.region else pick_region(sl, S)
    print(f"TCGA {sl.W}x{sl.H} @ {sl.mpp:.3f} мкм/px | регион x{rx} y{ry} {S}x{S}")
    xy, Fe, _ = scan_block(sl, sd, normalize, uni, norm, device, rx, ry, S, args.patch_size, False)
    print(f"ядер: {len(xy)}")
    if len(xy) < 40:
        raise SystemExit("мало ядер — выбери другой --region")
    cls, med = classify(sl, xy, Fe, l1, l2, device, args.knn, args.max_edge_um)
    he = sl.read(rx, ry, S)
    zone = fill_zones_solid(he, xy, cls, rx, ry, alpha=0.55)
    Image.fromarray(thumb(he, args.fig_dim)[0]).save(out / "d1_tcga_he.png")
    Image.fromarray(thumb(zone, args.fig_dim)[0]).save(out / "d2_zonemap.png")
    print("d1_tcga_he.png, d2_zonemap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", help="H&E обучающего среза (для панели c)")
    ap.add_argument("--slide", help="имя графа, напр. ovary2_he (для панели c)")
    ap.add_argument("--tcga", help="svs TCGA (для graph и панели d)")
    ap.add_argument("--layer1", required=True)
    ap.add_argument("--layer2", required=True)
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--zoom", type=int, default=2048)
    ap.add_argument("--whole", action="store_true", help="весь срез зонами по готовому графу (без StarDist)")
    ap.add_argument("--whole-dim", type=int, default=2600, help="макс. сторона обзора всего среза")
    ap.add_argument("--bin-um", type=float, default=30.0, help="размер ячейки сглаживания зон, мкм")
    ap.add_argument("--fill-um", type=float, default=110.0, help="радиус достройки пустых ячеек, мкм")
    ap.add_argument("--size", type=int, default=3072, help="сторона региона TCGA для панели d")
    ap.add_argument("--graph-size", type=int, default=768, help="малый регион для графа")
    ap.add_argument("--region", type=int, nargs=2, default=None)
    ap.add_argument("--graph-region", type=int, nargs=2, default=None)
    ap.add_argument("--target-mpp", type=float, default=0.27)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--fig-dim", type=int, default=1200)
    ap.add_argument("--out-dir", default="outputs/figure")
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)
    device = get_device()
    print("device:", device, "| гружу графовые модели...")
    l1 = load_gnn(args.layer1, device)
    l2 = load_gnn(args.layer2, device)

    if not ((args.he and args.slide) or args.tcga):
        raise SystemExit("нужен --he+--slide (панель c) и/или --tcga (graph, панель d)")

    # весь срез считается по готовому графу; StarDist/UNI нужны только для окна и TCGA
    need_star = (args.he and args.slide and not args.whole) or args.tcga
    sd = normalize = uni = norm = None
    if need_star:
        print("гружу StarDist, UNI, нормализатор...")
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize as _normalize
        from seg_infer import build_uni
        from stain_norm import MacenkoNormalizer
        normalize = _normalize
        sd = StarDist2D.from_pretrained("2D_versatile_he")
        uni = build_uni(device)
        norm = MacenkoNormalizer()

    if args.he and args.slide:
        if args.whole:
            panel_c_whole(args, device, l1, l2, out)
        else:
            panel_c(args, device, l1, l2, sd, normalize, uni, NoNorm(), out)  # окно — без Macenko
    if args.tcga:
        panel_graph(args, device, l1, sd, normalize, uni, norm, out)
        panel_d(args, device, l1, l2, sd, normalize, uni, norm, out)
    print("\nготово ->", out)


if __name__ == "__main__":
    main()