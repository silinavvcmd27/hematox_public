# ============================================================================
# ДОПОЛНЕНИЕ к make_panels_cd.py — две недостающие панели схемы:
#   c1_layer1_pred.png — карта опухоль/строма (выход слоя 1) на валидационном срезе
#   d2_nuclei.png      — ядра StarDist на фрагменте TCGA (панель d, шаг 13)
#
# Как применить: вставить обе функции в make_panels_cd.py после panel_c3,
# и добавить их вызовы в main() (см. блок «ВЫЗОВЫ» внизу файла).
# ============================================================================

def panel_c1(args, device, l1, l2, out):
    """Слой 1 на том же окне, что c3: опухоль красным, строма серым."""
    from src.data.patching import load_image
    from src.utils import TUMOR

    g = np.load(Path(args.graph_dir) / f"{args.slide}_graph.npz")
    pos, label, edges = g["pos"], g["label"], g["edges"]
    X = torch.from_numpy(g["feat"].astype(np.float32)).to(device)
    A = adjacency(edges, len(label), device)
    with torch.no_grad():
        p1 = l1(X, A).argmax(1).cpu().numpy()

    # 0 = опухоль, 1 = строма; красим в те же цвета, что легенда схемы
    cls1 = np.where(p1 == 0, TUMOR, 200).astype(np.uint8)   # 200 — служебный «строма»
    COL[200] = (150, 150, 150)

    img = load_image(args.he)
    H, W = img.shape[:2]
    z = args.zoom
    x0, y0 = pick_zoom(pos, label, W, H, z)     # то же окно, что у c3 — пара к нему
    crop = img[y0:y0 + z, x0:x0 + z]
    m = ((pos[:, 0] >= x0) & (pos[:, 0] < x0 + z) &
         (pos[:, 1] >= y0) & (pos[:, 1] < y0 + z))
    zxy = pos[m] - [x0, y0]
    Image.fromarray(dots(crop, zxy, cls1[m], r=6, outline=True)).save(
        out / "c1_layer1_pred.png")
    acc = float((p1[label >= 0] == (label[label >= 0] != TUMOR).astype(int)).mean())
    print(f"c1_layer1_pred.png  (окно {x0},{y0}, клеток {int(m.sum())}, "
          f"согласие с истиной по срезу {acc:.3f})")
    del img


def panel_d2_nuclei(args, device, out):
    """Ядра StarDist на фрагменте TCGA — вход графа, без классов."""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    from seg_infer import SlideReader
    from PIL import ImageDraw

    sd = StarDist2D.from_pretrained("2D_versatile_he")
    sl = SlideReader(args.tcga, args.target_mpp, None)
    S = min(args.nuclei_size, sl.W, sl.H)
    rx, ry = args.region if args.region else ((sl.W - S) // 2, (sl.H - S) // 2)
    he = sl.read(rx, ry, S)
    _, det = sd.predict_instances(normalize(he), prob_thresh=0.5, nms_thresh=0.3)
    pts = det["points"][:, ::-1]                       # (y,x) -> (x,y)
    im = Image.fromarray(he)
    dr = ImageDraw.Draw(im)
    for x, y in pts:
        dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(20, 20, 20), width=2)
    im.resize((args.fig_dim, args.fig_dim), Image.LANCZOS).save(out / "d2_nuclei.png")
    print(f"d2_nuclei.png  (регион x{rx} y{ry} {S}x{S}, ядер {len(pts)})")


# ------------------------------------------------------- ВЫЗОВЫ: правки в main()
# 1) добавить аргумент:
#      ap.add_argument("--nuclei-size", type=int, default=1024,
#                      help="сторона окна TCGA для d2_nuclei.png")
# 2) в конце main(), рядом с существующими вызовами:
#      if args.he and args.slide:
#          panel_c1(args, device, l1, l2, out)      # <- добавить
#          panel_c3(args, device, l1, l2, out)
#      if args.tcga:
#          panel_d(args, device, l1, l2, out)
#          panel_d2_nuclei(args, device, out)       # <- добавить
