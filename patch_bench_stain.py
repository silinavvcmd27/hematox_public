# Приведение окраски тайлов STHELAR к нашему эталону.
# Матрица красителей считается по объединённым пикселям тайлов одного среза:
# на одном кадре 256 px оценка неустойчива, а по срезу в целом надёжна.
import pathlib

p = pathlib.Path("sthelar_bench.py")
s = p.read_text(encoding="utf-8")

anchor_loop = "    for i in shards:"
anchor_decode = "            img = decode_img(img_col[j].as_py())"
anchor_main = "def main():"

for a in (anchor_loop, anchor_decode, anchor_main):
    if s.count(a) != 1:
        raise SystemExit("якорь встретился %d раз:\n%s" % (s.count(a), a[:60]))

s = s.replace(anchor_loop,
              "    norms = fit_stains(shards, args) if args.stain_ref else {}\n\n"
              + anchor_loop)

s = s.replace(anchor_decode,
              anchor_decode + "\n"
              "            if s in norms:\n"
              "                img = norms[s](img)")

s = s.replace(anchor_main,
              'def fit_stains(shards, args):\n'
              '    """Окраска каждого среза по его же тайлам. Отдельный быстрый проход:\n'
              '    картинки декодируются, UNI не запускается."""\n'
              '    from src.stain import Normalizer, StainStats, fit_stats, sample_pixels\n'
              '\n'
              '    pool = {s: [] for s in SLIDES}\n'
              '    for i in shards:\n'
              '        if all(len(pool[s]) >= args.stain_tiles for s in SLIDES):\n'
              '            break\n'
              '        fn = f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"\n'
              '        try:\n'
              '            path = hf_hub_download(REPO, fn, repo_type="dataset")\n'
              '        except Exception:\n'
              '            continue\n'
              '        tbl = pq.read_table(path)\n'
              '        sid = tbl.column("slide_id").to_pylist()\n'
              '        col = tbl.column("image")\n'
              '        for j, sl in enumerate(sid):\n'
              '            if sl in SLIDES and len(pool[sl]) < args.stain_tiles:\n'
              '                pool[sl].append(decode_img(col[j].as_py()))\n'
              '\n'
              '    ref = StainStats.load(args.stain_ref)\n'
              '    out = {}\n'
              '    for sl, tiles in pool.items():\n'
              '        if len(tiles) < 5:\n'
              '            print(f"{sl}: тайлов на оценку окраски мало, оставляю как есть")\n'
              '            continue\n'
              '        px = np.concatenate([t.reshape(-1, 3) for t in tiles], 0)\n'
              '        src = fit_stats(sample_pixels(px))\n'
              '        print(f"{sl}: окраска по {len(tiles)} тайлам,",\n'
              '              np.round(src.he.T, 3).tolist())\n'
              '        out[sl] = Normalizer(src, ref)\n'
              '    return out\n'
              '\n'
              '\n' + anchor_main)

lines = s.split("\n")
idx = next(i for i, l in enumerate(lines) if l.strip() == "args = ap.parse_args()")
lines.insert(idx, '    ap.add_argument("--stain-ref", default=None,\n'
                  '                    help="npz эталона окраски, тот же что при seg_extract.py")\n'
                  '    ap.add_argument("--stain-tiles", type=int, default=40,\n'
                  '                    help="сколько тайлов брать на оценку окраски среза")')
s = "\n".join(lines)

p.write_text(s, encoding="utf-8")
print("правка внесена")
