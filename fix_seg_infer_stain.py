import pathlib

p = pathlib.Path("seg_infer.py")
s = p.read_text(encoding="utf-8")

edits = [
    ("from stain_norm import MacenkoNormalizer\n", ""),
    ('    ap.add_argument("--stain-norm", choices=["macenko", "none"], default="macenko")\n'
     '    ap.add_argument("--stain-ref", default=None,\n'
     '                    help="картинка-эталон окраски; без неё берётся табличный эталон")\n'
     '    ap.add_argument("--out", default=None)\n',
     '    ap.add_argument("--out", default=None)\n'),
    ('    norm = None\n'
     '    if args.stain_norm == "macenko":\n'
     '        norm = MacenkoNormalizer()\n'
     '        if args.stain_ref:\n'
     '            norm.fit(np.asarray(Image.open(args.stain_ref).convert("RGB")))\n'
     '            print("эталон окраски:", args.stain_ref)\n'
     '        else:\n'
     '            print("эталон окраски: табличный (Macenko)")\n\n', ""),
    ("stain_norm=args.stain_norm", 'stain_norm=args.stain_ref or "none"'),
]

for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit("якорь встретился %d раз:\n%s" % (s.count(old), old[:60]))
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("правка внесена")
