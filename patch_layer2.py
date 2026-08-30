# Третий режим второго слоя: только гормональная и матриксная строма.
# Иммунные и прочие в карту не входят, а to_layer2 всё, чего нет в карте,
# и так помечает -100, то есть NA.
import pathlib

p = pathlib.Path("cell_graph_train2.py")
s = p.read_text(encoding="utf-8")

edits = [
    ('MERGE_CAF = False',
     '# режим --caf-only: только два типа CAF, иммунные и прочие уходят в NA\n'
     'L2_CAFONLY = {2: 0, 3: 1}\n'
     'CLASSES2_CAFONLY = ("hormonal", "matrix")\n\n'
     'MODE = "full"          # full | merge-caf | caf-only\n\n\n'
     'def _maps():\n'
     '    if MODE == "merge-caf":\n'
     '        return L2_CAF, CLASSES2_CAF\n'
     '    if MODE == "caf-only":\n'
     '        return L2_CAFONLY, CLASSES2_CAFONLY\n'
     '    return L2, CLASSES2'),
    ('    mp = L2_CAF if MERGE_CAF else L2',
     '    mp = _maps()[0]'),
    ('    return CLASSES2_CAF if MERGE_CAF else CLASSES2',
     '    return _maps()[1]'),
    ('    global MERGE_CAF\n    MERGE_CAF = args.merge_caf',
     '    global MODE\n'
     '    MODE = ("caf-only" if args.caf_only else\n'
     '            "merge-caf" if args.merge_caf else "full")'),
    ('    ap.add_argument("--out", required=True)',
     '    ap.add_argument("--caf-only", action="store_true",\n'
     '                    help="только гормональная и матриксная строма, '
     'иммунные и прочие в NA")\n'
     '    ap.add_argument("--out", required=True)'),
]

for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"якорь встретился {s.count(old)} раз, ожидался один:\n{old[:60]}")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("правка внесена")
