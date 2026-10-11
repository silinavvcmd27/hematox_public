import pathlib

p = pathlib.Path("sthelar_bench.py")
s = p.read_text(encoding="utf-8")

old_acc = ('            MIXCONF.setdefault(key, np.zeros((2, 2), np.int64))')
new_acc = ('            key = f"{s}, {key}"\n'
           '            MIXCONF.setdefault(key, np.zeros((2, 2), np.int64))')
old_rep = ('    for key in ("однородные", "смешанные"):\n'
           '        if key in MIXCONF:\n'
           '            report_l1(MIXCONF[key], "тайлы " + key)')
new_rep = ('    for key in sorted(MIXCONF):\n'
           '        report_l1(MIXCONF[key], "тайлы " + key)')

for a in (old_acc, old_rep):
    if s.count(a) != 1:
        raise SystemExit("якорь встретился %d раз:\n%s" % (s.count(a), a[:60]))

p.write_text(s.replace(old_acc, new_acc).replace(old_rep, new_rep), encoding="utf-8")
print("правка внесена")
