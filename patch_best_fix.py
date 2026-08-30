import pathlib
p = pathlib.Path("cell_graph_train2.py")
s = p.read_text(encoding="utf-8")
old = "            best, best_ep, bad = miou, ep, 0"
new = "            best, best_ep, bad = score, ep, 0"
if s.count(old) != 1:
    raise SystemExit(f"якорь встретился {s.count(old)} раз")
p.write_text(s.replace(old, new), encoding="utf-8")
print("исправлено")
