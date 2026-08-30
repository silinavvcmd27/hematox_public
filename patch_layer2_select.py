# Отбор лучшей эпохи по худшему классу вместо среднего.
import pathlib

p = pathlib.Path("cell_graph_train2.py")
s = p.read_text(encoding="utf-8")

edits = [
    ('        miou = float(np.mean(vals)) if vals else float("nan")\n'
     '        if miou > best:',
     '        miou = float(np.mean(vals)) if vals else float("nan")\n'
     '        # отбор по худшему классу: модель тут норовит схлопнуться в один\n'
     '        # класс, и у такой константы средний IoU выше, чем у слабой, но\n'
     '        # настоящей модели. По минимуму константа получает ноль и выбывает\n'
     '        score = float(np.min(vals)) if vals else float("nan")\n'
     '        if score > best:'),
    ('            print("epoch %3d  loss %.4f  IoU: %s  (mIoU %.3f)%s"\n'
     '                  % (ep, float(loss.detach()), per, miou, mark))',
     '            print("epoch %3d  loss %.4f  IoU: %s  (mIoU %.3f, min %.3f)%s"\n'
     '                  % (ep, float(loss.detach()), per, miou, score, mark))'),
    ('            print("early stop на", ep, "| лучший mIoU %.3f (эпоха %d)" % (best, best_ep))',
     '            print("early stop на", ep,\n'
     '                  "| лучший min IoU %.3f (эпоха %d)" % (best, best_ep))'),
]

for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"якорь встретился {s.count(old)} раз:\n{old[:70]}")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("правка внесена")
