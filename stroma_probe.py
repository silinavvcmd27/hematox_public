# Решаема ли вообще задача деления стромы на подтипы по морфологии.
#
# Проверка идёт на уже посчитанных признаках UNI: ничего тяжелее линейного
# классификатора и k-средних не обучается. Отвечаем на четыре вопроса.
#
# 1. Потолок. Пробник внутри одного среза, обучение и проверка на разных патчах.
#    Верхняя граница того, что вообще лежит в признаках на этом материале.
# 2. Перенос. Обучение на двух срезах, проверка на третьем.
# 3. Чинится ли перенос простыми средствами. Тот же перенос считается ещё дважды:
#    с центрированием признаков внутри каждого среза и с выбрасыванием первых главных
#    компонент. И то и другое убирает общий сдвиг среза, не трогая метки. Если перенос
#    поднимается, различие между пациентками это техника, если нет, биология или метки.
# 4. Кластеры. Те же токены кластеризуются без меток, и для каждого k печатается,
#    насколько кластеры совпадают с подтипом и насколько просто с номером среза.
#
# Токен попадает в выборку, только если его блок 8 на 8 точек карты однороден по
# подтипу (доля не ниже --purity). Классы уравниваются по числу токенов.
#
# PCA считается на всех срезах сразу, но без меток, это допустимо: разметка проверочного
# среза нигде не используется.
#
# Память: X одного среза около 5 ГБ в float16, срезы читаются по одному.
#
#   python stroma_probe.py
#   python stroma_probe.py --purity 0.9 --max-per-class 6000

import os

# OpenBLAS на этой машине собран под 128 потоков и на k-средних падает, если ядер
# больше; переменная читается при импорте numpy, поэтому стоит до него.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HORMONAL, MATRIX = 2, 3
GRID, CELL = 14, 8


def token_labels(y, purity):
    n = len(y)
    blocks = (y.reshape(n, GRID, CELL, GRID, CELL)
               .transpose(0, 1, 3, 2, 4)
               .reshape(n, GRID * GRID, CELL * CELL))
    label = np.full((n, GRID * GRID), -1, np.int8)
    label[(blocks == HORMONAL).mean(2) >= purity] = 0
    label[(blocks == MATRIX).mean(2) >= purity] = 1
    return label


def sample_slide(path, purity, per_class, rng):
    d = np.load(path)
    label = token_labels(d["y"], purity)
    keep = []
    for cls in (0, 1):
        idx = np.argwhere(label == cls)
        if len(idx) > per_class:
            idx = idx[rng.choice(len(idx), per_class, replace=False)]
        keep.append(idx)
    keep = np.concatenate(keep)
    if not len(keep):
        return None
    keep = keep[np.argsort(keep[:, 0])]
    X = d["X"]
    feats = X[keep[:, 0], keep[:, 1]].astype(np.float32)
    del X
    return feats, label[keep[:, 0], keep[:, 1]].astype(int), keep[:, 0]


def fit_predict(train_x, train_y, test_x, test_y):
    fit = LogisticRegression(max_iter=2000, class_weight="balanced")
    fit.fit(train_x, train_y)
    pred = fit.predict(test_x)
    auc = roc_auc_score(test_y, fit.predict_proba(test_x)[:, 1]) if len(set(test_y)) > 1 else np.nan
    recall = [float((pred[test_y == c] == c).mean()) if (test_y == c).any() else np.nan
              for c in (0, 1)]
    return auc, balanced_accuracy_score(test_y, pred), recall


def embed(feats, slides, dim, center, drop):
    x = feats
    if center:
        x = feats.copy()
        for s in np.unique(slides):
            m = slides == s
            x[m] -= x[m].mean(0)
    z = PCA(dim, random_state=0).fit_transform(StandardScaler().fit_transform(x))
    return z[:, drop:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", default=["ovary_prime_he", "ovary2_he", "ovary3_he"])
    ap.add_argument("--dir", default="data/processed/seg")
    ap.add_argument("--purity", type=float, default=0.8)
    ap.add_argument("--max-per-class", type=int, default=8000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--drop-pcs", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--min-class", type=int, default=300)
    ap.add_argument("--out", default="outputs/results/pan/stroma_clusters.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    data = {}
    for name in args.slides:
        got = sample_slide(Path(args.dir) / f"{name}_feat.npz", args.purity,
                           args.max_per_class, rng)
        if got is None:
            print(f"{name}: чистых токенов нет, пропускаю")
            continue
        data[name] = got
        labels = got[1]
        print(f"{name}: токенов {len(labels)}, гормональная {(labels == 0).sum()}, "
              f"матриксная {(labels == 1).sum()}")

    feats = np.concatenate([d[0] for d in data.values()])
    labels = np.concatenate([d[1] for d in data.values()])
    slides = np.concatenate([[n] * len(d[1]) for n, d in data.items()])
    patches = np.concatenate([d[2] for d in data.values()])

    variants = [("как есть", False, 0),
                ("центрирование по срезу", True, 0),
                (f"центрирование и минус {args.drop_pcs} компонент", True, args.drop_pcs)]
    spaces = {name: embed(feats, slides, args.dim, center, drop)
              for name, center, drop in variants}

    z = spaces["как есть"]
    print("\nпотолок внутри среза (обучение и проверка на разных патчах)")
    for name in data:
        m = slides == name
        uniq = np.unique(patches[m])
        half = np.isin(patches, uniq[: len(uniq) // 2])
        tr, te = m & half, m & ~half
        if len(set(labels[tr])) < 2 or len(set(labels[te])) < 2:
            print(f"  {name}: одного из классов нет, пропуск")
            continue
        auc, bal, rec = fit_predict(z[tr], labels[tr], z[te], labels[te])
        print(f"  {name}: AUC {auc:.3f}, точность {bal:.3f}, "
              f"гормональная {rec[0]:.2f}, матриксная {rec[1]:.2f}")

    print("\nперенос на отложенный срез")
    for title in spaces:
        print(f"  {title}")
        for name in data:
            te = slides == name
            counts = [(labels[te] == c).sum() for c in (0, 1)]
            if min(counts) < args.min_class:
                print(f"    {name}: {counts[0]} гормональных и {counts[1]} матриксных "
                      f"токенов, проверять нечего")
                continue
            auc, bal, rec = fit_predict(spaces[title][~te], labels[~te],
                                        spaces[title][te], labels[te])
            print(f"    {name}: AUC {auc:.3f}, точность {bal:.3f}, "
                  f"гормональная {rec[0]:.2f}, матриксная {rec[1]:.2f}")

    first = list(data)[0]
    uniq = np.unique(patches)
    half = np.isin(patches, uniq[: len(uniq) // 2])
    _, bal, _ = fit_predict(z[half], (slides[half] == first).astype(int),
                            z[~half], (slides[~half] == first).astype(int))
    print(f"\nсрез {first} отличается от остальных с точностью {bal:.3f} на отложенных патчах")

    rows = []
    print("\nкластеризация без меток")
    print("  k   ARI с подтипом   ARI со срезом")
    for k in args.ks:
        km = KMeans(k, n_init=10, random_state=0).fit_predict(z)
        print(f"  {k}      {adjusted_rand_score(labels, km):+.3f}          "
              f"{adjusted_rand_score(slides, km):+.3f}")
        for c in range(k):
            m = km == c
            row = {"k": k, "кластер": c, "токенов": int(m.sum()),
                   "матриксная_%": round(100 * labels[m].mean(), 1)}
            for s in np.unique(slides):
                row[f"{s}_%"] = round(100 * (slides[m] == s).mean(), 1)
            rows.append(row)

    table = pd.DataFrame(rows)
    print("\nсостав кластеров")
    print(table.to_string(index=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
