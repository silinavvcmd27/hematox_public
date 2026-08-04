# Вливает subtype_groups в groups: двухуровневая карта задумана в YAML,
# но CellTypeMapper читает только groups, из-за чего подробные подтипы
# (C7+ CAFs 1, apCAFs 1 и прочие) уходили в undefined.
#
# Подтип имеет приоритет: если имя есть и там и там, побеждает subtype_groups.
#
#   python merge_cell_map.py --in config/cell_type_map.yaml

import argparse
import shutil
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="config/cell_type_map.yaml")
    ap.add_argument("--out", default=None, help="по умолчанию — на место, с бэкапом")
    args = ap.parse_args()

    src = Path(args.src)
    cfg = yaml.safe_load(open(src))
    groups = {k: list(v or []) for k, v in (cfg.get("groups") or {}).items()}
    sub = {k: list(v or []) for k, v in (cfg.get("subtype_groups") or {}).items()}
    if not sub:
        raise SystemExit("в карте нет subtype_groups — сливать нечего")

    owner = {}                       # имя подтипа -> класс по subtype_groups
    for cls, names in sub.items():
        for n in names:
            owner[n.strip().lower()] = cls

    moved, added = [], 0
    for cls in list(groups):         # убрать имена, которые переопределяет подтип
        keep = []
        for n in groups[cls]:
            o = owner.get(n.strip().lower())
            if o is not None and o != cls:
                moved.append((n, cls, o))
            else:
                keep.append(n)
        groups[cls] = keep

    for cls, names in sub.items():
        groups.setdefault(cls, [])
        have = {n.strip().lower() for n in groups[cls]}
        for n in names:
            if n.strip().lower() not in have:
                groups[cls].append(n)
                have.add(n.strip().lower())
                added += 1

    cfg["groups"] = groups
    out = Path(args.out) if args.out else src
    if out == src:
        bak = src.with_suffix(src.suffix + ".bak")
        shutil.copy(src, bak)
        print("бэкап:", bak)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=200)

    print(f"добавлено имён в groups: {added}")
    for n, was, now in moved:
        print(f"  переопределено: {n!r}: {was} -> {now}")
    print("итог по классам:", {k: len(v) for k, v in groups.items()})
    print("записано:", out)


if __name__ == "__main__":
    main()
