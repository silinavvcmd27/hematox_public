# Кто чем пользуется: разбор файлов проекта перед уборкой.
#
# Файл нужен, если его импортируют другие модули или он упомянут в README и в
# пайплайне. Всё остальное — кандидаты на удаление, но решение за человеком:
# скрипт мог остаться от разведки, а мог быть рабочим входом, о котором просто
# не написали.
#
#   python audit_files.py

import re
import subprocess
from pathlib import Path

SKIP_DIRS = {"headroom", ".git", "__pycache__", "data", "outputs", "runs", "patch"}
DOCS = ["README.md", "scripts/run_pipeline.sh"]


def collect():
    files = []
    for p in sorted(Path(".").rglob("*.py")) + sorted(Path(".").rglob("*.sh")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        files.append(p)
    return files


def git_age(path):
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "не в гите"
    except Exception:
        return "?"


def main():
    files = collect()
    modules = {p.stem: p for p in files if p.suffix == ".py"}

    text = {p: p.read_text(encoding="utf-8", errors="replace") for p in files}
    docs = ""
    for d in DOCS:
        if Path(d).exists():
            docs += Path(d).read_text(encoding="utf-8", errors="replace")

    imported_by = {name: set() for name in modules}
    for p, body in text.items():
        for name in modules:
            if modules[name] == p:
                continue
            pattern = rf"(?:^|\n)\s*(?:from|import)\s+(?:[\w.]*\.)?{re.escape(name)}\b"
            if re.search(pattern, body):
                imported_by[name].add(p.name)

    rows = []
    for p in files:
        name = p.stem
        users = sorted(imported_by.get(name, ()))
        in_docs = p.name in docs
        if users and in_docs:
            kind = "ядро"
        elif users:
            kind = "модуль"
        elif in_docs:
            kind = "вход"
        else:
            kind = "сирота"
        rows.append((kind, str(p), len(text[p].splitlines()), git_age(p), users))

    order = {"ядро": 0, "вход": 1, "модуль": 2, "сирота": 3}
    rows.sort(key=lambda r: (order[r[0]], r[1]))

    print("%-8s %-26s %6s %-12s %s" % ("роль", "файл", "строк", "правился", "кто импортирует"))
    for kind, path, lines, age, users in rows:
        print("%-8s %-26s %6d %-12s %s"
              % (kind, path[:26], lines, age, ", ".join(users)[:44]))

    orphans = [r for r in rows if r[0] == "сирота"]
    print(f"\nвсего файлов: {len(rows)}, из них сирот: {len(orphans)}")
    print("сироты никем не импортируются и не упомянуты в README или пайплайне")

    baks = sorted(Path(".").rglob("*.bak"))
    baks = [b for b in baks if not any(part in SKIP_DIRS for part in b.parts)]
    if baks:
        print("\nрезервные копии от прежних правок:")
        for b in baks:
            print(" ", b)


if __name__ == "__main__":
    main()
