# BCSS (Amgad и др., 2019, CC0): 151 область из срезов TCGA-BRCA при 0.25 мкм/px,
# картинки и маски лежат на Google Drive. Папку целиком Drive отдаёт только
# браузеру, а gdown упирается в лимит 50 файлов, поэтому список берётся со
# страницы встраиваемого вида папки и каждый файл качается по своему id.
# Скачанное пропускается, после обрыва можно просто запустить ещё раз.
#
#   python -u pan_get_bcss.py --out data/pan/bcss

import argparse
import re
import time
import urllib.request
from pathlib import Path

FOLDERS = {"rgbs_colorNormalized": "1cMvgHRIetLR1C_arBDUHa4iW_mddU2X3",
           "masks": "1uYg1I1gvL_v6w1N0qq7-s7opr5lzsSA1",
           "meta": "1e-mr3zqaEeOVQQLgWEgSL8dIgAdsclnl"}
ENTRY = re.compile(r'id="entry-([\w-]+)".*?flip-entry-title">([^<]+)<', re.S)


def list_folder(folder_id):
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return ENTRY.findall(r.read().decode("utf-8"))


def fetch(file_id, dest):
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    with urllib.request.urlopen(url, timeout=600) as r:
        if r.headers.get_content_type() == "text/html":
            raise OSError("Drive отдал страницу вместо файла, похоже на лимит скачиваний")
        body = r.read()
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pan/bcss")
    args = ap.parse_args()

    failed = 0
    for sub, folder_id in FOLDERS.items():
        dest_dir = Path(args.out) / sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = list_folder(folder_id)
        if not files:
            raise SystemExit(f"в папке {sub} не нашлось файлов: Drive поменял страницу или закрыл доступ")
        todo = [(fid, name) for fid, name in files if not (dest_dir / name).exists()]
        print(f"{sub}: файлов {len(files)}, скачать {len(todo)}")
        for i, (fid, name) in enumerate(todo, 1):
            for attempt in range(3):
                try:
                    fetch(fid, dest_dir / name)
                    break
                except OSError as err:
                    print(f"\n  {name}: {err}")
                    time.sleep(30 * (attempt + 1))
            else:
                failed += 1
            print(f"  {i}/{len(todo)}", end="\r")
        print()
    if failed:
        raise SystemExit(f"не скачалось файлов: {failed}, запустите ещё раз")


if __name__ == "__main__":
    main()
