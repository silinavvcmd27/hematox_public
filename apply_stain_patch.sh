#!/bin/bash
# Нормализация окраски в скрипты инференса: импорт, параметр norm в run_batch,
# ключ --stain-ref. Правка идёт по месту, поверх текущей версии файла.
set -u

patch_one () {
  f=$1; src=$2
  grep -q "slide_normalizer" "$f" && { echo "$f: уже пропатчен"; return; }

  sed -i 's|^from src.utils import \(.*\)$|from src.utils import \1\nfrom src.stain import slide_normalizer|' "$f"
  sed -i 's|^def run_batch(uni, dec, arrs, device):$|def run_batch(uni, dec, arrs, device, norm=None):\n    if norm is not None:\n        arrs = [norm(a) for a in arrs]|' "$f"
  sed -i 's|run_batch(uni, dec, arrs, device)|run_batch(uni, dec, arrs, device, norm)|' "$f"
  sed -i "s|^    args = ap.parse_args()\$|    ap.add_argument(\"--stain-ref\", default=None,\n                    help=\"npz эталона окраски, тот же что при seg_extract.py\")\n    args = ap.parse_args()\n    norm = slide_normalizer(args.$src, args.stain_ref) if args.stain_ref else None|" "$f"

  for pat in "from src.stain import slide_normalizer" "device, norm=None" "device, norm)" "stain-ref"; do
    grep -q -- "$pat" "$f" || { echo "$f: НЕ ЛЁГ фрагмент [$pat]"; return; }
  done
  echo "$f: ок"
}

patch_one seg_infer.py he
patch_one seg_infer_img.py he
patch_one seg_infer_svs.py svs
patch_one seg_eval_slide.py he
