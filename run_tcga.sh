#!/bin/bash
# Пакетная разметка срезов TCGA, строго по одному: каждый держит UNI в памяти.
# Пропускаем только то, что посчитано текущей моделью: файлы старше её остались
# от прежних прогонов, до нормализации окраски, и в когорту их брать нельзя.
set -u
MODEL=outputs/models/seg3_deploy.pth
REF=data/processed/stain_ref_prime.npz
OUT=outputs/results

for svs in $(ls -S -r data/tcga_ov_flat/*.svs | head -${1:-6}); do
  name=$(basename "$svs" .svs)
  map="$OUT/${name}_seg_map.npz"
  if [ -f "$map" ] && [ "$map" -nt "$MODEL" ]; then
    echo "== $name посчитан текущей моделью, пропускаю"
    continue
  fi
  echo "== $name | $(date '+%d.%m %H:%M')"
  python -u seg_infer_svs.py --svs "$svs" --model "$MODEL" --stain-ref "$REF" \
    --target-mpp 0.2738 --patch-size 256 --stride 256 --no-prob
done
echo "== вся очередь готова | $(date '+%d.%m %H:%M')"
