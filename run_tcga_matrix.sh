#!/bin/bash
# Разметка TCGA двумя графовыми слоями (опухоль, строма, матриксная строма),
# по одному срезу за раз, как run_tcga.sh. Срез пропускается, только если его
# карта свежее модели второго слоя.
#   bash run_tcga_matrix.sh 6
set -u
L1=runs/graph/l1.pth
L2=outputs/models/matrix_win.pth
REF=data/processed/stain_ref_prime.npz
OUT=outputs/results

for svs in $(ls -S -r data/tcga_ov_flat/*.svs | head -${1:-6}); do
  name=$(basename "$svs" .svs)
  map="$OUT/${name}_matrix_graph.npz"
  if [ -f "$map" ] && [ "$map" -nt "$L2" ]; then
    echo "== $name посчитан текущей моделью, пропускаю"
    continue
  fi
  echo "== $name | $(date '+%d.%m %H:%M')"
  python -u matrix_window_infer.py --slide "$svs" --layer1 "$L1" --layer2 "$L2" \
    --stain-ref "$REF" --target-mpp 0.2738
done
echo "== вся очередь готова | $(date '+%d.%m %H:%M')"
