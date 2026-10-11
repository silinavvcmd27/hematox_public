#!/bin/bash
# Восемь срезов TCGA других опухолей итоговой моделью первого слоя.
# Ждёт, пока run_pan_final.sh сохранит pan_l1_deploy (в его логе появится строка
# "сохранено"), потом идёт по срезам от меньшего файла к большему.
#
#   nohup bash run_tcga_deploy.sh > outputs/results/pan/tcga_deploy.log 2>&1 &
set -u
MODEL=outputs/models/pan_l1_deploy.pth
OUT=outputs/results/pan/tcga_deploy
mkdir -p "$OUT"

echo "== жду $MODEL | $(date '+%d.%m %H:%M')"
until grep -q "^сохранено" outputs/results/pan/pan_l1_deploy.log 2>/dev/null; do
  sleep 300
done

for svs in $(ls -S -r data/pan/tcga_other/*.svs); do
  name=$(basename "$svs" .svs)
  echo "== $name | $(date '+%d.%m %H:%M')"
  python -u seg_infer_svs.py --svs "$svs" --model "$MODEL" \
    --stain-ref data/processed/stain_ref_prime.npz --target-mpp 0.2738 \
    --patch-size 256 --stride 256 --out "$OUT/${name}_seg_map.png"
done
echo "== вся очередь готова | $(date '+%d.%m %H:%M')"
