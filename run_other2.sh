#!/bin/bash
# Последняя проверка перед выбором итоговой модели первого слоя. На молочной железе
# модели без опухоли и стромы NCT-CRC заметно лучше (deploy_nocrc против
# deploy_nomac), а класс «прочее» улучшает TSR и чинит середину гнёзд LUAD (deploy3).
# Здесь обе правки вместе, в двух вариантах:
#   deploy3_nocrc     три класса, яичник и IGNITE, кишки нет совсем;
#   deploy3_crcother  то же плюс «прочее» из NCT-CRC (детрит, слизь, жир, нормальная
#                     слизистая), но без её опухоли и стромы, размеченных целым фрагментом.
# Потом обе размечают срез лёгкого, рисуются те же поля, что в прошлый раз, и
# бенчмарк новых моделей рядом с deploy3 и deploy_nocrc.
#
#   nohup bash run_other2.sh > outputs/results/pan/run_other2.log 2>&1 &
set -u
LOG=outputs/results/pan
M=outputs/models
OVARY="ovary_prime_he ovary2_he ovary3_he"
stamp() { echo "== $1 | $(date '+%d.%m %H:%M')"; }

train() {
  name=$1
  shift
  python -u pan_train.py --other "$@" --val nct_crc_val nct_crc_val_other ignite_test3 \
    --select-on ignite_test3 --epochs 40 --patience 10 \
    --out $M/pan_l1_$name.pth > $LOG/pan_l1_$name.log 2>&1
}
stamp "обучение"
train deploy3_nocrc --train $OVARY ignite_train3 &
train deploy3_crcother --train $OVARY nct_crc_other ignite_train3 &
wait

stamp "срез лёгкого"
svs=$(ls data/pan/tcga_other/TCGA-38-4631*.svs)
name=$(basename "$svs" .svs)
for m in deploy3_nocrc deploy3_crcother; do
  mkdir -p $LOG/tcga_$m
  python -u seg_infer_svs.py --svs "$svs" --model $M/pan_l1_$m.pth \
    --stain-ref data/processed/stain_ref_prime.npz --target-mpp 0.2738 \
    --out "$LOG/tcga_$m/${name}_seg_map.png" > $LOG/tcga_$m.log 2>&1 &
done
wait
python pan_crops.py --svs "$svs" --npz $LOG/tcga_deploy/${name}_seg_map.npz \
  $LOG/tcga_deploy3/${name}_seg_map.npz $LOG/tcga_deploy3_nocrc/${name}_seg_map.npz \
  $LOG/tcga_deploy3_crcother/${name}_seg_map.npz \
  --out $LOG/tcga_deploy3_nocrc/luad_crops_compare.png > $LOG/crops_compare2.log 2>&1

stamp "бенчмарк"
python -u pan_bench.py --models $M/pan_l1_deploy_nocrc.pth $M/pan_l1_deploy3.pth \
  $M/pan_l1_deploy3_nocrc.pth $M/pan_l1_deploy3_crcother.pth \
  --sets nct_crc_val nct_crc_val_other ignite_test3 bcss3 \
  --out $LOG/bench4.csv > $LOG/bench4.log 2>&1
stamp "вся очередь готова"
