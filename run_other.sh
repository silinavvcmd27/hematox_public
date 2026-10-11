#!/bin/bash
# Правка макрофагов IGNITE и проверка третьего класса «прочее» за один заход:
#   1) маски IGNITE и BCSS заново (pan_relabel.py): прочее кодом 7, макрофаги в игнор;
#   2) из NCT-CRC детрит, слизь, жир и нормальная слизистая как прочее;
#   3) три модели, параллельно, если хватает памяти:
#      deploy_nomac  опухоль и строма на всех данных, как deploy, но без макрофагов;
#      deploy_nocrc  то же без NCT-CRC: на молочной железе ov_lung без кишки
#                    оказался лучше deploy, проверяем, не мешает ли разметка кишки
#                    целым патчем;
#      deploy3       как deploy_nomac плюс прочее;
#   4) все три размечают срез лёгкого, где deploy путал середину гнёзд, и рисуются
#      поля тех же мест, что в прошлый раз, рядом с deploy;
#   5) бенчмарк всех моделей на новой разметке.
#
#   nohup bash run_other.sh > outputs/results/pan/run_other.log 2>&1 &
set -u
LOG=outputs/results/pan
M=outputs/models
OVARY="ovary_prime_he ovary2_he ovary3_he"
stamp() { echo "== $1 | $(date '+%d.%m %H:%M')"; }

stamp "маски заново"
python -u pan_relabel.py --sets ignite_train ignite_test bcss > $LOG/relabel.log 2>&1 || exit 1
stamp "прочее из NCT-CRC"
python -u pan_prep_crc.py --root data/pan/crc/NCT-CRC-HE-100K-NONORM --name nct_crc_other \
  --classes DEB MUC ADI NORM --per-class 2000 > $LOG/prep_crc_other.log 2>&1 || exit 1
python -u pan_prep_crc.py --root data/pan/crc/CRC-VAL-HE-7K --name nct_crc_val_other \
  --classes DEB MUC ADI NORM > $LOG/prep_crc_val_other.log 2>&1 || exit 1

train() {
  name=$1
  shift
  python -u pan_train.py "$@" --select-on ignite_test3 --epochs 40 --patience 10 \
    --out $M/pan_l1_$name.pth > $LOG/pan_l1_$name.log 2>&1
}
jobs_nomac() { train deploy_nomac --train $OVARY nct_crc ignite_train3 --val nct_crc_val ignite_test3; }
jobs_nocrc() { train deploy_nocrc --train $OVARY ignite_train3 --val nct_crc_val ignite_test3; }
jobs_other() {
  train deploy3 --other --train $OVARY nct_crc nct_crc_other ignite_train3 \
    --val nct_crc_val nct_crc_val_other ignite_test3
}
# обучение держит в памяти все свои наборы, 25-35 ГБ на модель
free_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
stamp "обучение, свободной памяти ${free_gb} ГБ"
if [ "$free_gb" -ge 110 ]; then
  jobs_nomac & jobs_nocrc & jobs_other & wait
else
  jobs_nomac
  jobs_nocrc
  jobs_other
fi

stamp "срез лёгкого"
svs=$(ls data/pan/tcga_other/TCGA-38-4631*.svs)
name=$(basename "$svs" .svs)
for m in deploy_nomac deploy_nocrc deploy3; do
  mkdir -p $LOG/tcga_$m
  python -u seg_infer_svs.py --svs "$svs" --model $M/pan_l1_$m.pth \
    --stain-ref data/processed/stain_ref_prime.npz --target-mpp 0.2738 \
    --out "$LOG/tcga_$m/${name}_seg_map.png" > $LOG/tcga_$m.log 2>&1 &
done
wait
python pan_crops.py --svs "$svs" --npz $LOG/tcga_deploy/${name}_seg_map.npz \
  $LOG/tcga_deploy_nomac/${name}_seg_map.npz $LOG/tcga_deploy_nocrc/${name}_seg_map.npz \
  $LOG/tcga_deploy3/${name}_seg_map.npz \
  --out $LOG/tcga_deploy3/luad_crops_compare.png > $LOG/crops_compare.log 2>&1

stamp "бенчмарк"
python -u pan_bench.py --models $M/seg4_deploy.pth $M/pan_l1_ov2_only.pth $M/pan_l1_ovary_crc.pth \
  $M/pan_l1_ov_lung.pth $M/pan_l1_no_ovary.pth $M/pan_l1_deploy.pth $M/pan_l1_deploy_nomac.pth \
  $M/pan_l1_deploy_nocrc.pth $M/pan_l1_deploy3.pth \
  --sets nct_crc_val nct_crc_val_other ignite_test3 ovary2_he ovary_prime_he bcss3 \
  --out $LOG/bench3.csv > $LOG/bench3.log 2>&1
stamp "вся очередь готова"
