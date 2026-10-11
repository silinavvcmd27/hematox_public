#!/bin/bash
# Внешняя проверка на яичнике по UBC-OCEAN: скачать маски и срезы, перевести в наш
# формат и прогнать бенчмарк. Перед запуском на сервере должен работать kaggle:
# pip install kaggle, ключ в ~/.kaggle/kaggle.json, правила конкурса UBC-OCEAN
# приняты на сайте. Скачанные срезы пропускаются, после обрыва можно запустить снова.
# Бенчмарк ждёт, пока run_other2.sh доучит свои модели, но не дольше 12 часов.
#
#   nohup bash run_ubc.sh > outputs/results/pan/run_ubc.log 2>&1 &
set -u
ROOT=data/pan/ubc
LOG=outputs/results/pan
M=outputs/models
stamp() { echo "== $1 | $(date '+%d.%m %H:%M')"; }
mkdir -p $ROOT/masks $ROOT/images

stamp "маски и таблица"
if [ -z "$(find $ROOT/masks -name '*.png' | head -n 1)" ]; then
  kaggle datasets download -d sohier/ubc-ovarian-cancer-competition-supplemental-masks \
    -p $ROOT/masks --unzip || exit 1
fi
if [ ! -f $ROOT/train.csv ]; then
  kaggle competitions download -c UBC-OCEAN -f train.csv -p $ROOT || exit 1
  if [ -f $ROOT/train.csv.zip ]; then
    unzip -q -o $ROOT/train.csv.zip -d $ROOT && rm $ROOT/train.csv.zip
  fi
fi

stamp "срезы"
n=0
for m in $(find $ROOT/masks -name '*.png' | sort); do
  id=$(basename "$m" .png)
  n=$((n + 1))
  [ -f $ROOT/images/$id.png ] && continue
  echo "  $n: $id | $(date '+%H:%M')"
  kaggle competitions download -c UBC-OCEAN -f train_images/$id.png -p $ROOT/images -q
  if [ -f $ROOT/images/$id.png.zip ]; then
    unzip -q -o $ROOT/images/$id.png.zip -d $ROOT/images && rm $ROOT/images/$id.png.zip
  fi
done

stamp "перевод в наш формат"
python -u pan_prep_ubc.py --root $ROOT --name ubc > $LOG/prep_ubc.log 2>&1 || exit 1

for i in $(seq 144); do
  grep -q "вся очередь готова" $LOG/run_other2.log 2>/dev/null && break
  sleep 300
done
stamp "бенчмарк"
python -u pan_bench.py --models $M/seg4_deploy.pth $M/pan_l1_no_ovary.pth $M/pan_l1_ov_lung.pth \
  $M/pan_l1_deploy.pth $M/pan_l1_deploy_nocrc.pth $M/pan_l1_deploy3.pth \
  $M/pan_l1_deploy3_nocrc.pth $M/pan_l1_deploy3_crcother.pth \
  --sets ubc --out $LOG/bench_ubc.csv > $LOG/bench_ubc.log 2>&1
stamp "вся очередь готова"
