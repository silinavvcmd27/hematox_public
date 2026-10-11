#!/bin/bash
# Первый слой: финальная модель и проверки с исключением органа целиком.
#   ov_lung    яичники + лёгкое, кишка не видна: проверка на CRC-VAL
#   deploy     всё сразу, модель для TCGA; эпоха по отложенной части IGNITE
#   no_ovary   кишка + лёгкое, яичник не виден: проверка на обоих срезах яичника
# Прогоны идут по очереди; третий можно остановить, если торопимся.
#   nohup bash run_pan_final.sh > /dev/null 2>&1 &
set -u
LOG=outputs/results/pan
mkdir -p "$LOG"

run() {
  name=$1; shift
  python -u pan_train.py "$@" --out "outputs/models/pan_l1_$name.pth" > "$LOG/pan_l1_$name.log" 2>&1
}

run ov_lung  --train ovary_prime_he ovary3_he ignite_train \
             --val ovary2_he nct_crc_val ignite_test --select-on ovary2_he
run deploy   --train ovary_prime_he ovary2_he ovary3_he nct_crc ignite_train \
             --val nct_crc_val ignite_test --select-on ignite_test
run no_ovary --train nct_crc ignite_train \
             --val ovary2_he ovary_prime_he nct_crc_val ignite_test --select-on ignite_test
