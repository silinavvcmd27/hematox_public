#!/bin/bash
# Первый слой: что даёт колоректальный набор. Четыре прогона по очереди.
# Эпоха выбирается только по отложенному яичнику, CRC-VAL остаётся внешней
# проверкой, которую выбор эпохи не видел. Отложенный яичник по очереди
# замороженный (ovary2) и FFPE (ovary_prime, как TCGA).
#   nohup bash run_pan_l1.sh > /dev/null 2>&1 &
set -u
LOG=outputs/results/pan
mkdir -p "$LOG"

run() {
  name=$1; shift
  echo "== $name | $(date '+%d.%m %H:%M')"
  python -u pan_train.py "$@" --out "outputs/models/pan_l1_$name.pth" > "$LOG/pan_l1_$name.log" 2>&1
}

run ov2_only  --train ovary_prime_he ovary3_he         --val ovary2_he nct_crc_val --select-on ovary2_he
run ov2_crc   --train ovary_prime_he ovary3_he nct_crc --val ovary2_he nct_crc_val --select-on ovary2_he
run prime_only --train ovary2_he ovary3_he             --val ovary_prime_he nct_crc_val --select-on ovary_prime_he
run prime_crc  --train ovary2_he ovary3_he nct_crc     --val ovary_prime_he nct_crc_val --select-on ovary_prime_he
echo "== готово | $(date '+%d.%m %H:%M')"
