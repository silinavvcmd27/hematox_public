#!/bin/bash
# Итоговый пересчёт моделью deploy3: восемь срезов TCGA других опухолей, потом вся
# когорта TCGA-OV, потом сводная таблица. Срезы считаются по $1 штук разом (по
# умолчанию 4): один срез занимает от полутора до трёх часов, а ядер 128.
# Посчитанные срезы пропускаются, так что после обрыва можно запустить снова.
#
#   nohup bash run_final.sh 4 > outputs/results/pan/run_final.log 2>&1 &
set -u
JOBS=${1:-4}
MODEL=outputs/models/pan_l1_deploy3.pth
REF=data/processed/stain_ref_prime.npz
stamp() { echo "== $1 | $(date '+%d.%m %H:%M')"; }

one() {
  svs=$1
  out=$2
  name=$(basename "$svs" .svs)
  [ -f "$out/${name}_seg_map.npz" ] && return
  python -u seg_infer_svs.py --svs "$svs" --model $MODEL --stain-ref $REF \
    --target-mpp 0.2738 --patch-size 256 --stride 256 \
    --out "$out/${name}_seg_map.png" > "$out/${name}.log" 2>&1
  echo "   готов $name | $(date '+%d.%m %H:%M')"
}

queue() {
  out=$1
  shift
  mkdir -p "$out"
  for svs in "$@"; do
    one "$svs" "$out" &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 20; done
  done
  wait
}

stamp "восемь срезов TCGA"
queue outputs/results/pan/tcga_final data/pan/tcga_other/*.svs
stamp "когорта TCGA-OV"
queue outputs/results/pan/tcga_ov_deploy3 data/tcga_ov_flat/*.svs
stamp "сводка"
python -u pan_collect_tsr.py --dir outputs/results/pan/tcga_ov_deploy3 \
  > outputs/results/pan/cohort_deploy3.log 2>&1
stamp "вся очередь готова"
