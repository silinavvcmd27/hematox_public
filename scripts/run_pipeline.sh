#!/usr/bin/env bash
# Полный прогон пиксельной сегментации от аннотации Xenium до таблицы метрик.
# Список срезов берётся из config/slides.csv — чтобы добавить срез, допишите строку.
#
# bash scripts/run_pipeline.sh
#
# Шаги, где результат уже на диске, пропускаются, так что скрипт можно
# перезапускать после обрыва.

set -euo pipefail

: "${HF_TOKEN:?поставь токен: export HF_TOKEN=hf_xxx}"

MANIFEST=config/slides.csv
SEG_DIR=data/processed/seg12k     # отдельная папка: старые признаки не трогаем
CSV_DIR=data/seurat_csv
TAG=$(basename "$SEG_DIR")        # попадёт в имена моделей и таблицы метрик
PATCH_UM=70          # поле зрения патча; в пиксели пересчитывается под каждый срез
MAX_PATCHES=12000
AUGMENT=4            # 4 — повороты, 8 — повороты и отражения
EPOCHS=40
MASK_DOWNSCALE=2     # во сколько раз грубее среза строится маска
STRIDE_FRAC=2        # шаг окна = патч / STRIDE_FRAC; 2 = перекрытие 50%
MIN_CONF=0.5         # порог уверенности при инференсе

[ -f "$MANIFEST" ] || { echo "нет $MANIFEST"; exit 1; }

slides=()
while IFS=, read -r slide rds image alignment; do
    [ "$slide" = "slide" ] && continue
    [ -z "$slide" ] && continue
    slides+=("$slide")

    raw="$CSV_DIR/$(basename "${rds%.*}")_cells.csv"
    aligned="$CSV_DIR/${slide}_cells.csv"

    if [ ! -f "$aligned" ]; then
        if [ ! -f "$raw" ]; then
            [ -n "$rds" ] || { echo "$slide: нет ни $aligned, ни rds в манифесте"; exit 1; }
            echo "== $slide: выгрузка из Seurat"
            Rscript R/export_seurat.R "$rds"
        fi
        echo "== $slide: координаты в пиксели H&E"
        python align_xenium_he.py --cells "$raw" --align "$alignment" \
            --he "$image" --out "$aligned"
    fi
done < "$MANIFEST"

# Метки патчей для всех срезов сразу. Файлы перечисляем явно: без --cells
# скрипт заберёт по маске и невыровненные CSV, и слайды задвоятся.
cells=()
for s in "${slides[@]}"; do cells+=("$CSV_DIR/${s}_cells.csv"); done
python -m src.data.seurat_labels --cells "${cells[@]}"

# Проверка: разделяются ли два типа стромы по маркерам. Если нет — метки будут
# шумом, и учить модель бессмысленно. Смотреть outputs/results/stroma_split_*.png
for s in "${slides[@]}"; do
    [ -f "$CSV_DIR/${s}_expr.csv" ] || {
        echo "$s: нет ${s}_expr.csv — перезапустите Rscript R/export_seurat.R"; exit 1; }
    python check_stroma_split.py --cells "$CSV_DIR/${s}_cells.csv" \
        --expr "$CSV_DIR/${s}_expr.csv" \
        --out "outputs/results/stroma_split_${s}" || exit 1
done
echo "!! посмотрите outputs/results/stroma_split_*.png прежде чем учить модель"

# Маски и признаки — самая долгая часть, около часа на срез
while IFS=, read -r slide rds image alignment; do
    [ "$slide" = "slide" ] && continue
    [ -z "$slide" ] && continue

    ps=$(python -c "from src.utils import patch_px_for_um; print(patch_px_for_um('$image', $PATCH_UM))") \
        || { echo "$slide: не удалось определить мкм/px в $image"; exit 1; }
    echo "== $slide: патч $PATCH_UM мкм = $ps px"

    [ -f "$SEG_DIR/${slide}_mask.npz" ] || python -u make_seg_masks_v.py \
        --slide "$slide" --cells "$CSV_DIR/${slide}_cells.csv" --he "$image" \
        --patch-size "$ps" --stride "$((ps / STRIDE_FRAC))" \
        --downscale "$MASK_DOWNSCALE" --out-dir "$SEG_DIR"

    [ -f "$SEG_DIR/${slide}_feat.npz" ] || python -u seg_extract.py \
        --slide "$slide" --he "$image" --patch-um "$PATCH_UM" \
        --max-patches "$MAX_PATCHES" --augment "$AUGMENT" --seg-dir "$SEG_DIR"
done < "$MANIFEST"

# leave-one-slide-out: учим на всех срезах кроме одного, меряем на нём
for v in "${slides[@]}"; do
    python -u seg_train2.py --val "$v" --slides "${slides[@]}" \
        --select fixed --epochs "$EPOCHS" --bs 64 --seg-dir "$SEG_DIR" \
        --metrics-csv "outputs/results/seg2_${TAG}_metrics.csv" \
        --out "outputs/models/seg2_${TAG}_${v}.pth"
done

column -s, -t "outputs/results/seg2_${TAG}_metrics.csv"
