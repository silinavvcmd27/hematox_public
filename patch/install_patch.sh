#!/usr/bin/env bash
# Установщик правок hematox. Запускать ИЗ КОРНЯ репозитория:
#     bash patch/install_patch.sh --check      # только посмотреть, ничего не менять
#     bash patch/install_patch.sh --apply      # применить
#
# Скрипт ничего не удаляет молча: сначала --check покажет, что найдено, а что
# нет. Откатиться можно всегда: если это git-репозиторий, правки идут в
# отдельную ветку; если нет — всё заменённое и удалённое складывается в папку
# hematox_backup_<дата>, и рядом кладётся restore.sh.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---check}"

# Файлы на удаление: заменены или не вызываются ниоткуда.
REMOVE=(
  seg_train.py
  make_seg_masks.py
  src/train.py
  src/inference.py
  src/build_graph.py
  src/models/graph_head.py
  seg_infer_img.py
  seg_infer_svs.py
  slide_mpp.py
  seg_threshold.py
  preview_slide.py
)

# Что куда положить: источник в patch/ -> место в репозитории.
declare -A COPY=(
  [cell_type_map.yaml]=config/cell_type_map.yaml
  [export_seurat.R]=R/export_seurat.R
  [stain_norm.py]=stain_norm.py
  [seg_infer.py]=seg_infer.py
  [check_stroma_split.py]=check_stroma_split.py
  [describe_slide.py]=describe_slide.py
)

# Файлы, в которые apply_edits.py вносит точечные правки.
EDITED=(src/utils.py tsr_regions.py make_seg_masks_v.py seg_train2.py scripts/run_pipeline.sh)

# Проверяем, что запуск из корня проекта: git может и не быть (например, папка
# скопирована на сервер без .git), но эти два файла быть обязаны.
for need in tsr_regions.py src/utils.py; do
  [ -f "$need" ] || { echo "ОШИБКА: в текущей папке нет $need."; \
    echo "Вы не в корне проекта. Перейдите туда, где лежит tsr_regions.py."; exit 1; }
done

if git rev-parse --git-dir >/dev/null 2>&1; then
  HAVE_GIT=1
  echo "git найден: правки пойдут в отдельную ветку, откат одной командой."
else
  HAVE_GIT=0
  echo "git в этой папке НЕТ. Всё заменённое и удалённое сложу в папку-бэкап."
fi
echo

echo "=========== ЧТО БУДЕТ УДАЛЕНО ==========="
missing_rm=0
for f in "${REMOVE[@]}"; do
  if [ -f "$f" ]; then printf "  удалить   %-30s %5s строк\n" "$f" "$(wc -l < "$f")"
  else printf "  НЕТ ФАЙЛА %-30s (пропущу)\n" "$f"; missing_rm=$((missing_rm+1)); fi
done

echo
echo "=========== ЧТО БУДЕТ ЗАПИСАНО ==========="
for src in "${!COPY[@]}"; do
  dst="${COPY[$src]}"
  [ -f "$HERE/$src" ] || { echo "  ОШИБКА: нет $HERE/$src"; exit 1; }
  if [ -f "$dst" ]; then printf "  заменить  %-30s (старый уйдёт в %s.bak)\n" "$dst" "$dst"
  else printf "  создать   %-30s\n" "$dst"; fi
done

echo
echo "=========== ТОЧЕЧНЫЕ ПРАВКИ (26 мест в 5 файлах) ==========="
for f in "${EDITED[@]}"; do
  [ -f "$f" ] && printf "  есть  %s\n" "$f" || printf "  НЕТ   %s  <-- проверьте имя файла\n" "$f"
done
echo
python "$HERE/apply_edits.py" || {
  echo
  echo "Правки не совпали с вашими файлами (см. выше). Ничего не изменено."
  exit 1
}

if [ "$MODE" != "--apply" ]; then
  echo
  echo "Это была проверка. Если список выше верный, запустите:"
  echo "    bash patch/install_patch.sh --apply"
  exit 0
fi

echo
echo "=========== ПРИМЕНЯЮ ==========="
BACKUP="hematox_backup_$(date +%Y%m%d_%H%M%S)"
branch="two-stroma-$(date +%Y%m%d)"

if [ "$HAVE_GIT" = 1 ]; then
  git rev-parse --verify "$branch" >/dev/null 2>&1 \
    && { echo "ветка $branch уже есть, переключаюсь"; git checkout "$branch"; } \
    || git checkout -b "$branch"
else
  mkdir -p "$BACKUP"
  echo "бэкап: $BACKUP/"
fi

# Удаляемые файлы не стираем, а переносим: их всегда можно вернуть.
for f in "${REMOVE[@]}"; do
  [ -f "$f" ] || continue
  if [ "$HAVE_GIT" = 1 ]; then
    git rm -q "$f" && echo "  удалён $f"
  else
    mkdir -p "$BACKUP/$(dirname "$f")"
    mv "$f" "$BACKUP/$f" && echo "  перенесён в бэкап: $f"
  fi
done

# Патч можно накатывать повторно. Поэтому: файл, который уже совпадает с
# новым, не трогаем вовсе, а .bak делаем ТОЛЬКО если его ещё нет — иначе
# второй запуск затёр бы бэкапом самого себя, и настоящий исходник пропал бы.
for src in "${!COPY[@]}"; do
  dst="${COPY[$src]}"
  mkdir -p "$(dirname "$dst")"
  if [ -f "$dst" ] && cmp -s "$HERE/$src" "$dst"; then
    echo "  уже такой же $dst"
    continue
  fi
  if [ -f "$dst" ]; then
    if [ -f "$dst.bak" ]; then
      echo "  бэкап $dst.bak уже есть, не перезаписываю"
    else
      cp "$dst" "$dst.bak" && echo "  бэкап $dst.bak"
    fi
    [ "$HAVE_GIT" = 1 ] || { mkdir -p "$BACKUP/$(dirname "$dst")"
                             [ -f "$BACKUP/$dst" ] || cp "$dst" "$BACKUP/$dst"; }
  fi
  cp "$HERE/$src" "$dst" && echo "  записан $dst"
done

# Старая таблица TSR: в ней дубли строк без ключа, новый формат несовместим.
if [ -f outputs/results/tsr_regions.csv ]; then
  mv outputs/results/tsr_regions.csv outputs/results/tsr_regions_old.csv
  echo "  outputs/results/tsr_regions.csv -> tsr_regions_old.csv"
fi

echo
echo "--- точечные правки ---"
python "$HERE/apply_edits.py" --apply || {
  echo "ОШИБКА при внесении правок. Файлы восстановить: for f in *.bak; do mv \"$f\" \"${f%.bak}\"; done"
  exit 1
}

if [ "$HAVE_GIT" = 1 ]; then
  # Бэкапы не коммитим: они нужны только для откатки на диске.
  grep -qx '*.bak' .gitignore 2>/dev/null || echo '*.bak' >> .gitignore
  grep -qx 'hematox_backup_*' .gitignore 2>/dev/null || echo 'hematox_backup_*' >> .gitignore
  git add -A
  git commit -q -m "Два типа стромы: гормональная и матриксная; нормализация окраски; один инференс" \
    && echo "  коммит сделан"
  ROLLBACK="git checkout - && git branch -D $branch"
else
  # Сохраняем и правленые файлы, и скрипт возврата.
  for f in "${EDITED[@]}"; do
    [ -f "$f.bak" ] || continue
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp "$f.bak" "$BACKUP/$f"
  done
  {
    echo '#!/usr/bin/env bash'
    echo '# Возврат к состоянию до правок. Запускать из корня проекта:'
    echo "#     bash $BACKUP/restore.sh"
    echo 'set -eu'
    echo "cd \"\$(dirname \"\${BASH_SOURCE[0]}\")/..\""
    echo "cp -a $BACKUP/. . 2>/dev/null || true"
    echo "rm -f $BACKUP/restore.sh"
    echo 'echo "файлы возвращены из бэкапа"'
  } > "$BACKUP/restore.sh"
  # сам restore.sh не должен копировать себя обратно
  ROLLBACK="bash $BACKUP/restore.sh"
  echo "  бэкап готов: $BACKUP/ (внутри restore.sh)"
fi

echo
echo "ГОТОВО. Дальше:"
echo "  1) Rscript R/export_seurat.R   — нужен ..._expr.csv для проверки стромы"
echo "  2) python check_stroma_split.py --cells ... --expr ... --out outputs/results/stroma_split"
echo "  3) посмотреть stroma_split_*.png ПРЕЖДЕ чем учить модель"
echo
echo "Откатить всё:  $ROLLBACK"
