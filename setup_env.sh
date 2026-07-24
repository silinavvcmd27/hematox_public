#!/usr/bin/env bash
# создаёт conda-окружение hematox и ставит всё нужное (CPU)
# запуск из корня проекта: bash setup_env.sh
set -e

ENV=hematox

if command -v conda >/dev/null 2>&1; then
  echo ">> создаю conda env '$ENV' из environment.yml"
  conda env create -f environment.yml || conda env update -f environment.yml
  echo
  echo ">> готово. активируй:"
  echo "   conda activate $ENV"
else
  # фолбэк на venv, если conda нет
  echo ">> conda не найден, делаю venv"
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
  echo ">> готово. активируй: source .venv/bin/activate"
fi

echo
echo ">> проверка после активации:"
echo "   python -c 'import torch, torch_geometric, timm; print(torch.__version__, \"geom ok\")'"
