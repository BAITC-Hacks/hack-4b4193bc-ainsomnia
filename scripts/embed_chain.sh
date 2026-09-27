#!/bin/zsh
# Модуль 2 целиком (раздел 5l CLAUDE.md): базовые линии -> подбор lr и финал
# seed 42 -> seed 43..46 с тем же lr -> правило трёх условий.
#
#   nohup caffeinate -i zsh scripts/embed_chain.sh > /dev/null 2>&1 < /dev/null & disown
#
# Опции (в любом порядке):
#   --baselines PATH       взять готовые базовые линии, не считать заново
#   --lr-search-from PATH  взять пробы lr, сохранённые остановленным прогоном
#
# ИМЕНА ЛОГОВ С МЕТКОЙ ВРЕМЕНИ ЗАПУСКА ПРОГОНА. Прежняя ad hoc-цепочка писала
# в embed-s42.log без метки, и перезапуск затёр лог прерванного прогона —
# его числа сохранились только в записи сессии (раздел 5l, «Известные
# ограничения»). Теперь перезапуск всегда пишет в новый файл.
#
# Запускать ИЗ КОРНЯ репозитория. caffeinate -i не спасает от закрытия
# крышки ноутбука — время эпох тогда портится, числа нет.
set -u
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
L=analysis/finetune_logs; mkdir -p $L
S=$L/embed-chain-$(date +%Y%m%d-%H%M%S).status
COMMON=(--csv data/external/massive.csv --text-col text --label-col label --split-col split --id-col id)
SEEDS=(43 44 45 46)
B=""; LS=""
while (( $# )); do
  case $1 in
    --baselines) B=$2; shift 2 ;;
    --lr-search-from) LS=$2; shift 2 ;;
    *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done

st()  { echo "$1 $(date '+%F %T')" >> $S; }
log() { echo "$L/embed-$1-$(date +%Y%m%d-%H%M%S).log"; }   # новое имя на каждый запуск
RUNS=()                                                    # metrics.json прогонов ЭТОЙ цепочки
run() {                                                    # run <метка> <аргументы…>
  local tag=$1; shift; local f=$(log $tag)
  st "старт $tag, лог $f"
  .venv/bin/python -m src.embed.train $COMMON "$@" > $f 2>&1 || { st "ОШИБКА $tag, лог $f"; exit 1; }
  st "конец $tag"
  [[ $tag == s* ]] && RUNS+=($(ls -td reports/embed/massive-multilingual-e5-small-$tag-*/metrics.json | head -1))
}

if [[ -z $B ]]; then
  run baselines --baselines-only --tag b
  B=$(ls -d reports/embed/massive-baselines-b-*/metrics.json | tail -1)
fi
st "базовые линии: $B"

if [[ -n $LS ]]; then
  run s42 --lr-search-from $LS --epochs 10 --seed 42 --tag s42 --baselines-from $B
else
  run s42 --lr-grid 1e-5,3e-5,1e-4,3e-4 --lr-edge 1e-3 --lr-probe-epochs 3 \
      --epochs 10 --seed 42 --tag s42 --baselines-from $B
fi
M=${RUNS[1]}
LR=$(.venv/bin/python -c "import json;print(json.load(open('$M'))['lr'])")
st "seed 42: lr=$LR"

for s in $SEEDS; do
  run s$s --lr $LR --epochs 10 --seed $s --tag s$s --baselines-from $B
done

# Решение — ровно по прогонам этой цепочки, а не по маске: иначе при повторном
# запуске в него попали бы и старые прогоны тех же seed.
.venv/bin/python -m src.embed.decide --baselines $B --runs $RUNS > $(log decision) 2>&1
st "ГОТОВО"
