#!/bin/zsh
# Модуль 1 на синтетике: перенос между языками (раздел 5n, проверка 6).
# Обучение только на русском -> тест на русском и на казахском; обучение только
# на казахском (kk + kk-ru, как в 5n) -> тест на казахском и на русском.
# Эпоха выбирается по валидации ИСХОДНОГО языка: целевой язык в выборе не
# участвует. lr 5e-5 взят из основного прогона модуля 1 (подбирался на
# смешанной валидации — связь с целевым языком слабая, но она есть).
#
#   nohup caffeinate -i zsh scripts/xling_chain.sh > /dev/null 2>&1 < /dev/null & disown
#
# Входные CSV — data/synth/xling/{ru,kk,test_ru,test_kk}.csv, собираются из
# data/synth/train_eval.csv (см. src/synth/xling_report.py). Имена логов — с
# меткой времени запуска каждого прогона; статус — synth-xling-<время>.status.
# Переменные XLING_* — только для прогона в миниатюре перед большим.
set -u
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
L=analysis/finetune_logs; mkdir -p $L
S=$L/synth-xling-$(date +%Y%m%d-%H%M%S).status
MODEL=xlm-roberta-base
X=data/synth/xling
SEEDS=(${=XLING_SEEDS:-42 43 44 45 46})
COMMON=(--text-col text --label-col label --split-col split --id-col id
        --model $MODEL --max-len 128 --lr 5e-5 --epochs ${XLING_EPOCHS:-15}
        --synthetic)

st()  { echo "$1 $(date '+%F %T')" >> $S; }
log() { echo "$L/synth-xling-$1-$(date +%Y%m%d-%H%M%S).log"; }
last_run() { ls -td reports/finetune/$1-$MODEL-$2-*(/) | head -1 | xargs basename; }

for src in ru kk; do
  other=$([[ $src == ru ]] && echo kk || echo ru)
  BF=()   # массив: в zsh ${B:+--флаг $B} не делится на слова
  for s in $SEEDS; do
    tag=x$src-s$s; f=$(log $tag)
    st "старт $tag, лог $f"
    .venv/bin/python -m src.finetune.train --csv $X/$src.csv $COMMON --seed $s \
      --tag $tag $BF > $f 2>&1 || { st "ОШИБКА $tag, лог $f"; exit 1; }
    run=$(last_run $src $tag)
    st "конец $tag: $run"
    (( ${#BF} )) || BF=(--baselines-from reports/finetune/$run/metrics.json)
    f=$(log $tag-test_$other)
    .venv/bin/python -m src.finetune.predict --model-dir models/finetune/$run \
      --csv $X/test_$other.csv --text-col text --id-col id --label-col label \
      --synthetic --output models/synth/xling/$run/test_$other.csv \
      --report reports/synth/xling/$run/test_$other.md > $f 2>&1 \
      || { st "ОШИБКА predict $tag, лог $f"; exit 1; }
    st "перенос $tag -> $other: reports/synth/xling/$run/"
  done
done
st "ГОТОВО"
