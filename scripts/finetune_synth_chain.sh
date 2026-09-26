#!/bin/zsh
# Модуль 1 на синтетическом корпусе (раздел 5n CLAUDE.md): подбор lr с правилом
# края и финал seed 42 -> seed 43..46 с тем же lr -> перемешанные метки ->
# оценка каждой модели на дополнительных наборах через src.finetune.predict.
#
#   nohup caffeinate -i zsh scripts/finetune_synth_chain.sh > /dev/null 2>&1 < /dev/null & disown
#
# Имена логов — с меткой времени запуска каждого прогона: перезапуск пишет
# в новый файл и не затирает прежний (урок 5l). Статус цепочки — в
# analysis/finetune_logs/synth-chain-<время>.status.
#
# Все отчёты начинаются рамкой 5n: без --synthetic обучение и predict на
# data/synth/ отказываются работать. Числа на C печатаются с оговоркой об
# эффективном объёме (514 различных текстов из 1 200).
#
# Запускать ИЗ КОРНЯ репозитория. caffeinate -i не спасает от закрытия
# крышки ноутбука — время эпох тогда портится, числа нет.
set -u
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
L=analysis/finetune_logs; mkdir -p $L
S=$L/synth-chain-$(date +%Y%m%d-%H%M%S).status
# Переменные CHAIN_* — только для прогона цепочки в миниатюре перед большим
# прогоном; значения по умолчанию и есть большой прогон по 5n.
CSV=${CHAIN_CSV:-data/synth/train_eval.csv}
STEM=${CSV:t:r}
GRID=${CHAIN_GRID:-1e-5,2e-5,3e-5,5e-5}
EDGE=${CHAIN_EDGE:-8e-5}
PROBE=${CHAIN_PROBE:-5}
MODEL=xlm-roberta-base
COMMON=(--csv $CSV --text-col text --label-col label
        --split-col split --id-col id --model $MODEL --max-len 128
        --epochs ${CHAIN_EPOCHS:-15} --synthetic)
SEEDS=(${=CHAIN_SEEDS:-43 44 45 46})
EXTRA=(test_c test_c_masked seed_holdout seed_holdout_masked test_a_masked)

st()  { echo "$1 $(date '+%F %T')" >> $S; }
log() { echo "$L/synth-$1-$(date +%Y%m%d-%H%M%S).log"; }
# У прогона с перемешанными метками в имени есть «-shuffled» перед меткой.
last_run() { ls -td reports/finetune/$STEM-$MODEL*-$1-*(/) | head -1 | xargs basename; }

train() {                                   # train <метка> <аргументы…>
  local tag=$1; shift; local f=$(log $tag)
  st "старт $tag, лог $f"
  .venv/bin/python -m src.finetune.train $COMMON "$@" --tag $tag > $f 2>&1 \
    || { st "ОШИБКА $tag, лог $f"; exit 1; }
  st "конец $tag: $(last_run $tag)"
}

evaluate() {                                # evaluate <метка>: predict на EXTRA
  local tag=$1 run=$(last_run $1) set f
  for set in $EXTRA; do
    f=$(log $tag-$set)
    .venv/bin/python -m src.finetune.predict --model-dir models/finetune/$run \
      --csv data/synth/$set.csv --text-col text --id-col id --label-col label \
      --synthetic --output models/synth/module1/$run/$set.csv \
      --report reports/synth/module1/$run/$set.md > $f 2>&1 \
      || { st "ОШИБКА predict $tag $set, лог $f"; exit 1; }
  done
  st "оценка $tag на ${#EXTRA} наборах: reports/synth/module1/$run/"
}

train s42 --lr-grid $GRID --lr-edge $EDGE --lr-probe-epochs $PROBE --seed 42
evaluate s42
M=reports/finetune/$(last_run s42)/metrics.json
LR=$(.venv/bin/python -c "import json;print(json.load(open('$M'))['lr'])")
st "seed 42: lr=$LR, базовые линии — $M"

for s in $SEEDS; do
  train s$s --lr $LR --seed $s --baselines-from $M
  evaluate s$s
done

# Без --baselines-from: train.py не берёт базовые линии из прогона с другим
# режимом меток — на перемешанных метках они считаются заново.
train shuf --lr $LR --seed 42 --shuffle-labels
st "ГОТОВО"
