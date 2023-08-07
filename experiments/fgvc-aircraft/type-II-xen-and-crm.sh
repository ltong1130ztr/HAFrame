#!/bin/bash


DATASET=fgvc-aircraft
SEEDS=(0 1 2 3 4)
EPOCHS=100
BATCH_SIZE=64
NUM_WORKERS=16
OPTIMIZER=custom_sgd

MODEL=haframe_resnet50
POOLING=average
LOSS=cross-entropy

for seed in "${SEEDS[@]}";
do
  output_dir=/home/ubuntu/out/${DATASET}/${LOSS}-${MODEL}-seed_${seed}

  # train
  python main.py \
  --start training \
  --arch ${MODEL} \
  --pool ${POOLING} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --loss ${LOSS} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}"

  # test
  python main.py \
  --start testing \
  --arch ${MODEL} \
  --pool ${POOLING} \
  --loss ${LOSS} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}"

  # apply CRM
  python main.py \
  --start crm \
  --arch ${MODEL} \
  --pool ${POOLING} \
  --loss ${LOSS} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}"

done

# 5 runs evaluation on baseline model
python experiments/multiple_runs_eval.py --arch ${MODEL} --loss ${LOSS} \
--nseed ${#SEEDS[@]} --output /home/ubuntu/out/${DATASET}

# 5 runs evaluation on CRM of baseline model
python experiments/multiple_runs_eval.py --arch ${MODEL} --loss ${LOSS} \
--nseed ${#SEEDS[@]} --output /home/ubuntu/out/${DATASET} --crm
