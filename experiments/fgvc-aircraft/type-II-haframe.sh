#!/bin/bash


DATASET=fgvc-aircraft
SEEDS=(0 1 2 3 4)
EPOCHS=100
BATCH_SIZE=64
NUM_WORKERS=16
OPTIMIZER=custom_sgd

MODEL=haframe_resnet50
POOLING=average
LOSS=mixed-ce-gscsl
GAMMA=1.0 # Gamma = 1.0 for fgvc-aircraft
ALPHA=0.5

for seed in "${SEEDS[@]}";
do
  output_dir=/home/ubuntu/out/${DATASET}/${ALPHA}-${LOSS}-${GAMMA}-${MODEL}-seed_${seed}

  # train
  python main.py \
  --start training \
  --arch ${MODEL} \
  --pool ${POOLING} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --loss ${LOSS} \
  --loss-schedule ${ALPHA} \
  --haf-gamma ${GAMMA} \
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
  --loss-schedule ${ALPHA} \
  --haf-gamma ${GAMMA} \
  --optimizer ${OPTIMIZER} \
  --data ${DATASET} \
  --workers ${NUM_WORKERS} \
  --output "${output_dir}" \
  --seed "${seed}"

done

# 5 runs evaluation on baseline model
python experiments/multiple_runs_eval.py --arch ${MODEL} --loss ${ALPHA}-${LOSS}-${GAMMA} \
--nseed ${#SEEDS[@]} --output /home/ubuntu/out/${DATASET}

