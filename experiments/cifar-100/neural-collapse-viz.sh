#!/bin/bash


DATASET=cifar-100
seed=0
EPOCHS=100
BATCH_SIZE=64
NUM_WORKERS=16
OPTIMIZER=custom_sgd

# neural collapse
CKPT_FREQ=5 # checkpoint for every 5 epochs
PARTITION=train


# ETF-collapse with type-I baseline ---------------------------------------------------------------------------------- #
MODEL=wide_resnet
LOSS=cross-entropy
FRAME=ETF
etf_dir=/home/ubuntu/out/${DATASET}/${LOSS}-${MODEL}-NC-seed_${seed}

# train
python main.py \
--start training \
--arch ${MODEL} \
--batch-size ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--loss ${LOSS} \
--ckpt-freq ${CKPT_FREQ} \
--optimizer ${OPTIMIZER} \
--data ${DATASET} \
--workers ${NUM_WORKERS} \
--output "${etf_dir}" \
--seed "${seed}"

# viz
python main.py \
--start neural-collapse \
--arch ${MODEL} \
--batch-size ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--loss ${LOSS} \
--ckpt-freq ${CKPT_FREQ} \
--optimizer ${OPTIMIZER} \
--data ${DATASET} \
--workers ${NUM_WORKERS} \
--output "${etf_dir}" \
--seed "${seed}" \
--frame ${FRAME} \
--partition ${PARTITION}


# HAF-collapse with type-II model ------------------------------------------------------------------------------------ #
MODEL=haframe_wide_resnet
LOSS=mixed-ce-gscsl
GAMMA=2.0
ALPHA=0.4
FRAME=HAF
haf_dir=/home/ubuntu/out/${DATASET}/${ALPHA}-${LOSS}-${GAMMA}-${MODEL}-NC-seed_${seed}

# train
python main.py \
--start training \
--arch ${MODEL} \
--batch-size ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--loss ${LOSS} \
--loss-schedule ${ALPHA} \
--haf-gamma ${GAMMA} \
--ckpt-freq ${CKPT_FREQ} \
--optimizer ${OPTIMIZER} \
--data ${DATASET} \
--workers ${NUM_WORKERS} \
--output "${haf_dir}" \
--seed "${seed}"

# viz
python main.py \
--start neural-collapse \
--arch ${MODEL} \
--batch-size ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--loss ${LOSS} \
--loss-schedule ${ALPHA} \
--haf-gamma ${GAMMA} \
--ckpt-freq ${CKPT_FREQ} \
--optimizer ${OPTIMIZER} \
--data ${DATASET} \
--workers ${NUM_WORKERS} \
--output "${haf_dir}" \
--seed "${seed}" \
--frame ${FRAME} \
--partition ${PARTITION}


# plot ETF vs HAFrame ------------------------------------------------------------------------------------------------ #
etf_path=${etf_dir}/${PARTITION}_penultimate_feature_neural_collapse.pkl
haf_path=${haf_dir}/${PARTITION}_penultimate_feature_neural_collapse.pkl

python experiments/haframe_vs_etf_viz.py \
--etf-path ${etf_path} \
--haframe-path ${haf_path} \
--partition ${PARTITION} \
--epochs ${EPOCHS} \
--ckpt-freq ${CKPT_FREQ} \
--output /home/ubuntu/out/${DATASET}

