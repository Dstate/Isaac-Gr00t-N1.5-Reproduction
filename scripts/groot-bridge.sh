#!/bin/bash

# training settings
PORT=20200
NUM_PROCS=8
CONFIG_NAME="groot-bridge"
AVAILABLE_GPUS="0,1,2,3,4,5,6,7"

# base settings
NUM_RUNS=1
INITIAL_SEED=42
CONDA_ENV_NAME="gr00t_n15"
OUTPUT_ROOT="runnings/gr00t_exp"

for (( run=0; run < NUM_RUNS; run++ )); do
    SEED=$(( INITIAL_SEED + run ))
    OUTPUT_DIR="$OUTPUT_ROOT/$CONFIG_NAME/$SEED"
    # training
    echo "===== [RUN $run] Starting training (torchrun) ====="
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME" || exit 1

    CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS torchrun \
        --nproc_per_node=$NUM_PROCS \
        --master_port=$PORT \
        train_vla.py \
            --config_name $CONFIG_NAME \
            --output_dir $OUTPUT_DIR \
            --seed $SEED

    TRAIN_EXIT_CODE=$?
    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "===== [RUN $run] Training script failed (exit code: $TRAIN_EXIT_CODE), terminating process ====="
        exit $TRAIN_EXIT_CODE
    fi
    echo "===== [RUN $run] Training completed, starting evaluation process ====="
    sleep 5

done

echo "===== All processes completed ====="
exit 0