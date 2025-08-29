PORT=19958
AVAILABLE_GPUS="0"
CONFIG_NAME="groot-calvin"
CONDA_ENV_NAME="gr00t_n15"
OUTPUT_ROOT="runnings/gr00t_exp"

SEED=42
EVAL_CKPT_NAME="Model_ckpt_200000"
OUTPUT_DIR="$OUTPUT_ROOT/$CONFIG_NAME/$SEED"

# server
echo "===== [RUN $run] Starting server... ====="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME" || exit 1
python server_vla.py \
    --port $PORT \
    --device_id $(echo "$AVAILABLE_GPUS" | cut -d ',' -f 1) \
    --ckpt_path $OUTPUT_DIR \
    --ckpt_name $EVAL_CKPT_NAME 