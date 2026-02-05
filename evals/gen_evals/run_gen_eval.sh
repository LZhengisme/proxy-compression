set -ex

NUM_GPUS=1
USER_MODEL=${1:-"none"}
USER_DATASET=${2:-"none"}
DUMP_DIR=${3:-"logs"}
TOKENIZER_MODE=${4:-'default'}
SPM_PATH=${5:-""}
PROMPT_HEALING=${6:-"strip"}
DECODING_MODE=${7:-"vanilla"}

if [ $# -lt 7 ]; then
    shift $#
else
    shift 7
fi

# We current only evaluate base models.
MODEL_TYPE="base"

if [[ $USER_MODEL == "none" ]]; then
    MODELS=(
        "EvaByte/EvaByte"
    )
elif [[ $USER_MODEL == "pretrain" ]]; then
    MODELS=(
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0484865"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0400000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0350000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0300000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0250000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0200000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0175000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0150000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0125000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0100000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0050000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0020000"
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints#pretrain_iter_0010000"
    )
elif [[ $USER_MODEL == "pretrain8b" ]]; then
    MODELS=(
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0300000"
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0250000"
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0200000"
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0150000"
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0100000"
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0050000"
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0020000"
        "OpenCoder-LLM/OpenCoder-8B-Base-Checkpoints#pretrain_iter_0010000"
    )
else
    MODELS=(
        $USER_MODEL
    )
fi

if [[ $USER_DATASET == "none" ]]; then
    DATASETS=(
        "humaneval_plus"
        "mbpp_plus"
    )
else
    DATASETS=(
        $USER_DATASET
    )
fi

for DATASET in "${DATASETS[@]}"
do
    TASK_MODE="greedy"
    if [[ "$DATASET" == *:* ]]; then
        TASK_MODE="${DATASET#*:}"
        DATASET="${DATASET%%:*}"
    fi
    if [[ "$TASK_MODE" == "greedy" ]]; then
        TASK_ARGS="-g greedy"
    elif [[ "$TASK_MODE" == "sample" ]]; then
        TASK_ARGS="-g sample -b 1 -n 20 -t 0.2"
    else
        echo "Error: unknown task mode $TASK_MODE"
        exit 1
    fi

    for index in "${!MODELS[@]}"
    do
        MODEL="${MODELS[$index]}"

        if [[ $DECODING_MODE == "multibyte" ]]; then
            DECODING_ARGS="--multi_byte_decoding"
            save_subdir=multibyte
        elif [[ $DECODING_MODE == "vanilla" ]]; then
            DECODING_ARGS=""
            save_subdir=vanilla
        else
            echo "Error: unknown decoding mode $DECODING_MODE"
            exit 1
        fi

        if [[ $TOKENIZER_MODE == "raw_sentinel" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode raw_sentinel --max_length_generation 8192 --max_new_tokens_generation 2048 $@"
            save_subdir=${save_subdir}_raw_sentinel
        elif [[ $TOKENIZER_MODE == "spm_"* ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode $TOKENIZER_MODE --spm_path $SPM_PATH --prompt_healing $PROMPT_HEALING --max_length_generation 2048 --max_new_tokens_generation 512 $@"
            save_subdir=${save_subdir}_${TOKENIZER_MODE}
        elif [[ $TOKENIZER_MODE == "token_plus_byte" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode $TOKENIZER_MODE --spm_path $SPM_PATH --max_length_generation 2048 --max_new_tokens_generation 512 $@"
            save_subdir=${save_subdir}_${TOKENIZER_MODE}
        elif [[ $TOKENIZER_MODE == "halfbyte" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode $TOKENIZER_MODE --max_length_generation 16384 --max_new_tokens_generation 4096"
            save_subdir=${save_subdir}_${TOKENIZER_MODE}
        elif [[ $TOKENIZER_MODE == "subbyte" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode $TOKENIZER_MODE --max_length_generation 16384 --max_new_tokens_generation 4096"
            save_subdir=${save_subdir}_${TOKENIZER_MODE}
        elif [[ $TOKENIZER_MODE == "doublebit" || $TOKENIZER_MODE == "bit" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode $TOKENIZER_MODE --max_length_generation 16384 --max_new_tokens_generation 4096"
            save_subdir=${save_subdir}_${TOKENIZER_MODE}
        elif [[ $TOKENIZER_MODE == "doublebyte" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode $TOKENIZER_MODE --prompt_healing pad_to_even --max_length_generation 4096 --max_new_tokens_generation 1024"
            save_subdir=${save_subdir}_${TOKENIZER_MODE}
        elif [[ $TOKENIZER_MODE == "default" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode default --max_length_generation 2048 --max_new_tokens_generation 512"
            save_subdir=${save_subdir}_default
        elif [[ $TOKENIZER_MODE == "hf_tokenizer" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode hf_tokenizer --spm_path $SPM_PATH --max_length_generation 2048 --max_new_tokens_generation 512"
            save_subdir=${save_subdir}_hf_tokenizer
        elif [[ $TOKENIZER_MODE == "hf_spm" ]]; then
            DECODING_ARGS=${DECODING_ARGS}" --tokenizer_mode hf_spm --spm_path $SPM_PATH --prompt_healing $PROMPT_HEALING --max_length_generation 2048 --max_new_tokens_generation 512"
            save_subdir=${save_subdir}_hf_spm
        else
            echo "Error: unknown $TOKENIZER_MODE"
            exit 1
        fi

        MODEL_PATH="$(basename "$MODEL")"

        if [[ "$TASK_MODE" == "greedy" ]]; then
            SAVE_DIR=${DUMP_DIR}/$DATASET/${MODEL_PATH}_${save_subdir}
        elif [[ "$TASK_MODE" == "sample" ]]; then
            SAVE_DIR=${DUMP_DIR}/${DATASET}-${TASK_MODE}/${MODEL_PATH}_${save_subdir}
        else
            echo "Error: unknown task mode $TASK_MODE"
            exit 1
        fi

        mkdir -p $SAVE_DIR

        # Handle MODEL#REVISION syntax for HuggingFace model revisions
        if [[ "$MODEL" == *"#"* ]]; then
            MODEL_BASE="${MODEL%%#*}"
            REVISION="${MODEL##*#}"
            REVISION_ARGS="--revision $REVISION"
        else
            MODEL_BASE="$MODEL"
            REVISION_ARGS=""
        fi

        if [[ $MODEL_TYPE == "base" ]]; then
            MODEL_ARGS="$REVISION_ARGS"
        elif [[ $MODEL_TYPE == "instruct" ]]; then
            MODEL_ARGS="$REVISION_ARGS --instruct_format"
        else
            echo "Error: unknown decoding mode $MODEL_TYPE"
            exit 1
        fi

        bash launch.sh -r gen \
            -d $DATASET \
            -m $MODEL_BASE \
            -s $SAVE_DIR \
            $TASK_ARGS \
            -p $NUM_GPUS \
            -e True \
            $DECODING_ARGS $MODEL_ARGS

        bash launch.sh -r eval \
            -d $DATASET \
            -m $MODEL_BASE \
            -s $SAVE_DIR \
            $TASK_ARGS \
            -p 10 \
            -e True \
            $DECODING_ARGS $MODEL_ARGS

    done
done
