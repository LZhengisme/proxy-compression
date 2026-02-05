#!/bin/bash
set -x

err_msg() { echo "Invalid arguments" 1>&2; exit 1; }

# process named arguments
while getopts ":s:m:t:g:n:b:p:d:r:e:" o; do
    case "${o}" in
        d)
            # possible options: [
            #    `humaneval`,
            #    `mbpp`,
            #    `humaneval_plus`,
            #    `mbpp_plus`,
            #    `recode`,
            # ]
            DATASET=${OPTARG}
            ;;
        n)
            GEN_NUM_SAMPLES=${OPTARG}
            ;;
        b)
            # batch size for sampling generation
            BATCH_SIZE=${OPTARG}
            ;;
        s)
            # path to save output results
            SAVE_DIR=${OPTARG}
            ;;
        g)
            # `GENERATE_MODE` could be set to either
            #    - `greedy` to decode a single solution for each
            #               problem (for pass@1 evaluation), or
            #    - `sample` for batched generation (the number of samples are
            #               specified below). This mode is typically used for
            #               evaluating pass@$k$ metrics with $k>1$
            #               (or doing majority voting for GSM-8k).
            GENERATION_MODE=${OPTARG}
            ;;
        m)
            # HF model path
            MODEL=${OPTARG}
            ;;
        t)
            GEN_TEMPERATURE=${OPTARG}
            ;;
        r)
            # `RUN_MODE` could be either `gen` for generation or `eval` for evaluation.
            # It could also be `gen+eval` to generate and evaluate solutions in one command.
            RUN_MODE=${OPTARG}
            ;;
        p)
            # the number of processes used in evaluation
            NUM_PROCESSES=${OPTARG}
            ;;
        e)
            # use -e to separate program-level arguments and custom arguments.
            break
            ;;
        *)
            err_msg
            ;;
    esac
done
shift $((OPTIND-1))
SAVE_DIR=${SAVE_DIR:-'results'}
MODEL=${MODEL:-'bigcode/santacoder'}
DATASET=${DATASET:-"humaneval"}
BATCH_SIZE=${BATCH_SIZE:-20}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-'none'}
GEN_NUM_SAMPLES=${GEN_NUM_SAMPLES:-'none'}
GENERATION_MODE=${GENERATION_MODE:-"greedy"}
RUN_MODE=${RUN_MODE:-"gen"}
NUM_PROCESSES=${NUM_PROCESSES:-0}

case $RUN_MODE in
    "gen"|"gen+eval"|"eval") echo "RUN_MODE is valid" ;;
    *) echo "RUN_MODE not valid; only gen, gen+eval, eval mode are supported" ;;
esac

#########################################
# Setup HF model arguments

# TOKENIZER defaults to be the same as MODEL
TOKENIZER=$MODEL

MODEL_ARGS="--model $MODEL --tokenizer_path $TOKENIZER --trust_remote_code"
HF_TOKEN=${HF_TOKEN:-'None'}
if [[ $HF_TOKEN != 'None' ]]; then
    MODEL_ARGS=$MODEL_ARGS" --use_auth_token $HF_TOKEN"
fi

#########################################
# Setup decoding arguments
PRECISION=bf16

if [[ $GENERATION_MODE == "sample" ]]; then
    NUM_SAMPLES=200
    TEMPERATURE=0.8
    if [[ "$GEN_NUM_SAMPLES" == none ]]; then
        GEN_NUM_SAMPLES=$NUM_SAMPLES
    fi
    if [[ "$GEN_TEMPERATURE" == none ]]; then
        GEN_TEMPERATURE=$TEMPERATURE
    fi
    DECODE_ARGS="--temperature $GEN_TEMPERATURE \
        --do_sample True \
        --top_p 0.95 \
        --n_samples $GEN_NUM_SAMPLES \
        --batch_size $BATCH_SIZE"

    DECODE_NAME="temp$GEN_TEMPERATURE"
elif [[ $GENERATION_MODE == "greedy" ]]; then
    DECODE_ARGS="--temperature 0.0 \
        --do_sample False \
        --n_samples 1 \
        --batch_size 1" 

    DECODE_NAME="greedy"
else
    echo "unknown decoding setting."
    exit 1
fi

#########################################
# Setup script launch entries

# 1 node by default
nnodes=1
node_rank=0
# calculate number of GPUs based on CUDA_VISIBLE_DEVICES
nproc_per_node=$NUM_PROCESSES
function ACCELERATE_LAUNCH_GEN() { 
    accelerate launch --num_processes $nproc_per_node \
        --num_machines $nnodes \
        --machine_rank $node_rank \
        $@
}

function ACCELERATE_LAUNCH_EVAL() { 
    accelerate launch --cpu $@
}

#########################################
# Run in either gen or eval mode

if [[ $RUN_MODE == *"gen"* ]]; then
    ACCELERATE_LAUNCH_GEN main.py \
    --tasks $DATASET \
    $MODEL_ARGS \
    --precision $PRECISION \
    $DECODE_ARGS \
    --generation_only \
    --save_generations \
    --save_generations_path "$SAVE_DIR/generations_$DECODE_NAME.json" $@
fi

if [[ $RUN_MODE == *"eval"* ]]; then
    export TF_CPP_MIN_LOG_LEVEL=3 
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    ACCELERATE_LAUNCH_EVAL main.py \
    --tasks $DATASET \
    $MODEL_ARGS \
    $DECODE_ARGS \
    --num_processes $NUM_PROCESSES \
    --allow_code_execution \
    --load_generations_path "$SAVE_DIR/generations_$DECODE_NAME.json" \
    --metric_output_path "$SAVE_DIR/evaluations_$DECODE_NAME.json" $@
fi