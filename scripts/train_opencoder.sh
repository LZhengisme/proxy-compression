#!/bin/bash
# =============================================================================
# Train Standard Transformer (OpenCoder-style)
# =============================================================================

NUM_GPUS=4

# Adjust dp_shard based on number of GPUs (default is 8)
if [[ $NUM_GPUS -le 8 ]]; then
  EXTRA_ARGS="distributed.dp_shard=$NUM_GPUS"
else
  EXTRA_ARGS=""
fi

EXP_NAME=opencoder_1b5
torchrun --nproc-per-node $NUM_GPUS -m apps.main.train \
  config=apps/main/configs/opencoder_1B5.yaml \
  data.compression_alg_config=vanilla \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=true \
  data.batch_size=8 \
  data.seq_len=4096 \
  steps=50000 $EXTRA_ARGS
