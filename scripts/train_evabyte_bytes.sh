#!/bin/bash
# =============================================================================
# Train EvaByte on Raw Bytes
# =============================================================================

NUM_GPUS=4

# Adjust dp_shard based on number of GPUs (default is 8)
if [[ $NUM_GPUS -le 8 ]]; then
  EXTRA_ARGS="distributed.dp_shard=$NUM_GPUS"
else
  EXTRA_ARGS=""
fi

EXP_NAME=evabyte_bytes_multibytepred8
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.raw_compression_mix_option=sentinel \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=true \
  model.num_pred_heads=8 \
  data.n_views=9 \
  data.batch_size=2 \
  data.seq_len=16384 \
  steps=50000 $EXTRA_ARGS
