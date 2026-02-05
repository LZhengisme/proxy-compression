#!/bin/bash
# =============================================================================
# Train EvaByte on BPE tokens
# =============================================================================

NUM_GPUS=4

# Adjust dp_shard based on number of GPUs (default is 8)
if [[ $NUM_GPUS -le 8 ]]; then
  EXTRA_ARGS="distributed.dp_shard=$NUM_GPUS"
else
  EXTRA_ARGS=""
fi

EXP_NAME=evabyte_tokens_multibytepred2
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.tokenizer.name=vanilla_hf \
  data.tokenizer.path=infly/OpenCoder-1.5B-Base \
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
  model.num_pred_heads=2 \
  data.n_views=3 \
  model.vocab_size=96640 \
  data.batch_size=8 \
  data.seq_len=4096 \
  steps=50000 $EXTRA_ARGS
