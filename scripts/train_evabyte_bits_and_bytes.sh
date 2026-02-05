#!/bin/bash
# =============================================================================
# Train EvaByte with Sub-byte and Multi-byte Representations
# 
# This script demonstrates training with different granularities:
#   - doublebyte (16-bit)
#   - Sub-byte: halfbyte (4-bit), doublebit (2-bit), bit (1-bit)
# =============================================================================

NUM_GPUS=4

# Adjust dp_shard based on number of GPUs (default is 8)
if [[ $NUM_GPUS -le 8 ]]; then
  EXTRA_ARGS="distributed.dp_shard=$NUM_GPUS"
else
  EXTRA_ARGS=""
fi

# -----------------------------------------------------------------------------
# 1. Double-byte (16-bit tokens)
#    - Packs 2 consecutive bytes into one 16-bit token
#    - vocab_size=65536+sentinels
# -----------------------------------------------------------------------------
EXP_NAME=evabyte_doublebytes
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.tokenizer.name=doublebyte \
  data.compression_sampling_rate=1.0 \
  data.raw_compression_mix_option=sentinel \
  data.compression_alg_config=doublebyte \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=true \
  model.vocab_size=65600 \
  model.num_pred_heads=1 \
  data.n_views=2 \
  data.batch_size=2 \
  data.seq_len=16384 \
  steps=50000 $EXTRA_ARGS

# -----------------------------------------------------------------------------
# 2. Half-byte (4-bit tokens)
#    - Splits each byte into two 4-bit tokens
#    - vocab_size=16+sentinels (set to 128 for practicality: small, and a multiple of 64)
# -----------------------------------------------------------------------------
EXP_NAME=evabyte_halfbytes
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.compression_sampling_rate=1.0 \
  data.raw_compression_mix_option=sentinel \
  data.compression_alg_config=halfbyte \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=true \
  model.vocab_size=128 \
  model.num_pred_heads=1 \
  data.n_views=2 \
  data.batch_size=1 \
  data.seq_len=32768 \
  steps=50000 $EXTRA_ARGS

# -----------------------------------------------------------------------------
# 3. Double-bit (2-bit tokens)
#    - Splits each byte into four 2-bit tokens
#    - vocab_size=4+sentinels
# -----------------------------------------------------------------------------
EXP_NAME=evabyte_doublebits
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.compression_sampling_rate=1.0 \
  data.raw_compression_mix_option=sentinel \
  data.compression_alg_config=doublebit \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=true \
  model.vocab_size=128 \
  model.num_pred_heads=1 \
  data.n_views=2 \
  data.batch_size=1 \
  data.seq_len=65536 \
  steps=50000 $EXTRA_ARGS

# -----------------------------------------------------------------------------
# 4. Bit-level (1-bit tokens)
#    - Splits each byte into eight 1-bit tokens
#    - vocab_size=2+sentinels
# -----------------------------------------------------------------------------
EXP_NAME=evabyte_bits
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.compression_sampling_rate=1.0 \
  data.raw_compression_mix_option=sentinel \
  data.compression_alg_config=bit \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=true \
  model.vocab_size=128 \
  model.num_pred_heads=1 \
  data.n_views=2 \
  data.batch_size=1 \
  data.seq_len=65536 \
  steps=50000 $EXTRA_ARGS
