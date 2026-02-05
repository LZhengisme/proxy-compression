#!/bin/bash
# =============================================================================
# Train EvaByte with token-byte proxy compression, where BPE tokens are encoded as bytes
# =============================================================================

NUM_GPUS=4

# Adjust dp_shard based on number of GPUs (default is 8)
if [[ $NUM_GPUS -le 8 ]]; then
  EXTRA_ARGS="distributed.dp_shard=$NUM_GPUS"
else
  EXTRA_ARGS=""
fi

# -----------------------------------------------------------------------------
# 1. BPE tokens encoded as bytes (token-byte proxy compression)
#    - we train a 65K BPE vocab, encode each token ID as 2 bytes
#    - vocab_size=256+sentinels
#    - Proxy compression: 90% compressed, 10% raw bytes
# -----------------------------------------------------------------------------
EXP_NAME=evabyte_token_byte_proxy
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.compression_sampling_rate=0.90 \
  data.raw_compression_mix_option=sentinel \
  data.compression_alg_config=spm_byte \
  data.tokenizer.spm_byte_path=artifacts/superbpe_vocab65k.json \
  data.tokenizer.separate_embedding=true \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=True \
  model.vocab_size=576 \
  model.num_pred_heads=1 \
  data.n_views=2 \
  data.batch_size=2 \
  data.seq_len=16384 \
  enable_compression_rate_schedule=true \
  compression_warmup_steps=10000 \
  compression_steady_steps=40000 \
  compression_decay_steps=0 \
  compression_initial_rate=0.4 \
  compression_peak_rate=0.9 \
  compression_final_rate=0.9 \
  compression_initial_mode=translation_random \
  compression_steady_mode=sentinel \
  compression_final_mode=sentinel \
  steps=50000 $EXTRA_ARGS

# -----------------------------------------------------------------------------
# 2. BPE tokens as bytes with Gray coding
#    - Same as above, but uses Gray code for byte encoding
#    - Gray coding preserves locality: adjacent (surface-wise) token IDs differ by 1 bit
# -----------------------------------------------------------------------------
EXP_NAME=evabyte_token_byte_proxy_gray_coding
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.compression_sampling_rate=0.90 \
  data.raw_compression_mix_option=sentinel \
  data.compression_alg_config=spm_byte \
  data.tokenizer.spm_byte_path=artifacts/superbpe_vocab65k.json \
  data.tokenizer.separate_embedding=true \
  data.tokenizer.byte_converter_config.byte_converter_type=gray \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=True \
  model.vocab_size=576 \
  model.num_pred_heads=1 \
  data.n_views=2 \
  data.batch_size=2 \
  data.seq_len=16384 \
  enable_compression_rate_schedule=true \
  compression_warmup_steps=10000 \
  compression_steady_steps=40000 \
  compression_decay_steps=0 \
  compression_initial_rate=0.4 \
  compression_peak_rate=0.9 \
  compression_final_rate=0.9 \
  compression_initial_mode=translation_random \
  compression_steady_mode=sentinel \
  compression_final_mode=sentinel \
  steps=50000 $EXTRA_ARGS
