#!/bin/bash
# =============================================================================
# Train EvaByte with Proxy Compression (neural)
# =============================================================================

NUM_GPUS=4

# Adjust dp_shard based on number of GPUs (default is 8)
if [[ $NUM_GPUS -le 8 ]]; then
  EXTRA_ARGS="distributed.dp_shard=$NUM_GPUS"
else
  EXTRA_ARGS=""
fi

EXP_NAME=evabyte_neural_proxy_90neural10bytes_multisymbolpredc1r1
torchrun --nproc-per-node $NUM_GPUS -m apps.evabyte.train \
  config=apps/evabyte/configs/evabyte_1b5.yaml \
  data.compression_sampling_rate=0.90 \
  data.raw_compression_mix_option=sentinel \
  data.compression_alg_config=ac_m1_key-pack_16-m1_ac_ow16 \
  dump_dir=checkpoints/$EXP_NAME \
  log_dump_dir=logs/$EXP_NAME \
  data.root_dir=data \
  data.sources="{'stackedu_neural_compressed_ow16':1.0}" \
  name=$EXP_NAME \
  logging.wandb.name=$EXP_NAME \
  logging.wandb.project=$WANDB_PROJECT \
  logging.wandb.entity=$WANDB_ENTITY \
  apply_doc_boundary_mask=True \
  model.vocab_size=65856 \
  model.num_pred_heads=1 \
  model.apply_raw_multibyte_lm_head=true \
  model.num_raw_pred_heads=1 \
  data.n_views=3 \
  apply_fused_linear_chunked_ce_loss=true \
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
