# Proxy Compression for Language Modeling

This repository contains the code for our paper [Proxy Compression for Language Modeling](https://arxiv.org/abs/2602.04289).

## Overview

This codebase supports pretraining language models with various input representations. It is built as an extension of [Lingua](https://github.com/facebookresearch/lingua) and implements **Proxy Compression Training**: training language models on proxy-compressed data (gzip, neural compression, tokenizer-based compression, where tokens can be represented as-is or encoded as bytes, e.g., a 65K-vocab BPE token encoded as 2 bytes). For neural compression, see our [neural compression library](https://github.com/LZhengisme/neural-ac-compression) for offline data preprocessing.

We currently support training [OpenCoder](https://arxiv.org/abs/2411.04905)-style Transformer and [EvaByte](https://hkunlp.github.io/blog/2025/evabyte/) architectures. EvaByte supports bytes, tokens, sub-byte representations, and compressed formats; Transformers currently support token-level training only.

We also include a minimal evaluation library for downstream code generation tasks (HumanEval+, MBPP+) and BPB (Bits Per Byte) metrics.

## Setup

```bash
# Clone the repository
git clone https://github.com/LZhengisme/proxy-compression.git
cd proxy-compression

# Create virtual environment
bash setup/create_env.sh
source envs/lingua_<date>/bin/activate

# Optional: Install Flash Attention for efficient training of OpenCoder-like models
pip install psutil
pip install flash-attn --no-build-isolation
```

### Environment Variables

Set up the following environment variables for logging and model access:

```bash
# Weights & Biases logging (or use `wandb login`)
export WANDB_PROJECT=your_project_name
export WANDB_ENTITY=your_entity
export WANDB_API_KEY=your_api_key

# HuggingFace access
export HF_TOKEN=your_huggingface_token
export HF_HOME=/path/to/hf_cache
```

### Evaluation Dependencies

Install additional dependencies for code generation evaluation:
```bash
cd evals
# Alternatively, install eval dependencies in a separate environment
pip install -r requirements.txt
pip install evalplus==0.3.1 --no-deps  # Code generation evaluation
```

## Quick Start

### 1. Data Preparation

We follow the same data format as [Lingua](https://github.com/facebookresearch/lingua), and assume training data is stored as JSONL files with a `text` field, where the files are named as `<your_data_dir>/<dataset_name>/<some_prefix>.<i>.jsonl`. For instance, the running example below downloads and shuffles an example dataset (subsampled [stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)) at path `data/stackedu/stackedu.chunk.<i>.jsonl` with 4 chunks.

**Example: Download and prepare Stack-Edu data**

```bash
# Download sample data from HuggingFaceTB/stack-edu (adjust --num_samples as needed)
python setup/download_stackedu.py --output data/stackedu_raw/data.jsonl --num_samples 1000000

# Shuffle and split into chunks
bash setup/in_mem_shuffle_split.sh data/stackedu_raw/data.jsonl '*.jsonl' 4 data/stackedu stackedu
```

Your data directory structure should look like:
```
data/
├── stackedu/
│   ├── stackedu.chunk.01.jsonl
│   ├── stackedu.chunk.02.jsonl
│   ├── stackedu.chunk.03.jsonl
│   └── stackedu.chunk.04.jsonl
```

### 2. Training

We provide example training scripts in `scripts/`. You can edit these scripts to customize hyperparameters for your setup.

| Script | Description |
|--------|-------------|
| `scripts/train_opencoder.sh` | Standard OpenCoder-style Transformer on tokens |
| `scripts/train_evabyte_bytes.sh` | EvaByte on raw bytes with multi-byte prediction |
| `scripts/train_evabyte_tokenized.sh` | EvaByte on BPE tokens with multi-token prediction |
| `scripts/train_evabyte_gzip_proxy.sh` | EvaByte with gzip proxy compression |
| `scripts/train_evabyte_neural_proxy.sh` | EvaByte with neural proxy compression |
| `scripts/train_evabyte_token_proxy.sh` | EvaByte with token proxy compression |
| `scripts/train_evabyte_token_byte_proxy.sh` | EvaByte with token-byte proxy compression, where BPE tokens are encoded as bytes |
| `scripts/train_evabyte_bits_and_bytes.sh` | EvaByte with various bit/byte-level representations |

By default, we use OpenCoder's [tokenizer](https://huggingface.co/infly/OpenCoder-1.5B-Base) for most experiments. For token-byte proxy compression, we also use custom tokenizers such as `artifacts/superbpe_vocab65k.json`.

```bash
# Example: Train EvaByte with token-based proxy compression
bash scripts/train_evabyte_token_proxy.sh
```

#### Key Training Parameters

**Distributed Training:**
- **`distributed.dp_shard`**: FSDP sharding degree. Set to the GPU count or GPUs per node depending on the hardware setup (default: 8).

**Multi-byte/Token Prediction:**
- **`model.num_pred_heads`**: Number of prediction heads for future positions.
- **`model.apply_raw_multibyte_lm_head`**: When `true`, adds separate raw byte prediction heads when using proxy compression training. Set `model.num_raw_pred_heads` for the number of raw byte prediction heads. Conceptually, this means when training with proxy compression, each position will be predicted by `model.num_pred_heads` prediction heads for compressed tokens and `model.num_pred_heads + model.num_raw_pred_heads` prediction heads for raw bytes.
- **`data.n_views`**: The number of different views during data processing (for instance, with 2 multi-token prediction heads, then `n_views = 3`, consisting of the input and targets for each head). Equal to `num_pred_heads + 1` (or `num_pred_heads + num_raw_pred_heads + 1` with raw heads).
- **`apply_fused_linear_chunked_ce_loss`**: Memory-efficient chunked cross-entropy for large vocabularies. Use with `apply_multibyte_loss_mask=true` (disable when using raw prediction heads, as the masking is already handled inside the loss function).

<details>
<summary><b>Prediction Head Tensor Shapes</b></summary>
Consider mixed input representations, where raw bytes take 256 possible values and compressed tokens take 65536 possible values.

The vocabulary would be 64 for special tokens, 256 for raw bytes, 65536 for compressed bytes (tokens): `vocab_size = 65536 + 256 + 64 = 65856`

For a single prediction head, the tensor shapes would be
```python
# B = batch size, N = sequence length, V = vocabulary size
# inputs: [B, N]
# logits: [B, N, V]
# labels: [B, N]
```

For vanilla multi-token/byte prediction, tensor shapes would be
```python
# B = batch size, N = sequence length, V = vocabulary size, H = prediction heads
# inputs: [B, N]
# logits: [B, N, V * H]
# labels: [B, N, H]
```

For multi-byte prediction with additional raw prediction heads, tensor shapes would be
```python
# B = batch size, N = sequence length, V = vocabulary size, H_1 = main prediction heads, H_2 = additional raw prediction heads
# we assume there are N_r raw bytes out of N elements in total
# inputs: [B, N]
# main_logits: [B, N, V * H_1]
# additional_raw_logits: [B, N_r, 320 * H_2]
# labels: [B, N, H_1 + H_2]
```

</details>


**Attention Masking:**
- **`apply_doc_boundary_mask`**: Prevents cross-document attention within packed contexts.

**Proxy Compression:**
- **`data.compression_sampling_rate`**: Fraction of compressed data (e.g., `0.9` = 90% compressed, 10% raw).
- **`data.tokenizer.separate_embedding`**: Separate embeddings for raw vs compressed tokens.

**Neural Compression:**
- Requires offline preprocessing with our [neural compression tool](https://github.com/LZhengisme/neural-ac-compression) to build the compressed JSONL files (e.g., `data.sources="{'stackedu_neural_compressed_ow16':1.0}"`).
- Config: `data.compression_alg_config=ac_m1_key-pack_<N>-<key>` where `N` = bits per compressed symbol (default: 16, meaning there will be 2^16 = 65536 possible compressed symbols), `key` = JSONL field with compressed data.


### 3. Evaluation

#### BPB Evaluation

To run BPB evaluation on your checkpoints:

```bash
# Set evaluation parameters
CKPT_DIR=checkpoints/your_experiment/step_50000     # Path to checkpoint directory
VAL_DATA_PATH=data/validation                        # Path to validation data (JSONL format)
OUTPUT_DIR=eval_results/bpb_eval                     # Directory to save evaluation results
NUM_GPUS=1                                           # Number of GPUs to use

# Optional: Pre-consolidate distributed checkpoints
python3 -m apps.utils.pre_consolidate ckpt_dir=$CKPT_DIR

# Run BPB evaluation

# 1. For EvaByte models
torchrun --nproc_per_node=$NUM_GPUS \
    -m apps.evabyte.bpb_eval config=apps/evabyte/configs/bpb_eval.yaml \
    name=bpb_eval dump_dir=$OUTPUT_DIR metric_log_dir=$OUTPUT_DIR \
    ckpt_dir=$CKPT_DIR val_data_path=$VAL_DATA_PATH \
    use_train_doc_attn_mask_config=false use_train_data_config=false \
    apply_doc_boundary_mask=true \
    evaluate_on_raw_only=true \
    context_seqlen=4096 context_stride=4096

# 2. For OpenCoder (Transformer) models
torchrun --nproc_per_node=$NUM_GPUS \
    -m apps.main.bpb_eval config=apps/main/configs/bpb_eval.yaml \
    name=bpb_eval dump_dir=$OUTPUT_DIR metric_log_dir=$OUTPUT_DIR \
    ckpt_dir=$CKPT_DIR val_data_path=$VAL_DATA_PATH \
    use_train_doc_attn_mask_config=false use_train_data_config=true \
    apply_doc_boundary_mask=true \
    context_seqlen=4096 context_stride=4096
```

**Key parameters:**
- `use_train_data_config`: When `true`, processes validation data the same way as training data. Set `true` for byte/token models; `false` for proxy-trained models to evaluate on raw bytes.
- `evaluate_on_raw_only`: When `true`, computes BPB only on raw bytes. Use for proxy compression models.
- `context_seqlen`: Context window size for evaluation.
- `context_stride`: Stride between evaluation windows.

Results will be saved to `$OUTPUT_DIR/val_results.json`.

#### Code Generation Evaluation

**Step 1: Convert checkpoints to HuggingFace format**

```bash
# For EvaByte models
python -m apps.evabyte.scripts.hf_lingua_conversion mode=dcp_to_hf \
    dcp_to_hf_args.dcp_checkpoint_dir=checkpoints/your_experiment/step_50000 \
    dcp_to_hf_args.hf_output_dir=ckpts/your_experiment_hf \
    dcp_to_hf_args.ref_hf_output_dir=evals/ref_hf_evabyte_impl

# For Transformer (OpenCoder) models
python -m apps.main.scripts.opencoder_hf_lingua_conversion mode=dcp_to_hf \
    dcp_to_hf_args.dcp_checkpoint_dir=checkpoints/your_experiment/step_50000 \
    dcp_to_hf_args.hf_output_dir=ckpts/your_experiment_hf \
    dcp_to_hf_args.ref_hf_output_dir=evals/ref_hf_opencoder_1B5_impl
```

**Step 2: Run evaluation**

```bash
cd evals/gen_evals
bash run_gen_eval.sh <MODEL_PATH> <DATASET> <DUMP_DIR> <TOKENIZER_MODE> [SPM_PATH] [PROMPT_HEALING] [DECODING_MODE]
```

| Argument | Description |
|----------|-------------|
| `MODEL_PATH` | HuggingFace checkpoint path |
| `DATASET` | `humaneval_plus`, `mbpp_plus`, or append `:sample` for sampling |
| `DUMP_DIR` | Output directory for results |
| `TOKENIZER_MODE` | Tokenizer mode for evaluation (see below) |
| `SPM_PATH` | Path to tokenizer on HF Hub (required for `hf_tokenizer` mode) or locally trained SPM models |
| `PROMPT_HEALING` | Prompt boundary handling strategy (see below) |
| `DECODING_MODE` | `vanilla` (default) or `multibyte` (not fully tested for proxy compression training) |

**Argument Values:**

<details>
<summary><b>TOKENIZER_MODE</b> - Controls how input is tokenized during evaluation</summary>

| Value | Description |
|-------|-------------|
| `default` | Use the HuggingFace tokenizer bundled with the model checkpoint. Suitable for standard token-level models (e.g., OpenCoder). |
| `raw_sentinel` | Evaluate on raw UTF-8 bytes with a `<raw>` sentinel token prepended to the input. Use for byte-level models (e.g., EvaByte trained on bytes or proxy-compressed data). |
| `hf_tokenizer` | Use a different HuggingFace tokenizer specified via `SPM_PATH`. Useful when the model was trained with a tokenizer different from its HF model specification (e.g., EvaByte trained on BPE tokens). |
| `doublebyte` | Two bytes are grouped into 16-bit tokens. Use for models trained with double-byte representations. |
| `hf_spm` | Use a locally trained HuggingFace SPM tokenizer specified via `SPM_PATH`. |

</details>

<details>
<summary><b>PROMPT_HEALING</b> - Handles tokenization boundary artifacts at the end of prompts</summary>

| Value | Description |
|-------|-------------|
| `strip` | (Default) Strip whitespace at prompt boundary. |
| `heal` | Apply token healing to merge partial tokens across prompt/completion boundary. Can improve generation quality but may affect reproducibility. |
| `pad_to_even` | Pad prompt to even byte length. Required for `doublebyte` tokenizer mode to ensure proper 16-bit alignment. |

</details>


**Examples:**

```bash
cd evals/gen_evals

# OpenCoder trained on tokens
bash run_gen_eval.sh ckpts/opencoder_1b5_hf humaneval_plus logs default

# EvaByte trained on bytes
bash run_gen_eval.sh ckpts/your_evabyte_hf humaneval_plus logs raw_sentinel

# EvaByte trained on tokens
bash run_gen_eval.sh ckpts/evabyte_tokens_multibytepred2_hf humaneval_plus logs hf_tokenizer infly/OpenCoder-1.5B-Base

# EvaByte trained on tokens with sampling (20 samples, temperature=0.2)
bash run_gen_eval.sh ckpts/evabyte_tokens_multibytepred2_hf humaneval_plus:sample logs hf_tokenizer infly/OpenCoder-1.5B-Base

# EvaByte trained on proxy compressed data, evaluated on bytes
bash run_gen_eval.sh ckpts/evabyte_neural_proxy humaneval_plus logs raw_sentinel
```

Results are saved to `<DUMP_DIR>/<DATASET>/<MODEL_NAME>/` with generated samples and evaluation scores.

## Configuration Details

### Model Configurations

Pre-defined model configs are available in:
- `apps/main/configs/` — OpenCoder-style Transformer (1.5B, 8B)
- `apps/evabyte/configs/` — EvaByte architecture (500M, 1.5B, 4B, 7B, 14B)

### Data Representations

#### Tokenizer Types (`data.tokenizer.name`)

| Use Case | Type | Description |
|----------|------|-------------|
| OpenCoder training on tokens | `data.tokenizer.name=vanilla_hf` | Standard HuggingFace tokenizer. Specify tokenizer via `data.tokenizer.path`. |
| EvaByte on bytes or most proxy compression | `data.tokenizer.name=hf` | A customized HuggingFace tokenizer with sentinel tokens (`<raw>`, `<compressed>`) added to vocabulary. |
| Token-based proxy compression | `data.tokenizer.name=token_plus_byte` | Hybrid tokenizer supporting both BPE tokens and raw bytes in a unified vocabulary. Specify BPE tokenizer via `data.tokenizer.spm_byte_path`. |
| Double-byte experiments | `data.tokenizer.name=doublebyte` | Packs consecutive byte pairs into 16-bit tokens (vocab_size=65536 + sentinels). |

**Related tokenizer options:**
- `data.tokenizer.spm_byte_path`: Path to BPE tokenizer for byte-level encoding of tokens, or for token-byte proxy compression.
- `data.tokenizer.separate_embedding`: When `true`, uses separate embedding vectors for raw bytes vs compressed tokens. This is a must when tokens take a large vocabulary (e.g., 65k); while for tokens encoded as bytes, they can share the same embedding space with raw bytes (we did not observe significant performance differences in our experiments).
- `data.tokenizer.byte_converter_config.byte_converter_type`: Byte encoding scheme for token-byte proxy compression (e.g., `gray` for Gray coding to promote locality)

#### Compression Algorithms (`data.compression_alg_config`)

| Type | Description | vocab_size |
|------|-------------|------------|
| `vanilla` | Use standard tokenization. | tokenizer vocab |
| `gzip` / `gzip_no_mtime` | Gzip compression on raw bytes. `gzip_no_mtime` omits timestamps for deterministic compression. | 256 + sentinels |
| `spm_byte` | BPE tokens as the compression. | either 256 or vocab size + sentinels |
| `doublebyte` | Packs pairs of UTF-8 bytes into 16-bit tokens. | 65536 + sentinels |
| `halfbyte` | Splits each byte into two 4-bit tokens. | 16 + sentinels |
| `doublebit` | Splits each byte into four 2-bit tokens. | 4 + sentinels |
| `bit` | Splits each byte into eight 1-bit tokens. | 2 + sentinels |
| `ac_m1_key-pack_N-KEY` | Neural compression with arithmetic coding. `N` = bits per symbol (typically 16), `KEY` = JSONL field storing the compressed data. Requires [offline preprocessing with our neural compression library](https://github.com/LZhengisme/neural-ac-compression). |

#### Mixed Training Modes (`data.raw_compression_mix_option`)

Controls how raw and compressed representations are mixed during training.

| Mode | Description |
|------|-------------|
| `vanilla` | Single representation only, no mixing. Used in standard byte-level or token-level training |
| `sentinel` | Wraps data with sentinel tokens: `<raw>...</raw>` for raw bytes, `<compressed>...</compressed>` for compressed data. Model learns to distinguish representations. **Recommended** for proxy compression training |
| `translation_raw_compressed` | Raw data paired with compressed translation appended: `<raw>...</raw><compressed>...</compressed>`|
| `translation_compressed_raw` | Raw data paired with compressed translation prepended: `<compressed>...</compressed><raw>...</raw>` |
| `translation_random` | Randomly alternates between the two translation orders above. **Recommended** for initial warmup in proxy compression |
| `parallel_raw_compressed` | Similar to translation modes, but treats raw and compressed as separate samples, and one cannot attend to the other. |
| `parallel_compressed_raw` | Parallel mode with compressed prepended to raw. |
| `parallel_random` | Randomly alternates between the two parallel orders. |

### Compression Rate Scheduling

For proxy compression training, you can schedule both the **compression rate** (fraction of compressed vs raw data) and **mixing mode** across training phases:

**Rate scheduling:** The compression rate ramps up during warmup, stays constant during steady phase, then optionally decays.
- `compression_initial_rate` linearly increased to `compression_peak_rate` during `compression_warmup_steps`
- constant `compression_peak_rate` with `compression_steady_steps`
- `compression_peak_rate` decayed to `compression_final_rate` during `compression_decay_steps` (if any)

**Mode scheduling:** The mixing mode can also change across phases via `compression_initial_mode`, `compression_steady_mode`, and `compression_final_mode`. A common pattern: start with `translation_random` during warmup (helps model learn raw↔compressed mapping), then transition to `sentinel` for the main training.

```bash
enable_compression_rate_schedule=true \
compression_warmup_steps=10000 \
compression_steady_steps=40000 \
compression_decay_steps=0 \
compression_initial_rate=0.4 \
compression_peak_rate=0.9 \
compression_final_rate=0.9 \
compression_initial_mode=translation_random \
compression_steady_mode=sentinel \
compression_final_mode=sentinel
```


## Project Structure

```
proxy-compression/
├── apps/
│   ├── main/                       # Standard Transformer (OpenCoder) implementation
│   │   ├── train.py
│   │   ├── bpb_eval.py
│   │   ├── configs/                # Model configs (1.5B, 8B)
│   │   └── scripts/                # HF conversion scripts
│   ├── evabyte/                    # EvaByte architecture
│   │   ├── train.py
│   │   ├── bpb_eval.py
│   │   ├── configs/                # Model configs (500M to 14B)
│   │   ├── component/              # Triton kernels for efficient attention
│   │   └── scripts/                # HF conversion scripts
│   └── utils/                      # Shared utilities (checkpoint consolidation)
├── lingua/                         # Core library (data loading, tokenizers, training utils)
├── evals/
│   ├── gen_evals/                  # Code generation evaluation (HumanEval+, MBPP+)
│   ├── ref_hf_evabyte_impl/        # Reference HF implementation for EvaByte
│   └── ref_hf_opencoder_1B5_impl/  # Reference HF implementation for OpenCoder
├── scripts/                        # Example training scripts
├── setup/                          # Environment and data preparation scripts
└── data/                           # Training data directory (user-provided)
```

## Citation

If you find this codebase useful, please cite our paper:

```bibtex
@article{zheng2026proxy,
  title={Proxy Compression for Language Modeling},
  author={Zheng, Lin and Li, Xinyu and Liu, Qian and Feng, Xiachong and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2602.04289},
  year={2026}
}
```

## Acknowledgments

This codebase is built upon [Lingua](https://github.com/facebookresearch/lingua). We thank the authors for their excellent work.
