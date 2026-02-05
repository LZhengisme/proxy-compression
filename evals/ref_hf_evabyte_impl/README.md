---
license: apache-2.0
---
# EvaByte Model Card

**EvaByte** is a 6.5B **byte-level language model** built upon an improved architecture with multibyte prediction and EVA -- an efficient attention mechanism designed for scalability and performance. Trained on 1.5T bytes spanning natural language text, math, and code, EvaByte demonstrates the viability of efficient byte-level processing at scale -- rivaling top open-source tokenizer-based LMs using 5x less training data, excelling in coding tasks, and decoding up to 2x faster.

## Model Resources

- **Repository:** https://github.com/openevabyte/evabyte
- **Blog:** https://hkunlp.github.io/blog/2025/evabyte and https://sambanova.ai/blog/evabyte-efficient-byte-level-language-models-at-scale
- **Paper:** Coming soon

## Model Details

EvaByte is trained using the performant SambaNova SN30 RDU system with a batch size of 8M bytes and 32K context length. The training process consists of 3 phases: after pre-training on 1.2T bytes (yielding **EvaByte-Phase1**), two independent annealing runs (100B and 200B bytes respectively) are conducted with learning rate linearly decayed from 1e-4 to 0. The resulting checkpoints are merged via model soup (**EvaByte**), which then undergoes supervised fine-tuning (**EvaByte-SFT**).

| Stage | Model |
|:----- |:-----|
| Base (before annealing) | [EvaByte-Phase1](https://huggingface.co/evabyte/EvaByte-Phase1) |
| Base | [EvaByte](https://huggingface.co/evabyte/EvaByte) <-- you are here |
| SFT  | [EvaByte-SFT](https://huggingface.co/evabyte/EvaByte-SFT) |


## Usage

**Note:** Make sure to set `trust_remote_code=True` when loading the model (or tokenizer), as our implementation includes custom code.

The code snippet below demonstrates EvaByte-6.5B for completion:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")

prompt = "The quick brown fox jumps "

# Tokenize input
# Option 1: standard HF tokenizer interface
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Option 2: Direct UTF-8 byte encoding with offset
# Note: Each byte is offset by 64 with <bos> prepended.
input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]]).to("cuda")

# byte-by-byte generation (default)
generation_output = model.generate(
    input_ids=input_ids, 
    max_new_tokens=32
)
# alternatively, use faster multibyte generation
generation_output = model.multi_byte_generate(
    input_ids=input_ids, 
    max_new_tokens=32
)

# Decode and print the output
response = tokenizer.decode(
    generation_output[0][input_ids.shape[1]:], 
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)
print(response)
# Sample output:
# over the lazy dog.\n\nThe quick
```

### ⚙️ Generation Modes

EvaByte supports two generation interfaces:
- `model.generate()`: The default generation method compatible with Huggingface `transformers` library. This approach generates one byte at a time and might be slow.
- `model.multi_byte_generate()`: A faster alternative that generates multiple bytes per step and usually yields the same result as `model.generate()` under greedy decoding, with the implementation adapted from [Medusa](https://github.com/FasterDecoding/Medusa). `model.multi_byte_generate()` supports a subset of arguments in `model.generate()`:
    - `input_ids`: the input byte ids.
    - `temperature`: the temperature for sampling.
    - `max_length`: the maximum length of the generated sequence.
    - `max_new_tokens`: the maximum number of new bytes to generate.
    - `stopping_criteria`: the [stopping criteria](https://huggingface.co/docs/transformers/v4.47.1/en/internal/generation_utils#transformers.StoppingCriteria) for generation.
    - `top_p`: the top-p parameter for sampling.
    - `do_sample`: greedy decoding or sampling.

**Notes and Limitations:**
- `device_map="auto"` is not supported for >2 GPUs.
- Only batch size of 1 (with `attention_mask=None`) is supported for decoding.
- `torch_dtype=torch.bfloat16` is required.
- The multibyte generation `model.multi_byte_generate()` might return extra bytes after the end-of-sequence sentinel, due to the nature of the multibyte decoding. Manual truncation or cleaning may be needed.

## Bias, Risks, and Limitations
As a pretrained base model, **EvaByte** has not been fine-tuned for chat or instruction following, so users should not expect reliable performance in conversational or instruction-based tasks. Like other base models, it does not incorporate any moderation mechanisms, making it possible to generate potentially harmful or inappropriate content.

## Evaluation 

For detailed evaluation results, check out our blog post at [SambaNova](https://sambanova.ai/blog/evabyte-efficient-byte-level-language-models-at-scale) or [HKUNLP](https://hkunlp.github.io/blog/2025/evabyte).

## Citation
```bibtex
@misc{evabyte,
    title = {EvaByte: Efficient Byte-level Language Models at Scale},
    url = {https://hkunlp.github.io/blog/2025/evabyte},
    author = {Lin Zheng and Xueliang Zhao and Guangtao Wang and Chen Wu and David Dong and Angela Wang and Mingran Wang and Yun Du and Haige Bo and Amol Sharma and Bo Li and Kejie Zhang and Changran Hu and Urmish Thakker and Lingpeng Kong},
    year = {2025}
}
```
