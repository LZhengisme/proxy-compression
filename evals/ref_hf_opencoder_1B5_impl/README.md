---
license: other
license_name: inf
license_link: https://huggingface.co/infly/OpenCoder-1.5B-Base/blob/main/LICENSE
language:
- en
- zh
pipeline_tag: text-generation
library_name: transformers
---



<div align="center">
  <img src="https://github.com/OpenCoder-llm/opencoder-llm.github.io/blob/main/static/images/opencoder_icon.jpg?raw=true" width="50%" alt="OpenCoder-Icon" />
</div>



<p align="center">
  <!-- <a href="https://arxiv.org/pdf/2411.04905"><b>Paper Link</b>👁️</a> -->
  🏠 <a href="https://opencoder-llm.github.io/">Home Page</a>&nbsp&nbsp | 
  &nbsp&nbsp 🤗 <a href="https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e">Model</a>&nbsp&nbsp | 
  &nbsp&nbsp 📊 <a href="https://huggingface.co/collections/OpenCoder-LLM/opencoder-datasets-672e6db6a0fed24bd69ef1c2">Dataset</a>&nbsp&nbsp | 
  &nbsp&nbsp 📄<a href="https://arxiv.org/abs/2411.04905">Paper</a>&nbsp&nbsp
</p>


## 1. Introduction

**OpenCoder** is an open and reproducible code LLM family which includes 1.5B and 8B base and chat models, supporting both English and Chinese languages. Starting from scratch, OpenCoder is pretrained on 2.5 trillion tokens composed of 90% raw code and 10% code-related web data, and supervised finetuned on over 4.5M high-quality SFT examples, finally reaching the performance of top-tier code LLMs. We provide not only model weights and inference code, but also the reproducible training data, the complete data processing pipeline, rigorous experimental ablation results, and detailed training protocols. Empowering researchers to build and innovate, OpenCoder is your open foundation for advancing code AI. 

- **Complete Open Source**: OpenCoder ensures full transparency by releasing not only the model weights and forthcoming inference code but also the complete data-cleaning code for training. This release includes high-quality synthetic data, an extensive set of checkpoints, and a dataset of over 4.5 million supervised fine-tuning (SFT) entries, making OpenCoder one of the most comprehensively open-sourced models available.
- **Comprehensive Experimental Analysis**: OpenCoder is rigorously tested through extensive ablation studies on various data-cleaning strategies and training processes, including file-level and repository-level deduplication experiments, ensuring thorough exploration and validation of the model’s performance.
- **High-Quality Synthetic Data**: OpenCoder provides a fully developed synthetic data generation process and over 4.5 million SFT data entries, establishing a robust data foundation for model training and evaluation.
- **Exceptional Performance**: OpenCoder achieves high performance across multiple language model benchmarks, positioning it among the leading open-source models for code.


## 2. Models

|         Model         | Sequence Length |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| OpenCoder-1.5B-Base  |      4K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Base)  |
| OpenCoder-8B-Base  |      8K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Base)  |
| OpenCoder-1.5B-Instruct  |      4K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Instruct) |
| OpenCoder-8B-Instruct  |      8K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Instruct) |


## 3. Datasets

### Pre-training
  
|         Dataset       | Size |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| fineweb-code-corpus  |      148 GB       | 🤗 [HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/fineweb-code-corpus)  |
| fineweb-math-corpus  |       10 GB    | 🤗 [HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/fineweb-math-corpus)  |


**This is not the end; we are organizing the remaining data and uploading it progressively.**

## 4. Benchmarks

**Note:** For the detailed evaluation results, please refer to [our paper](https://arxiv.org/pdf/2411.04905).

<!-- ### Base Model -->
|      model      | OpenCoder-1.5B-Base | OpenCoder-8B-Base |
|:---------------:|:-------------:|:------------:|
| HumanEval(+) |     54.3 (49.4)      |     66.5 (63.4)     |
| MBPP(+) |     70.6 (58.7)      |     79.9 (70.4)   |
| BigCodeBench |      24.5     |     40.5     |
| BigCodeBench-Hard |     5.4     |     9.5     |


<!-- ### Chat Model
|      model      | OpenCoder-1.5B-Instruct | OpenCoder-8B-Instruct |
|:---------------:|:-------------:|:------------:|
| HumanEval(+) |     72.5 (67.7)      |     83.5 (78.7)     |
| MBPP(+) |     72.7 (61.9)      |     79.1 (69.0)   |
| BigCodeBench |      33.3     |     40.3    |
| BigCodeBench-Hard |     11.5     |     16.9     |
| LiveCodeBench |     12.8     |     23.2     |
| MultiPL-E (AVG) |     57.5     |     71.0     | -->


## 5. Inference

### Inference with Huggingface's Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "infly/OpenCoder-1.5B-Base"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


inputs = tokenizer("# write a quick sort algorithm", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=128)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

<!-- ### Inference with vLLM (recommended) -->

## 6. License

OpenCoder series (including Base and Chat) support commercial applications under a permissive [License](https://huggingface.co/infly/OpenCoder-1.5B-Base/blob/main/LICENSE).

## 7. Citation
```
@inproceedings{Huang2024OpenCoderTO,
  title={OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models},
  author={Siming Huang and Tianhao Cheng and Jason Klein Liu and Jiaran Hao and Liuyihan Song and Yang Xu and J. Yang and J. H. Liu and Chenchen Zhang and Linzheng Chai and Ruifeng Yuan and Zhaoxiang Zhang and Jie Fu and Qian Liu and Ge Zhang and Zili Wang and Yuan Qi and Yinghui Xu and Wei Chu},
  year={2024},
  url={https://arxiv.org/pdf/2411.04905}
}
```