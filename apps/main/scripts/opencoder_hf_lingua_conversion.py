# this script converts HF to Lingua-format checkpoints

# adapted from
# 1. torchtune's download CLI for downloading models from HF
# 2. torchtune's checkpointer for checkpointing models
# 3. torchtune's file utils mostly from torchtune/torchtune/training/checkpointing/_utils.py
# 4. omegaconf for loading config files

# download HF checkpoints to local directory
#   python -m apps.main.scripts.opencoder_hf_lingua_conversion mode=download download_args.repo_id=infly/OpenCoder-1.5B-Base download_args.output_dir=ckpts/opencoder-1.5B
# convert Lingua-format checkpoints to HF checkpoints
#   python -m apps.main.scripts.opencoder_hf_lingua_conversion mode=dcp_to_hf dcp_to_hf_args.dcp_checkpoint_dir=/home/lin/code/byte_lingua/checkpoints/checkpoints/0000000600 dcp_to_hf_args.hf_output_dir=ckpts/opencoder-1.5B-hf dcp_to_hf_args.ref_hf_output_dir=ckpts/opencoder-1.5B

from dataclasses import dataclass, field
import os
import traceback
import json
import yaml
from pathlib import Path
import gc
import shutil
import re
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
import torch
import torch.distributed.checkpoint as DCP
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Optional, Union, Literal
from safetensors import safe_open
from safetensors.torch import save_file

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner,
)
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

SAFETENSOR_INDEX_FNAME = "model.safetensors.index.json"
SHARD_FNAME = "model-{cpt_idx}-of-{num_shards}"
SUFFIXES_TO_NOT_COPY = [
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    SAFETENSOR_INDEX_FNAME,
]

def copy_files(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    ignore_suffixes: Optional[List[str]] = None,
    max_file_size_mb: int = 100,
) -> None:
    """
    Copies files from the input directory to the output directory, preserving the directory structure.

    This function will skip copying files that already exist in the output directory or have specific suffixes.
    It will also skip folders and files that start with '.'. E.g. ".cache/" and ".git".

    Args:
        input_dir (Union[str, Path]): The path to the input directory containing files to be copied.
        output_dir (Union[str, Path]): The path to the output directory where files should be copied.
        ignore_suffixes (Optional[List[str]]): A list of file suffixes to exclude from copying.
          Defaults to ['.pt', '.bin', '.safetensors'] if not provided.
        max_file_size_mb (int): The maximum file size in megabytes to copy. Defaults to 100 MB.
    Returns:
        None
    Example:
    >>> copy_files('path/to/input_dir', 'path/to/output_dir')

    This will copy all files from 'path/to/input_dir' to 'path/to/output_dir', except those that
    already exist in the destination or have the specified suffixes.
    """

    max_file_size = max_file_size_mb * 1024 * 1024
    for root, dirs, files in os.walk(input_dir):

        # Filter out directories that start with '.'. E.g. ".cache/"
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Construct the corresponding directory in the output
        relative_path = os.path.relpath(root, input_dir)
        dest_dir = os.path.join(output_dir, relative_path)

        # Create the directory in the output if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            # Skip files that start with '.'. E.g. ".git"
            if file.startswith("."):
                continue

            # Check if the file has one of the specified suffixes
            if ignore_suffixes and any(
                file.endswith(suffix) for suffix in ignore_suffixes
            ):
                continue

            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)

            # Check the file size
            if os.path.getsize(src_file) > max_file_size:
                print(
                    f"Skipping copying {src_file} to {output_dir} as it exceeds the size limit of {max_file_size_mb} MiB."
                )
                continue

            # Copy the file if it doesn't already exist in the destination
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)

    return

def get_model_checkpoint_path(
    checkpoint_dir: Union[str, Path],
    checkpoint_file_suffix: str,
) -> list[Path]:
    if not checkpoint_dir.is_dir():
        raise ValueError(f"{checkpoint_dir} is not a valid directory.")

    checkpoint_paths: List[Path] = []
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(checkpoint_file_suffix):
            continue
        checkpoint_path = Path.joinpath(checkpoint_dir, f)
        if not checkpoint_path.is_file():
            raise ValueError(f"No file with name: {f} found in {checkpoint_dir}.")
        checkpoint_paths.append(checkpoint_path)

    return sorted(checkpoint_paths)



def safe_torch_load(
    checkpoint_path: Union[Path, str], weights_only: bool = True, mmap: bool = True
) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file onto CPU in a safe manner. Provides separate handling for
    safetensors files.

    Args:
        checkpoint_path (Union[Path, str]): Path to the checkpoint file.
        weights_only (bool): Whether to load only tensors, primitive types, and dictionaries
            (passthrough to torch.load). Default: True
        mmap (bool): Whether to mmap from disk into CPU memory. Default: True

    Returns:
        Dict[str, Any]: State dict from the checkpoint file.

    Raises:
        ValueError: If the checkpoint file is not found or cannot be loaded.
    """
    try:
        # convert the path into a string since pathlib Path and mmap don't work
        # well together
        is_safetensors_file = (
            True if str(checkpoint_path).endswith(".safetensors") else False
        )
        if is_safetensors_file:
            result = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    result[k] = f.get_tensor(k)
            state_dict = result
        else:
            state_dict = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                mmap=mmap,
                weights_only=weights_only,
            )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict

def convert_layer_key_to_template(key: str) -> tuple[str, int]:
    """
    Convert a specific layer key to template format and extract layer number.
    
    Args:
        key: Specific layer key like "layers.5.attention.wq.weight"
        
    Returns:
        tuple: (template_key, layer_number) like ("layers.{}.attention.wq.weight", 5)
    """
    # Pattern to match layer numbers in the key
    pattern = r'layers\.(\d+)\.'
    match = re.search(pattern, key)
    
    if not match:
        raise ValueError(f"Could not extract layer number from key: {key}")
    
    layer_num = int(match.group(1))
    template_key = re.sub(pattern, 'layers.{}.', key)
    
    return template_key, layer_num

@dataclass
class DownloadArgs:
    repo_id: str = ""
    output_dir: Path = Path("/tmp")
    hf_token: Optional[str] = None
    ignore_patterns: Optional[List[str]] = None

@dataclass
class HFToDCPArgs:
    hf_output_dir: Optional[Path] = None
    dcp_checkpoint_dir: Optional[Path] = None

@dataclass
class DCPToHFArgs:
    dcp_checkpoint_dir: Optional[Path] = None
    hf_output_dir: Optional[Path] = None
    ref_hf_output_dir: Optional[Path] = None

@dataclass
class ConversionArgs:
    mode: str
    download_args: DownloadArgs = field(default_factory=DownloadArgs)
    hf_to_dcp_args: HFToDCPArgs = field(default_factory=HFToDCPArgs)
    dcp_to_hf_args: DCPToHFArgs = field(default_factory=DCPToHFArgs)

def download(args: DownloadArgs) -> None:
    """Downloads a model from the Hugging Face Hub."""
    # Download the tokenizer and PyTorch model files

    # Default output_dir is `/tmp/<model_name>`
    output_dir = args.output_dir
    if output_dir is None:
        model_name = args.repo_id.split("/")[-1]
        output_dir = Path("/tmp") / model_name

    print(f"Ignoring files matching the following patterns: {args.ignore_patterns}")
    try:
        true_output_dir = snapshot_download(
            args.repo_id,
            local_dir=output_dir,
            ignore_patterns=args.ignore_patterns,
            token=args.hf_token,
        )
    except GatedRepoError:
        if args.hf_token:
            raise ValueError(
                "It looks like you are trying to access a gated repository. Please ensure you "
                "have access to the repository."
            )
        else:
            raise ValueError(
                "It looks like you are trying to access a gated repository. Please ensure you "
                "have access to the repository and have provided the proper Hugging Face API token "
                "using the option `--hf-token` or by running `huggingface-cli login`."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )
    except RepositoryNotFoundError:
        raise ValueError(
            f"Repository '{args.repo_id}' not found on the Hugging Face Hub."
        )
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"Failed to download {args.repo_id} with error: '{e}' and traceback: {tb}"
        raise ValueError(msg)

    print(
        "Successfully downloaded model repo and wrote to the following locations:",
        *list(Path(true_output_dir).iterdir()),
        sep="\n",
    )

def dcp_to_hf(args: DCPToHFArgs) -> None:
    hf_output_dir = args.hf_output_dir
    ref_hf_output_dir = args.ref_hf_output_dir
    dcp_checkpoint_dir = args.dcp_checkpoint_dir
    hf_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading initial model from {dcp_checkpoint_dir}")
    # we assume the dcp checkpoint directory has the format of
    # /path/to/parent_dir/checkpoints/0000000{steps}
    # and the config.yaml is in the parent_dir
    # we need to find the config.yaml in the parent_dir
    config_path = Path(dcp_checkpoint_dir) / 'params.json'
    with open(config_path, 'r') as f:
        config = json.load(f)["model"]
    
    hf_config_path = Path(ref_hf_output_dir) / 'config.json'
    with open(hf_config_path, 'r') as f:
        hf_config = json.load(f)

    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    dim = config["dim"]
    dims_per_head = dim // n_heads
    base = config["rope_theta"]
    if config["n_kv_heads"] is not None:
        num_key_value_heads = config["n_kv_heads"]
        key_value_dim = dims_per_head * num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        key_value_dim = dim

    hf_config["hidden_size"] = dim
    hf_config["num_attention_heads"] = n_heads
    hf_config["num_key_value_heads"] = num_key_value_heads
    hf_config["num_hidden_layers"] = n_layers
    hf_config["intermediate_size"] = config["ffn_dim"]
    hf_config["max_position_embeddings"] = config["max_seqlen"]
    hf_config["rms_norm_eps"] = config["norm_eps"]
    hf_config["vocab_size"] = config["vocab_size"]
    hf_config["rope_theta"] = base

    _model_optimizer_state_dict: STATE_DICT_TYPE = {}
    _load_state_dict(
        _model_optimizer_state_dict,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(_model_optimizer_state_dict['model'].keys())

    if (Path(ref_hf_output_dir) / SAFETENSOR_INDEX_FNAME).exists():
        with open(Path(ref_hf_output_dir) / SAFETENSOR_INDEX_FNAME, 'r') as f:
            ref_weight_map = json.load(f)['weight_map']
        default_shard_name = None
    else:
        ref_weight_map = None
        default_shard_name = "model-00001-of-00001.safetensors"

    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    state_dict = _model_optimizer_state_dict['model']
    split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
    total_size = 0
    # MODEL_WEIGHTS_KEY_MAP = {
    #     "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    #     "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    #     "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    #     "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    #     "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    #     "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    #     "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    #     "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    #     "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    #     "tok_embeddings.weight": "model.embed_tokens.weight",
    #     "norm.weight": "model.norm.weight",
    #     "output.weight": "lm_head.weight",
    # }
    def write_to_weight_map(new_key_weight):
        _cur_total_size = 0
        for new_key, weight in new_key_weight.items():
            shard_name = ref_weight_map.get(new_key, default_shard_name) if ref_weight_map else default_shard_name

            # initialize dict
            if shard_name not in split_state_dicts:
                split_state_dicts[shard_name] = {}

            split_state_dicts[shard_name].update({new_key: weight})
            _cur_total_size += weight.numel() * weight.element_size()
            weight_map[new_key] = shard_name
        return _cur_total_size

    weight_map = {}
    for layer_i in range(n_layers):
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
        new_key_weight_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                state_dict[f"layers.{layer_i}.attention.wq.weight"], n_heads=n_heads
            ),
            f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                state_dict[f"layers.{layer_i}.attention.wk.weight"],
                n_heads=num_key_value_heads,
                dim1=key_value_dim,
            ),
            f"model.layers.{layer_i}.self_attn.v_proj.weight": state_dict[f"layers.{layer_i}.attention.wv.weight"],
            f"model.layers.{layer_i}.self_attn.o_proj.weight": state_dict[f"layers.{layer_i}.attention.wo.weight"],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": state_dict[f"layers.{layer_i}.feed_forward.w1.weight"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": state_dict[f"layers.{layer_i}.feed_forward.w2.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": state_dict[f"layers.{layer_i}.feed_forward.w3.weight"],
            f"model.layers.{layer_i}.input_layernorm.weight": state_dict[
                f"layers.{layer_i}.attention_norm.weight"
            ],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": state_dict[
                f"layers.{layer_i}.ffn_norm.weight"
            ],
            f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq,
        }
        total_size += write_to_weight_map(new_key_weight_dict)

    new_key_weight_dict = {
        "model.embed_tokens.weight": state_dict["tok_embeddings.weight"],
        "model.norm.weight": state_dict["norm.weight"],
        "lm_head.weight": state_dict["output.weight"],
    }
    total_size += write_to_weight_map(new_key_weight_dict)

    # write the partitioned state dicts to the right checkpoint file
    # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
    # num_shards = len(split_state_dicts)
    for shard_name, model_state_dict in split_state_dicts.items():
        # TODO: We should probably use the original shard name and just add a prefix
        # however, having the SHARD_FNAME standardizes our checkpoints
        # shard_name = SHARD_FNAME.format(
        #     cpt_idx=f"{cpt_idx}".zfill(5), num_shards=f"{num_shards}".zfill(5)
        # )
        # map_original_name_to_new_name[cpt_idx] = shard_name
        output_path = Path.joinpath(
            hf_output_dir, shard_name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_path.with_suffix(".safetensors")
        save_file(model_state_dict, output_path, metadata={"format": "pt"})

        print(
            "Model checkpoint of size "
            f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
            f"saved to {output_path}"
        )

    # Save the appropriate index file based on serialization format
    # e.g. {metadata: {total_size: 1234}, weight_map: {"key1": "model_0001.safetensors", "key2": "model_0002.safetensors"}}
    index_path = Path.joinpath(
        hf_output_dir, SAFETENSOR_INDEX_FNAME
    )

    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
    # So its easy to run inference with the model using this epoch's checkpoint
    copy_files(
        ref_hf_output_dir,
        hf_output_dir,
        ignore_suffixes=SUFFIXES_TO_NOT_COPY,
    )
    # overwrite config to match original lingua ckpts
    with open(Path(hf_output_dir) / 'config.json', 'w') as f:
        json.dump(hf_config, f, indent=2)

if __name__ == "__main__":
    default_cfg = OmegaConf.structured(ConversionArgs)
    cfg_cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(default_cfg, cfg_cli)
    if cfg.mode == "download":
        download(cfg.download_args)
    elif cfg.mode == "hf_to_dcp":
        raise NotImplementedError("Not implemented")
    elif cfg.mode == "dcp_to_hf":
        dcp_to_hf(cfg.dcp_to_hf_args)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")
