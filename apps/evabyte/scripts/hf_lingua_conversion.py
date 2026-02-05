# this script converts HF to Lingua-format checkpoints

# adapted from
# 1. torchtune's download CLI for downloading models from HF
# 2. torchtune's checkpointer for checkpointing models
# 3. torchtune's file utils mostly from torchtune/torchtune/training/checkpointing/_utils.py
# 4. omegaconf for loading config files

# download HF checkpoints to local directory
#   python -m apps.evabyte.scripts.hf_lingua_conversion mode=download download_args.repo_id=EvaByte/EvaByte download_args.output_dir=ckpts/EvaByte
# convert HF checkpoints to Lingua-format checkpoints
#   python -m apps.evabyte.scripts.hf_lingua_conversion mode=hf_to_dcp hf_to_dcp_args.hf_output_dir=ckpts/EvaByte hf_to_dcp_args.dcp_checkpoint_dir=ckpts/EvaByte-dcp
# convert Lingua-format checkpoints to HF checkpoints (discard raw_multibyte_lm_head, default behavior)
#   python -m apps.evabyte.scripts.hf_lingua_conversion mode=dcp_to_hf dcp_to_hf_args.dcp_checkpoint_dir=/path/to/dcp_ckpt dcp_to_hf_args.hf_output_dir=ckpts/EvaByte-hf dcp_to_hf_args.ref_hf_output_dir=ckpts/EvaByte
# convert Lingua-format checkpoints to HF checkpoints (merge raw_multibyte_lm_head into lm_head for pure byte-level model)
#   python -m apps.evabyte.scripts.hf_lingua_conversion mode=dcp_to_hf dcp_to_hf_args.dcp_checkpoint_dir=/path/to/dcp_ckpt dcp_to_hf_args.hf_output_dir=ckpts/EvaByte-hf dcp_to_hf_args.ref_hf_output_dir=ckpts/EvaByte dcp_to_hf_args.raw_multibyte_mode=merge

from dataclasses import dataclass, field
import os
import traceback
import json
import yaml
from pathlib import Path
import gc
import shutil
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
MODEL_HF_CONFIG_KEYS = [
  "hidden_size",
  "intermediate_size",
  "max_position_embeddings",
  "max_seq_length",
  "norm_add_unit_offset",
  "num_attention_heads",
  "num_hidden_layers",
  "num_key_value_heads",
  "num_pred_heads",
  "rms_norm_eps",
  "rope_theta",
  "vocab_size",
  "window_size",
  "chunk_size",
]

# Additional config keys needed for raw multibyte merge mode
RAW_MULTIBYTE_CONFIG_KEYS = [
  "raw_vocab_size",
  "num_raw_pred_heads",
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
    # raw_multibyte_mode controls how raw_multibyte_lm_head is handled:
    #   - "discard": skip raw_multibyte_lm_head.weight entirely (default, original behavior)
    #   - "merge": slice lm_head to raw_vocab_size and concatenate with raw_multibyte_lm_head,
    #              resulting in a pure byte-level model with (num_pred_heads + num_raw_pred_heads) heads
    raw_multibyte_mode: str = "discard"

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

def hf_to_dcp(args: HFToDCPArgs) -> Dict[str, Any]:
    hf_output_dir = args.hf_output_dir
    dcp_checkpoint_dir = args.dcp_checkpoint_dir
    _checkpoint_paths = get_model_checkpoint_path(
        checkpoint_dir=Path(hf_output_dir),
        checkpoint_file_suffix=".safetensors",
    )

    # merged state_dict contains keys and weights from all the checkpoint files
    merged_state_dict: Dict[str, torch.Tensor] = {}

    # converted_state_dict is the final state_dict passed to the recipe after the
    # keys are converted into the torchtune format. This optionally also contains
    # the recipe state and adapter weights
    converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    # _checkpoint_paths are already sorted so simply enumerate to generate the right id
    for cpt_idx, cpt_path in enumerate(_checkpoint_paths):
        state_dict = safe_torch_load(cpt_path)
        for key, value in state_dict.items():
            # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
            # will break recipe code
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Expected all values in the state dict to be torch.Tensor. "
                    f"Found {type(value)} instead."
                )
            # idx is written in the 4 digit format (eg: 0001, 0002, etc.)
        merged_state_dict.update(state_dict)

        # delete the state_dict to free up memory; TODO check if this del is needed
        del state_dict
        gc.collect()

    converted_state_dict = {}
    for key, value in merged_state_dict.items():
        if key.startswith("rotary_emb"):
            continue
        if key.startswith("model."):
            new_key = key[len("model."):]
        else:
            new_key = key
        converted_state_dict[new_key] = value

    print(f"Writing to DCP at '{dcp_checkpoint_dir}'")
    dcp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(dcp_checkpoint_dir, thread_count=8)
    DCP.save({"model": converted_state_dict}, storage_writer=storage_writer)

def dcp_to_hf(args: DCPToHFArgs) -> None:
    hf_output_dir = args.hf_output_dir
    ref_hf_output_dir = args.ref_hf_output_dir
    dcp_checkpoint_dir = args.dcp_checkpoint_dir
    raw_multibyte_mode = args.raw_multibyte_mode
    hf_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading initial model from {dcp_checkpoint_dir}")
    print(f"Raw multibyte mode: {raw_multibyte_mode}")
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

    for key in MODEL_HF_CONFIG_KEYS:
        if key in config:
            hf_config[key] = config[key]
        elif key == "max_seq_length" and "max_position_embeddings" in config:
            hf_config[key] = config["max_position_embeddings"]
        else:
            raise ValueError(f"Key {key} not found in config.yaml")

    # Load raw multibyte config if in merge mode
    raw_vocab_size = None
    num_raw_pred_heads = None
    if raw_multibyte_mode == "merge":
        for key in RAW_MULTIBYTE_CONFIG_KEYS:
            if key not in config:
                raise ValueError(f"Key {key} not found in config but required for merge mode")
        raw_vocab_size = config["raw_vocab_size"]
        num_raw_pred_heads = config["num_raw_pred_heads"]
        print(f"Merge mode: raw_vocab_size={raw_vocab_size}, num_raw_pred_heads={num_raw_pred_heads}")
    # FIXME legacy attributes. Will be removed next version
    hf_config["fp32_ln"] = True
    hf_config["fp32_logits"] = True
    hf_config["fp32_skip_add"] = False

    _model_optimizer_state_dict: STATE_DICT_TYPE = {}
    _load_state_dict(
        _model_optimizer_state_dict,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(_model_optimizer_state_dict['model'].keys())

    with open(Path(ref_hf_output_dir) / SAFETENSOR_INDEX_FNAME, 'r') as f:
        ref_weight_map = json.load(f)['weight_map']

    # split the state_dict into separate dicts, one for each output checkpoint file
    # e.g. split_state_dicts= {
    #       "0001": {"key1": tensor1, "key2": tensor2},
    #       "0002": {"key3": tensor3}
    #       }
    state_dict = _model_optimizer_state_dict['model']
    split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
    total_size = 0

    weight_map = {}

    # Store lm_head weights for potential merging
    lm_head_weight = None
    raw_multibyte_lm_head_weight = None

    for key, weight in state_dict.items():
        # Handle raw_multibyte_lm_head based on mode
        if "raw_multibyte_lm_head.weight" in key:
            if raw_multibyte_mode == "merge":
                raw_multibyte_lm_head_weight = weight
            # Skip in both modes - we'll handle it separately for merge mode
            continue

        # Handle lm_head specially for merge mode
        if key == "lm_head.weight" and raw_multibyte_mode == "merge":
            lm_head_weight = weight
            continue

        # map the key to HF's format
        if key != "lm_head.weight":
            new_key = "model." + key
        else:
            new_key = key
        try:
            shard_name = ref_weight_map[new_key]
        except:
            shard_name = "model-00003-of-00003.safetensors"

        # initialize dict
        if shard_name not in split_state_dicts:
            split_state_dicts[shard_name] = {}

        split_state_dicts[shard_name].update({new_key: weight})
        total_size += weight.numel() * weight.element_size()
        weight_map[new_key] = shard_name

    # Handle lm_head merging for merge mode
    if raw_multibyte_mode == "merge":
        if lm_head_weight is None or raw_multibyte_lm_head_weight is None:
            raise ValueError(
                "Merge mode requires both lm_head.weight and raw_multibyte_lm_head.weight to be present"
            )

        vocab_size = config["vocab_size"]
        num_pred_heads = config["num_pred_heads"]
        hidden_size = config["hidden_size"]

        # lm_head.weight: [vocab_size * num_pred_heads, hidden_size]
        # Reshape to [num_pred_heads, vocab_size, hidden_size]
        lm_head_reshaped = lm_head_weight.view(num_pred_heads, vocab_size, hidden_size)
        # Slice to only raw_vocab_size: [num_pred_heads, raw_vocab_size, hidden_size]
        lm_head_sliced = lm_head_reshaped[:, :raw_vocab_size, :]

        # raw_multibyte_lm_head.weight: [raw_vocab_size * num_raw_pred_heads, hidden_size]
        # Reshape to [num_raw_pred_heads, raw_vocab_size, hidden_size]
        raw_head_reshaped = raw_multibyte_lm_head_weight.view(
            num_raw_pred_heads, raw_vocab_size, hidden_size
        )

        # Concatenate along heads dimension: [(num_pred_heads + num_raw_pred_heads), raw_vocab_size, hidden_size]
        merged_head = torch.cat([lm_head_sliced, raw_head_reshaped], dim=0)
        # Flatten back: [(num_pred_heads + num_raw_pred_heads) * raw_vocab_size, hidden_size]
        merged_total_heads = num_pred_heads + num_raw_pred_heads
        merged_lm_head_weight = merged_head.view(merged_total_heads * raw_vocab_size, hidden_size)

        print(f"Merged lm_head: original shape {lm_head_weight.shape} -> {merged_lm_head_weight.shape}")
        print(f"  - Sliced lm_head from vocab_size={vocab_size} to raw_vocab_size={raw_vocab_size}")
        print(f"  - Concatenated {num_pred_heads} + {num_raw_pred_heads} = {merged_total_heads} prediction heads")

        # Update hf_config for the merged model
        hf_config["vocab_size"] = raw_vocab_size
        hf_config["num_pred_heads"] = merged_total_heads

        # Add merged lm_head to split_state_dicts
        new_key = "lm_head.weight"
        try:
            shard_name = ref_weight_map[new_key]
        except:
            shard_name = "model-00003-of-00003.safetensors"

        if shard_name not in split_state_dicts:
            split_state_dicts[shard_name] = {}

        split_state_dicts[shard_name].update({new_key: merged_lm_head_weight})
        total_size += merged_lm_head_weight.numel() * merged_lm_head_weight.element_size()
        weight_map[new_key] = shard_name

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
        hf_to_dcp(cfg.hf_to_dcp_args)
    elif cfg.mode == "dcp_to_hf":
        dcp_to_hf(cfg.dcp_to_hf_args)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")
