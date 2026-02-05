from collections import defaultdict
from dataclasses import dataclass, field, asdict
from timeit import default_timer as timer
import logging
import json
from pathlib import Path
from typing import Any, Optional, Iterator, Dict
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp
from itertools import tee
from contextlib import ExitStack
import gc
import math
import os
import torch
import numpy as np
from apps.main.train import (
    TrainArgs,
    prepare_opencoder_position_ids_numpy,
    get_cuseq_lens_and_max_seqlen,
)
from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dump_config, dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.data import (
    DataArgs,
    read_jsonl, 
    tokenize,
    distribute_data_to_rank,
    TRAIN_DATA_FILE_PATTERN
)
from lingua.distributed import (
    DistributedArgs,
    get_global_rank,
    get_device_mesh,
    setup_torch_distributed,
    get_world_size,
    get_is_master,
)
from lingua.tokenizer import (
    build_tokenizer,
)
from lingua.logger import init_logger

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

logger = logging.getLogger()


@dataclass
class BPBEvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    seed: int = 42

    use_train_data_config: bool = False
    use_train_doc_attn_mask_config: bool = False

    apply_doc_boundary_mask: bool = False

    val_batch_size: int = 4
    context_seqlen: int = 8192
    context_stride: int = 8192

    val_data_path: str = ""

    data: DataArgs = field(default_factory=DataArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)


def load_model_and_tokenizer_and_train_cfg(
    original_ckpt_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
):
    ckpt_path = Path(original_ckpt_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)

    default_train_cfg = OmegaConf.structured(TrainArgs())

    train_cfg = OmegaConf.merge(default_train_cfg, config)
    train_cfg = OmegaConf.to_object(train_cfg)

    assert train_cfg.data.tokenizer.name == "vanilla_hf"
    assert train_cfg.data.tokenizer.path.startswith("infly/OpenCoder")

    tokenizer = build_tokenizer(
        train_cfg.data.tokenizer.name, 
        train_cfg.data.tokenizer.path,
        None,
        None,
        False,
    )
    model = model_cls(model_args)
    _model_optimizer_state_dict: STATE_DICT_TYPE = {}
    _load_state_dict(
        _model_optimizer_state_dict,
        storage_reader=FileSystemReader(original_ckpt_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    model.load_state_dict(_model_optimizer_state_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer, train_cfg

def doc_mask_iter(
    data_loader,
    eos_id: int,
    apply_doc_boundary_mask: bool,
):
    for (batch, loss_mask) in data_loader:
        loss_mask = torch.from_numpy(loss_mask)
        if apply_doc_boundary_mask:
            position_ids = prepare_opencoder_position_ids_numpy(
                batch[:, :, 0],
                eos_token_id=eos_id,
            )
            # Calculate cuseq_lens and max_seqlen for flash attention varlen
            cuseq_lens, max_seqlen = get_cuseq_lens_and_max_seqlen(
                batch[:, :, 0],
                eos_id,
            )
        else:
            position_ids = None
            cuseq_lens = None
            max_seqlen = None
        yield (batch, loss_mask, cuseq_lens, max_seqlen, position_ids)

def collate_batch_iter(
    packed_dict_iter: Iterator[Dict[str, Any]],
    batch_size: int,
):
    buffer_for_batching = []
    for packed_dict in packed_dict_iter:
        input_ids = packed_dict["input_ids"]
        labels = packed_dict["labels"]
        loss_mask = packed_dict["loss_mask"]

        input_output = np.stack([input_ids, labels], axis=-1)
        buffer_for_batching.append((input_output, loss_mask))

        if len(buffer_for_batching) == batch_size:
            io_list, loss_mask_list = zip(*buffer_for_batching)
            yield np.stack(io_list, axis=0), np.stack(loss_mask_list, axis=0)
            buffer_for_batching = []

    if buffer_for_batching:
        io_list, loss_mask_list = zip(*buffer_for_batching)
        yield np.stack(io_list, axis=0), np.stack(loss_mask_list, axis=0)

def pack_tokens_with_stride_iter(
    data_iter: Iterator[Dict[str, Any]],
    tokenizer,
    stride: int,
    max_seq_len: int,
):
    cur_stride = (
        min(max_seq_len, stride)
        if stride != -1 else max_seq_len
    )

    assert cur_stride > 0

    buffer = []

    for tokens, _ in data_iter:
        buffer.extend(tokens)

    num_effective_tokens = len([token for token in buffer if token != tokenizer.eos_id and token != tokenizer.bos_id])

    start_idx = 0
    end_idx = 0
    prev_end_idx = 0
    actual_tokens_evaluated = 0
    additional_seq_len = 1

    while True:
        end_idx = start_idx + max_seq_len
        if end_idx + additional_seq_len > len(buffer):
            # The token fuffer is empty! The token batch would be filled with paddings.
            new_tokens = np.full([max_seq_len + additional_seq_len], tokenizer.eos_id, dtype=np.int32)
            new_tokens[:len(buffer[start_idx:])] = buffer[start_idx:]
            input_ids = new_tokens[:-1]
            labels = new_tokens[1:]
            loss_mask = np.zeros(max_seq_len, dtype=np.int32)
            loss_mask[prev_end_idx - start_idx : end_idx - start_idx] = 1
            loss_mask[labels == tokenizer.eos_id] = 0
            loss_mask[labels == tokenizer.bos_id] = 0
            actual_tokens_evaluated += np.sum(loss_mask)
            packed_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
            }
            yield packed_dict
            break
        else:
            # seq = 512, stride = 128
            # iter 0 : 0:512, mask 0:512
            # iter 1 : 128:640, mask 512:640 = 512-128 : 640-128
            # iter 2 : 256:768, mask 640:768 = 640-256 : 768-256
            # iter 3 : 384:896, mask 768:896
            loss_mask = np.zeros(max_seq_len, dtype=np.int32)
            loss_mask[prev_end_idx - start_idx : end_idx - start_idx] = 1
            input_ids = np.array(buffer[start_idx : end_idx])
            labels = np.array(buffer[start_idx + 1 : end_idx + 1])
            loss_mask[labels == tokenizer.eos_id] = 0
            loss_mask[labels == tokenizer.bos_id] = 0
            actual_tokens_evaluated += np.sum(loss_mask)
            packed_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
            }
            yield packed_dict
            start_idx = start_idx + cur_stride
            prev_end_idx = end_idx

    assert num_effective_tokens == actual_tokens_evaluated, "Mismatch {} {}".format(
        num_effective_tokens, actual_tokens_evaluated
    )

def val_data_iter(
    eval_data_path: str,
    rank: int,
    world_size: int,
    compression_sampling_rate: float,
    raw_compression_mix_option: str,
    add_bos: bool,
    add_eos: bool,
    tokenizer_name: str,
    tokenizer_path: str,
    spm_byte_path: str,
    byte_conversion_args: Dict[str, Any],
    compression_alg_config: str,
    separate_embedding: bool,
    tokenizer: Any,
    max_seq_len: int,
    stride: int,
    batch_size: int,
    apply_doc_boundary_mask: bool,
    eos_id: int,
):
    jsonl_state = distribute_data_to_rank(
        eval_data_path,
        rank,
        world_size,
        TRAIN_DATA_FILE_PATTERN,
        compression_sampling_rate,
        raw_compression_mix_option
    )

    # Create two iterators from the same source
    data_iter, byte_count_iter = tee(
        read_jsonl(
            jsonl_state["file_path"],
            jsonl_state["position"],
            jsonl_state["block_size"],
            jsonl_state["offset"],
            jsonl_state["current_iter"],
            jsonl_state["compression_rng_state"],
            jsonl_state["compression_sampling_rate"],
            jsonl_state["raw_compression_mix_option"],
        ),
        2
    )
    
    # Count bytes using one iterator
    total_bytes = sum(len(payload["utf8_bytes"]) for payload, _ in byte_count_iter)

    data_iter = tokenize(
        data_iter,
        add_bos,
        add_eos,
        tokenizer_name,
        tokenizer_path,
        spm_byte_path,
        byte_conversion_args,
        compression_alg_config,
        separate_embedding
    )

    data_iter = pack_tokens_with_stride_iter(data_iter, tokenizer, stride, max_seq_len)

    data_iter = collate_batch_iter(data_iter, batch_size)

    data_iter = doc_mask_iter(data_iter, eos_id, apply_doc_boundary_mask)

    return data_iter, total_bytes

def launch_eval(cfg: BPBEvalArgs):
    with ExitStack() as context_stack:
        if (
            cfg.distributed.dp_replicate
            * cfg.distributed.dp_shard
            * cfg.distributed.tp_size
            != get_world_size()
        ):
            assert get_world_size() % cfg.distributed.dp_shard == 0
            cfg.distributed.dp_replicate = get_world_size() // cfg.distributed.dp_shard

            assert cfg.distributed.dp_replicate % cfg.distributed.tp_size == 0
            cfg.distributed.dp_replicate = (
                cfg.distributed.dp_replicate // cfg.distributed.tp_size
            )
            logger.warning(
                f"Setting Data Parallel size to {cfg.distributed.dp_replicate * cfg.distributed.dp_shard}"
            )
            assert (
                cfg.distributed.dp_replicate
                * cfg.distributed.dp_shard
                * cfg.distributed.tp_size
                == get_world_size()
            )

            if cfg.distributed.fsdp_type == "no_shard":
                assert (
                    cfg.distributed.dp_shard == 1
                    and cfg.distributed.dp_replicate == get_world_size()
                )

        if get_is_master():
            Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
            dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)
        init_logger(Path(cfg.dump_dir) / "eval.log")
        if not torch.distributed.is_initialized():
            setup_torch_distributed(cfg.distributed)

        world_mesh = get_device_mesh(cfg.distributed)
        logger.info(f"Starting job: {cfg.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if cfg.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(cfg.seed)
        if (
            Path(cfg.ckpt_dir).exists()
            and (Path(cfg.ckpt_dir) / "params.json").exists()
            and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
        ):
            consolidate_path = Path(cfg.ckpt_dir)
        else:
            consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
            if not consolidate_path.exists() and get_global_rank() == 0:
                consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

        consolidate_path = str(consolidate_path)
        torch.distributed.barrier()
        logger.info("Loading model")

        model, tokenizer, train_cfg = load_model_and_tokenizer_and_train_cfg(
            cfg.ckpt_dir,
            model_cls=LMTransformer,
            model_args_cls=LMTransformerArgs,
        )

        logger.info("Model loaded")
        model.eval()

        logger.info(model)

        torch.distributed.barrier()

        if cfg.use_train_doc_attn_mask_config:
            logger.info("Using training-time doc attention mask configuration")
            apply_doc_boundary_mask = train_cfg.apply_doc_boundary_mask
        else:
            logger.info("Using passed apply_doc_boundary_mask")
            apply_doc_boundary_mask = cfg.apply_doc_boundary_mask

        if cfg.use_train_data_config:
            logger.info("Using training-time data configuration")
            _data_cfg = train_cfg.data
        else:
            logger.info("Using evaluation-time data configuration")
            _data_cfg = cfg.data

        logger.info(f"Loading data from {cfg.val_data_path}")
        gc.disable()

        val_data_loader, num_bytes = val_data_iter(
            eval_data_path=cfg.val_data_path,
            rank=dp_rank,
            world_size=dp_degree,
            compression_sampling_rate=_data_cfg.compression_sampling_rate,
            raw_compression_mix_option=_data_cfg.raw_compression_mix_option,
            add_bos=_data_cfg.add_bos,
            add_eos=_data_cfg.add_eos,
            tokenizer_name=_data_cfg.tokenizer.name,
            tokenizer_path=_data_cfg.tokenizer.path,
            spm_byte_path=_data_cfg.tokenizer.spm_byte_path,
            byte_conversion_args=asdict(_data_cfg.tokenizer.byte_converter_config),
            compression_alg_config=_data_cfg.compression_alg_config,
            separate_embedding=_data_cfg.tokenizer.separate_embedding,
            tokenizer=tokenizer,
            max_seq_len=cfg.context_seqlen,
            stride=cfg.context_stride,
            batch_size=cfg.val_batch_size,
            apply_doc_boundary_mask=apply_doc_boundary_mask,
            eos_id=tokenizer.eos_id,
        )

        torch.distributed.barrier()

        metrics = defaultdict(float)
        eval_log_every_n_steps = 50
        cur_iter = 0
        metrics["num_bytes"] = num_bytes
        while True:
            # get batch
            data_load_start = timer()
            with torch.no_grad():
                try:
                    (batch, loss_mask, cuseq_lens, max_seqlen, position_ids) = next(val_data_loader)
                except StopIteration:
                    break
                
                batch = torch.tensor(
                    batch,
                    dtype=torch.long,
                )

                if apply_doc_boundary_mask:
                    position_ids = torch.tensor(
                        position_ids,
                        dtype=torch.long,
                    ).cuda()
                    cuseq_lens = torch.tensor(
                        cuseq_lens,
                        dtype=torch.int32,
                    ).cuda()
                    attn_impl = "flash_attn2"
                else:
                    position_ids = None
                    cuseq_lens = None
                    attn_impl = "sdpa"

                input_ids = batch[:, :, 0].cuda()
                labels = batch[:, :, 1:].cuda()

                loss_mask = loss_mask.float().cuda()
                data_load_time = round(timer() - data_load_start, 4)

                # forward
                start_timer = torch.cuda.Event(enable_timing=True)
                end_timer = torch.cuda.Event(enable_timing=True)
                start_timer.record()

                logits = model(
                    input_ids,
                    target=None,
                    tok_idx=position_ids,
                    attn_impl=attn_impl, 
                    cuseq_lens=cuseq_lens,
                    max_seqlen=max_seqlen
                )

                end_timer.record()
                total_loss = F.cross_entropy(
                    logits.to(torch.float32).view(-1, logits.shape[-1]), 
                    labels.view(-1), 
                    reduction="none"
                )
                loss_mask = loss_mask.view(-1)
                current_loss = torch.sum(total_loss * loss_mask)
                current_token_count = torch.sum(loss_mask)  # tokens on device
                metrics["loss"] += current_loss.item()
                metrics["num_tokens"] += current_token_count.item()

                torch.cuda.synchronize()
                curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)
                cur_iter += 1
                if cur_iter % eval_log_every_n_steps == 0:
                    logger.info(f"Validation step {cur_iter} done. Time taken: data load {data_load_time}s, forward {curr_iter_time}s")

        logger.info(f"This rank process finished running through {cur_iter} validation steps. Waiting for the other ranks...")
        torch.distributed.barrier()
        logger.info(f"Metrics before reduction: {metrics}")
        for m in metrics:
            tensor = torch.tensor(metrics[m]).cuda()
            dist.all_reduce(tensor, op=ReduceOp.SUM)
            metrics[m] = tensor.item()
        logger.info(f"Metrics after reduction: {metrics}")
        logger.info(f"Validation done. Now calculating metrics...")

        loss_per_token = metrics["loss"] / metrics["num_tokens"] if metrics["num_tokens"] > 0 else 0.0
        bits_per_token = loss_per_token / math.log(2)  # log(e)/log(2)

        if "num_bytes" in metrics and metrics["num_bytes"] > 0:
            token_per_bytes = metrics["num_tokens"] / metrics["num_bytes"]
        else:
            token_per_bytes = -1.0
        
        # Just return metrics related to the loss.
        metrics = {
            "total_loss": metrics["loss"],
            "nats_per_token": loss_per_token,
            "bits_per_token": bits_per_token,
            "bits_per_byte": bits_per_token * token_per_bytes if token_per_bytes > 0 else -1.0,
            "num_tokens": metrics["num_tokens"],
            "num_bytes": metrics["num_bytes"],
        }
        logger.info(
            "Byte Model Evaluation:\n"
            "    ===> Sequence Length : {} Context Stride : {} \n"
            "    ===> Total Loss : {:.4f}, Total tokens : {} \n Total bytes : {} \n"
            "    ===> Avg Loss (Base e) : {:.4f}, Avg Loss (Base 2) : {:.4f}, BPB: {:.2f} \n".format(
                cfg.context_seqlen, cfg.context_stride,
                metrics["total_loss"], metrics["num_tokens"], metrics["num_bytes"],
                metrics["nats_per_token"], metrics["bits_per_token"], metrics["bits_per_byte"]
            )
        )
        if get_global_rank() == 0:
            val_results_path = Path(cfg.dump_dir) / "val_results.json"
            with open(val_results_path, "w") as f:
                f.write(json.dumps(metrics))
            logger.info(f"All evaluation results saved to {val_results_path}")

def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config
    if hasattr(cli_args, "data") and hasattr(cli_args.data, "sources"):
        sources_cli = cli_args.data.sources
        del cli_args.data.sources
    else:
        sources_cli = None

    default_cfg = OmegaConf.structured(BPBEvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    if sources_cli:
        cfg.data.sources = dict(sources_cli)
    launch_eval(cfg)


if __name__ == "__main__":
    main()
