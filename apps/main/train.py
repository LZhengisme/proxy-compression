# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from copy import deepcopy
import gc
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional
from functools import partial
import contextlib
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    PrefetchState,
    async_iterator,
    loop_on_jsonl,
    setup_sources,
    choose_source,
    pack_tokens,
    tokenize,
    batch_and_shuffle_prefetched_sequences,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    requeue_slurm_job,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.tokenizer import build_tokenizer
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
)
from lingua.probe import AutoProbeD
try:
    logging.getLogger("databus.databus_cache").setLevel(logging.WARNING)
except:
    pass
logger = logging.getLogger()


@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""
    log_dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    # Enable doc masking or not
    apply_doc_boundary_mask: bool = False
    
    override_saved_optim_and_data_config: bool = False
    continual_training_path: Optional[str] = None
    
    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

def prepare_opencoder_position_ids_numpy(
    input_ids: np.ndarray,
    eos_token_id: int,
) -> np.ndarray:
    """
    NumPy implementation of prepare_doc_mask_position_ids.
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        eos_token_id: Token ID that marks end of document
        
    Returns:
        position_ids: Position IDs with resets at document boundaries, shape (batch_size, seq_len)
    """
    bs, seq_len = input_ids.shape
    position_ids = []

    for b in range(bs):
        position_id = np.arange(0, seq_len, dtype=np.int64)
        # Find indices where EOS token is
        eos_mask = (input_ids[b] == eos_token_id)
        eos_indices = np.where(eos_mask)[0]

        # Loop through EOS indices and reset positions
        prev_index = 0
        for i in eos_indices:
            # Reset positions after each EOS token
            position_id[(i + 1):] -= (i + 1 - prev_index)
            prev_index = i + 1
            
        position_ids.append(position_id)
    
    position_ids = np.stack(position_ids, axis=0)
    return position_ids

def get_cuseq_lens_and_max_seqlen(input_ids, eos_id):
    """
    Calculates cuseq_lens and max_seqlen for varlen attention.
    
    Args:
        input_ids (np.array): A 2D array of token IDs from packed documents.
        eos_id (int): The end-of-sentence token ID.
        
    Returns:
        tuple[np.array, int]: A tuple containing the cuseq_lens array and max_seqlen integer.
    """
    flattened_input_ids = input_ids.reshape(-1)
    end_of_doc_indices = np.where(flattened_input_ids == eos_id)[0]

    if len(end_of_doc_indices) == 0:
        cuseq_lens = np.array([0, flattened_input_ids.shape[0]], dtype=np.int32)
    else:
        # The cumulative sequence lengths are the indices of the EOS tokens + 1.
        # For example, if an EOS is at index 49, the sequence length is 50.
        cuseq_lens = end_of_doc_indices + 1
        cuseq_lens = np.insert(cuseq_lens, 0, 0)
        if cuseq_lens[-1] < flattened_input_ids.shape[0]:
            cuseq_lens = np.append(cuseq_lens, flattened_input_ids.shape[0])
        cuseq_lens = cuseq_lens.astype(np.int32)
        
    # Calculate the length of each individual sequence by taking the difference
    # of the cumulative lengths.
    max_seqlen = np.max(np.diff(cuseq_lens))
    return cuseq_lens, int(max_seqlen)

def opencoder_doc_mask_iter(
    data_loader,
    apply_doc_boundary_mask: bool,
    eos_id: int,
):
    for batch, state in data_loader:
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
        yield (batch, position_ids, cuseq_lens, max_seqlen), state

@contextlib.contextmanager
def build_opencoder_dataloader(
    state: PrefetchState,
    apply_doc_boundary_mask: bool,
    eos_id: int,
):
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]

    path_to_iter = setup_sources(multi_state)
    data_it = choose_source(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
    )
    data_it = tokenize(
        data_it,
        tokenizer_state["add_bos"],
        tokenizer_state["add_eos"],
        tokenizer_state["name"],
        tokenizer_state["path"],
        tokenizer_state["spm_byte_path"],
        tokenizer_state["byte_conversion_args"],
        tokenizer_state["compression_alg_config"],
        tokenizer_state["separate_embedding"]
    )

    data_it = pack_tokens(
        data_it,
        pack_state,
    )

    data_it = batch_and_shuffle_prefetched_sequences(
        data_loader=data_it,
        seq_len=pack_state["output_seq_len"],
        n_views=pack_state["n_views"],
        batch_size=state["batch_size"],
        prefetch_size=state["prefetch_size"],
        state=state,
    )

    data_it = opencoder_doc_mask_iter(
        data_it, 
        apply_doc_boundary_mask,
        eos_id,
    )
    yield data_it
    for it in path_to_iter.values():
        it.close()
    data_it.close()

def build_opencoder_dataloader_from_args(
    args: DataArgs,
    state: Optional[PrefetchState] = None,
    apply_doc_boundary_mask: bool = False,
    eos_id: int = 1,
):
    data_builder = partial(build_opencoder_dataloader, state, apply_doc_boundary_mask, eos_id)
    if args.load_async:
        return async_iterator(args.prefetch_size, data_builder)
    else:
        return data_builder()

def validate_train_args(args: TrainArgs):
    logger.info(f"model vocab size is set to {args.model.vocab_size}")

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        ckpt_path = str(Path(args.dump_dir) / "checkpoints")
        logger.info(f"Setting checkpoint path to {ckpt_path}")
        args.checkpoint.path = ckpt_path

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"
    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        _tokenizer = build_tokenizer(
            args.data.tokenizer.name, 
            args.data.tokenizer.path,
            args.data.tokenizer.spm_byte_path,
            asdict(args.data.tokenizer.byte_converter_config),
            args.data.tokenizer.separate_embedding,
        )
        validate_train_args(args)#, _tokenizer.n_words)
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            os.makedirs(args.log_dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.log_dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        if args.override_saved_optim_and_data_config:
            if args.continual_training_path:
                logger.info(f"Loading continual training from {args.continual_training_path}")
                checkpoint.load(model, optimizer, train_state, world_mesh, Path(args.continual_training_path))
            else:
                logger.info(f"Loading latest checkpoint from {args.checkpoint.path}")
                checkpoint.load(model, optimizer, train_state, world_mesh)

            #### if we specify different sources from saved checkpoint, we need to reset the data_loader_state
            cur_multi_state = data_loader_state["it_state"]["it_state"]["it_state"]
            saved_multi_state = train_state.data_loader_state["it_state"]["it_state"]["it_state"]
            if set(cur_multi_state["sources"].items()) != set(saved_multi_state["sources"].items()):
                logger.info(f"[DATA MIXTURE CHANGE] data source mixtures changed, updating data_loader_state...")

                for cur_source, cur_weight in cur_multi_state["sources"].items():
                    # 1. Handle new sources - add them to saved state
                    if cur_source not in saved_multi_state["sources"]:
                        logger.info(f"[DATA MIXTURE CHANGE] Adding new data source {cur_source} with weight {cur_weight}")
                        saved_multi_state["sources"][cur_source] = cur_weight
                        # Copy the source state from current to saved
                        saved_multi_state["source_to_state"][cur_source] = deepcopy(cur_multi_state["source_to_state"][cur_source])
                
                    # 2. Handle weight changes for existing sources
                    elif cur_source in saved_multi_state["sources"]:
                        saved_weight = saved_multi_state["sources"][cur_source]
                        if cur_weight != saved_weight:
                            logger.info(f"[DATA MIXTURE CHANGE] Data source {cur_source} weight changed from {saved_weight} to {cur_weight}")
                            saved_multi_state["sources"][cur_source] = cur_weight
                
                # 3. Handle removed sources - remove them from saved state
                sources_to_remove = []
                for saved_source in saved_multi_state["sources"].keys():
                    if saved_source not in cur_multi_state["sources"]:
                        logger.info(f"[DATA MIXTURE CHANGE] Removing data source {saved_source}")
                        sources_to_remove.append(saved_source)
                
                for source_to_remove in sources_to_remove:
                    del saved_multi_state["sources"][source_to_remove]
                    if source_to_remove in saved_multi_state["source_to_state"]:
                        del saved_multi_state["source_to_state"][source_to_remove]
            else:
                logger.info(f"[DATA MIXTURE] Current data source mixtures are the same as the saved checkpoint, no need to update data_loader_state")
        else:
            logger.info(f"Loading latest checkpoint from {args.checkpoint.path}")
            checkpoint.load(model, optimizer, train_state, world_mesh)

        checkpoint.load(model, optimizer, train_state, world_mesh)
        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.log_dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.log_dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.log_dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_opencoder_dataloader_from_args(
                args.data,
                state=train_state.data_loader_state,
                apply_doc_boundary_mask=args.apply_doc_boundary_mask,
                eos_id=_tokenizer.eos_id,
            )
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.log_dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()

        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            (batch, position_ids, cuseq_lens, max_seqlen), train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(
                batch,
                dtype=torch.long,
            )
            if args.apply_doc_boundary_mask:
                position_ids = torch.tensor(
                    position_ids,
                    dtype=torch.long,
                ).cuda()
                cuseq_lens = torch.tensor(
                    cuseq_lens,
                    dtype=torch.int32,
                ).cuda()
                attn_impl = "flash_attn2"
                max_seqlen = int(max_seqlen)
            else:
                position_ids = None
                cuseq_lens = None
                attn_impl = "sdpa"
                max_seqlen = None

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            loss = model(
                input_ids, 
                labels, 
                tok_idx=position_ids,
                attn_impl=attn_impl, 
                cuseq_lens=cuseq_lens,
                max_seqlen=max_seqlen
            )

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config
    if hasattr(cli_args, "data") and hasattr(cli_args.data, "sources"):
        sources_cli = cli_args.data.sources
        del cli_args.data.sources
    else:
        sources_cli = None

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    if sources_cli:
        cfg.data.sources = dict(sources_cli)
    train(cfg)


if __name__ == "__main__":
    main()
