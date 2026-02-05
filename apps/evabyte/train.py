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
from torch import nn
import torch.distributed
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dump_config, flatten_dict
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
    clean_env,
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
from lingua.optim import OptimArgs, build_lr_fn, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from apps.evabyte.evabyte import (
    EvaByteModelArgs,
    EvaByte,
    ChunkedFusedLinearCrossEntropy,
    ChunkedFusedLinearwithRawMultibyteCrossEntropy,
    get_num_flop_per_token,
    get_no_recompute_ops,
    build_fsdp_grouping_plan,
)
from apps.evabyte.attn_mask_utils import (
    prepare_token_types_position_ids_numpy,
    prepare_multibyte_loss_weight_numpy,
)
from lingua.probe import AutoProbeD
from multiprocessing import Lock

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

    # disable weight decay in embedding weights
    disable_wd_emb: bool = False

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    # Enable doc masking or not
    apply_doc_boundary_mask: bool = False
    apply_fused_linear_chunked_ce_loss: bool = False
    apply_multibyte_loss_mask: bool = False
    disable_cross_byte_prediction: bool = False
    weighting_compressed_prediction: bool = False
    compressed_loss_weight: float = 0.333333
    
    override_saved_optim_and_data_config: bool = False
    continual_training_path: Optional[str] = None
    
    # Compression rate scheduling parameters
    enable_compression_rate_schedule: bool = False
    compression_warmup_steps: int = 1000
    compression_steady_steps: int = 5000
    compression_decay_steps: int = 2000
    compression_initial_rate: float = 0.0
    compression_peak_rate: float = 0.5
    compression_final_rate: float = 0.1
    compression_initial_mode: str = "parallel_raw_compressed"
    compression_steady_mode: str = "sentinel"
    compression_final_mode: str = "sentinel"

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: EvaByteModelArgs = field(default_factory=EvaByteModelArgs)
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

def evabyte_build_optimizer(model: nn.Module, args: OptimArgs, n_steps: int):
    logger.info("Starting build of EvaByte optimizer...")

    decay = set()
    no_decay = set()
    for pn, p in model.named_parameters():
        if pn == "embed_tokens.weight":
            no_decay.add(pn)
        else:
            decay.add(pn)
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    logger.info("Not applying weight_decay on following params: {}".format(no_decay))
    assert len(no_decay) > 0, "No params found in no_decay!"
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    param_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        fused=True,  # Faster optim.step but can throw errors
    )

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_fn
    )

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler

def compression_sampling_schedule(
    batch_count: int,
    warmup_steps: int = 1000,
    steady_steps: int = 5000, 
    decay_steps: int = 2000,
    initial_rate: float = 0.0,
    peak_rate: float = 0.5,
    final_rate: float = 0.1,
    warmup_mode: str = "sentinel",
    peak_mode: str = "sentinel",
    final_mode: str = "sentinel"
) -> float:
    """
    Compression sampling rate schedule: warmup -> steady -> decay
    
    Args:
        batch_count: Current batch number
        warmup_steps: Number of batches for warmup phase
        steady_steps: Number of batches for steady phase  
        decay_steps: Number of batches for decay phase
        initial_rate: Starting compression rate
        peak_rate: Peak compression rate during steady phase
        final_rate: Final compression rate after decay
    
    Returns:
        compression_sampling_rate for current batch
    """
    if batch_count < warmup_steps:
        # Linear warmup from initial_rate to peak_rate
        progress = batch_count / warmup_steps
        return (
            initial_rate + (peak_rate - initial_rate) * progress,
            warmup_mode
        )
    elif batch_count < warmup_steps + steady_steps:
        # Steady phase at peak_rate
        return (
            peak_rate,
            peak_mode
        )
    elif batch_count < warmup_steps + steady_steps + decay_steps:
        # Linear decay from peak_rate to final_rate
        decay_progress = (batch_count - warmup_steps - steady_steps) / decay_steps
        return (
            peak_rate - (peak_rate - final_rate) * decay_progress,
            final_mode
        )
    else:
        # After decay, maintain final_rate
        return (
            final_rate,
            final_mode
        )

class CompressionSamplingRateState:
    def __init__(
        self,
        warmup_steps: int = 1000,
        steady_steps: int = 5000,
        decay_steps: int = 2000,
        initial_rate: float = 0.0,
        peak_rate: float = 0.5,
        final_rate: float = 0.1,
        warmup_mode: str = "sentinel",
        peak_mode: str = "sentinel",
        final_mode: str = "sentinel",
        batch_count: int = 0
    ):
        self.lock = Lock()
        self.warmup_steps = warmup_steps
        self.steady_steps = steady_steps
        self.decay_steps = decay_steps
        self.initial_rate = initial_rate
        self.peak_rate = peak_rate
        self.final_rate = final_rate
        self.warmup_mode = warmup_mode
        self.peak_mode = peak_mode
        self.final_mode = final_mode
        self.current_batch_count = batch_count
        self.current_compression_sampling_rate = 0
        self.current_compression_sampling_mode = "sentinel"

    def update(self):
        """Called to atomically recompute the compression sampling rate."""
        with self.lock:
            self.current_batch_count += 1

    def get_compression_sampling_rate_mode(self):
        """Called by the reader to get the compression sampling rate."""
        with self.lock:
            current_tuple = compression_sampling_schedule(
                batch_count=self.current_batch_count,
                warmup_steps=self.warmup_steps,
                steady_steps=self.steady_steps,
                decay_steps=self.decay_steps,
                initial_rate=self.initial_rate,
                peak_rate=self.peak_rate,
                final_rate=self.final_rate,
                warmup_mode=self.warmup_mode,
                peak_mode=self.peak_mode,
                final_mode=self.final_mode
            )
            self.current_compression_sampling_rate, self.current_compression_sampling_mode = current_tuple
            return self.current_compression_sampling_rate, self.current_compression_sampling_mode

    def get_batch_count(self):
        with self.lock:
            return self.current_batch_count

def setup_sources_with_global_state(multi_state, compression_sampling_rate_state: CompressionSamplingRateState):
    path_to_iter = dict()
    for source in multi_state["sources"]:
        jsonl_state = multi_state["source_to_state"][source]
        path_to_iter[source] = loop_on_jsonl(
            jsonl_state["file_path"],
            jsonl_state["position"],
            jsonl_state["block_size"],
            jsonl_state["offset"],
            jsonl_state["current_iter"],
            jsonl_state["compression_rng_state"],
            jsonl_state["compression_sampling_rate"],
            jsonl_state["raw_compression_mix_option"],
            compression_sampling_rate_state,
        )

    return path_to_iter

def eva_doc_mask_iter_with_batch_tracking(
    data_loader,
    chunk_size: int,
    window_size: int,
    eos_id: int,
    apply_doc_boundary_mask: bool,
    compression_sampling_rate_state: CompressionSamplingRateState,
    apply_multibyte_loss_mask: bool,
    loss_mask_args: dict,
):
    """
    Enhanced eva_doc_mask_iter that tracks batch count and updates compression sampling rate
    """
    for batch, state in data_loader:
        # Calculate new compression sampling rate based on current batch count
        compression_sampling_rate_state.update()
        new_compression_rate, _ = compression_sampling_rate_state.get_compression_sampling_rate_mode()
        batch_count = compression_sampling_rate_state.get_batch_count()
        # Process the batch as before
        if apply_doc_boundary_mask:
            token_types, chunked_token_type_ids, intra_chunk_mask, position_ids = prepare_token_types_position_ids_numpy(
                batch[:, :, 0],
                chunk_size,
                window_size,
                eos_token_id=eos_id,
            )
            # token_types, chunked_token_type_ids, intra_chunk_mask, position_ids = prepare_token_types_position_ids(
            #     torch.tensor(batch[:, :, 0], dtype=torch.long),
            #     chunk_size,
            #     window_size,
            #     eos_token_id=eos_id,
            # )
        else:
            token_types = None
            chunked_token_type_ids = None
            intra_chunk_mask = None
            position_ids = None
        if apply_multibyte_loss_mask:
            labels = batch[:, :, 1:]
            input_ids = batch[:, :, 0]
            loss_mask = prepare_multibyte_loss_weight_numpy(
                input_ids,
                labels,
                loss_mask_args["num_pred_heads"],
                loss_mask_args["vocab_size"],
                loss_mask_args["raw_byte_offset"],
                loss_mask_args["bos_id"],
                loss_mask_args["raw_sentinel_id_start"],
                loss_mask_args["compressed_sentinel_id_start"],
                loss_mask_args["disable_cross_byte_prediction"],
                loss_mask_args["weighting_compressed_prediction"],
                loss_mask_args["compressed_loss_weight"],
            )
        else:
            loss_mask = None
        # Update state with current batch count and compression rate
        enhanced_state = {**state, "batch_count": batch_count, "compression_sampling_rate": new_compression_rate}
        yield (batch, token_types, chunked_token_type_ids, intra_chunk_mask, position_ids, loss_mask), enhanced_state


def eva_doc_mask_iter(
    data_loader,
    chunk_size: int,
    window_size: int,
    eos_id: int,
    apply_doc_boundary_mask: bool,
    apply_multibyte_loss_mask: bool,
    loss_mask_args: dict,
):
    for batch, state in data_loader:
        if apply_doc_boundary_mask:
            token_types, chunked_token_type_ids, intra_chunk_mask, position_ids = prepare_token_types_position_ids_numpy(
                batch[:, :, 0],
                chunk_size,
                window_size,
                eos_token_id=eos_id,
            )
            # token_types, chunked_token_type_ids, intra_chunk_mask, position_ids = prepare_token_types_position_ids(
            #     torch.tensor(batch[:, :, 0], dtype=torch.long),
            #     chunk_size,
            #     window_size,
            #     eos_token_id=eos_id,
            # )
        else:
            token_types = None
            chunked_token_type_ids = None
            intra_chunk_mask = None
            position_ids = None
        
        if apply_multibyte_loss_mask:
            labels = batch[:, :, 1:]
            input_ids = batch[:, :, 0]
            loss_mask = prepare_multibyte_loss_weight_numpy(
                input_ids,
                labels,
                loss_mask_args["num_pred_heads"],
                loss_mask_args["vocab_size"],
                loss_mask_args["raw_byte_offset"],
                loss_mask_args["bos_id"],
                loss_mask_args["raw_sentinel_id_start"],
                loss_mask_args["compressed_sentinel_id_start"],
                loss_mask_args["disable_cross_byte_prediction"],
                loss_mask_args["weighting_compressed_prediction"],
            )
        else:
            loss_mask = None
        yield (batch, token_types, chunked_token_type_ids, intra_chunk_mask, position_ids, loss_mask), state

@contextlib.contextmanager
def build_eva_dataloader(
    chunk_size: int,
    window_size: int,
    eos_id: int,
    apply_doc_boundary_mask: bool,
    apply_multibyte_loss_mask: bool,
    loss_mask_args: dict,
    state: PrefetchState,
    enable_compression_rate_schedule: bool = False,
    warmup_steps: int = 1000,
    steady_steps: int = 5000, 
    decay_steps: int = 2000,
    initial_rate: float = 0.0,
    peak_rate: float = 0.5,
    final_rate: float = 0.1,
    warmup_mode: str = "sentinel",
    peak_mode: str = "sentinel",
    final_mode: str = "sentinel",
):
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]

    if enable_compression_rate_schedule:
        compression_sampling_rate_state = CompressionSamplingRateState(
            warmup_steps=warmup_steps,
            steady_steps=steady_steps,
            decay_steps=decay_steps,
            initial_rate=initial_rate,
            peak_rate=peak_rate,
            final_rate=final_rate,
            warmup_mode=warmup_mode,
            peak_mode=peak_mode,
            final_mode=final_mode,
            batch_count=state.get("batch_count", 0),
        )
    else:
        compression_sampling_rate_state = None
    path_to_iter = setup_sources_with_global_state(multi_state, compression_sampling_rate_state)
    data_it = choose_source(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
    )
    # data_it = tokenize(
    #     data_it,
    #     tokenizer_state["add_bos"],
    #     tokenizer_state["add_eos"],
    #     tokenizer_state["name"],
    #     tokenizer_state["path"],
    #     tokenizer_state["spm_byte_path"],
    #     tokenizer_state["byte_conversion_args"],
    #     tokenizer_state["compression_alg_config"],
    #     tokenizer_state["separate_embedding"],
    # )
    data_it = tokenize(
        data_it,
        tokenizer_state["add_bos"],
        tokenizer_state["add_eos"],
        tokenizer_state["name"],
        tokenizer_state["path"],
        tokenizer_state.get("spm_byte_path", None),
        tokenizer_state.get("byte_conversion_args", None),
        tokenizer_state["compression_alg_config"],
        tokenizer_state.get("separate_embedding", False),
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

    if enable_compression_rate_schedule:
        data_it = eva_doc_mask_iter_with_batch_tracking(
            data_it, 
            chunk_size,
            window_size,
            eos_id,
            apply_doc_boundary_mask,
            compression_sampling_rate_state,
            apply_multibyte_loss_mask,
            loss_mask_args,
        )
    else:
        data_it = eva_doc_mask_iter(
            data_it, 
            chunk_size,
            window_size,
            eos_id,
            apply_doc_boundary_mask,
            apply_multibyte_loss_mask,
            loss_mask_args,
        )
    yield data_it
    for it in path_to_iter.values():
        it.close()
    data_it.close()

def build_eva_dataloader_from_args(
    args: DataArgs,
    chunk_size: int,
    window_size: int,
    eos_id: int,
    apply_doc_boundary_mask: bool,
    apply_multibyte_loss_mask: bool,
    loss_mask_args: dict,
    state = None,
    enable_compression_rate_schedule: bool = False,
    warmup_steps: int = 1000,
    steady_steps: int = 5000,
    decay_steps: int = 2000,
    initial_rate: float = 0.0,
    peak_rate: float = 0.5,
    final_rate: float = 0.1,
    warmup_mode: str = "sentinel",
    peak_mode: str = "sentinel",
    final_mode: str = "sentinel",
):
    data_builder = partial(
        build_eva_dataloader, 
        chunk_size, 
        window_size, 
        eos_id, 
        apply_doc_boundary_mask, 
        apply_multibyte_loss_mask,
        loss_mask_args,
        state,
        enable_compression_rate_schedule,
        warmup_steps,
        steady_steps,
        decay_steps,
        initial_rate,
        peak_rate,
        final_rate,
        warmup_mode,
        peak_mode,
        final_mode,
    )
    if args.load_async:
        return async_iterator(args.prefetch_size, data_builder)
    else:
        return data_builder()

# def validate_train_args(args: TrainArgs, n_words: int):
#     if args.model.vocab_size < 0:
#         logger.info(f"Setting model vocab size to {n_words}")
#         args.model.vocab_size = n_words
#     assert (
#         args.model.vocab_size == n_words
#     ), "Vocab size should be the same as tokenizer.n_words"

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

    args.model.max_position_embeddings = args.data.seq_len
    args.model.seed = args.seed

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

    if args.model.apply_raw_multibyte_lm_head:
        assert args.model.num_raw_pred_heads + args.model.num_pred_heads == args.data.n_views - 1
        assert not args.apply_multibyte_loss_mask
        assert not args.disable_cross_byte_prediction
        assert not args.weighting_compressed_prediction
    else:
        assert args.model.num_pred_heads == args.data.n_views - 1


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
        logger.info(f"Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = EvaByte(args.model)
        logger.info(f"Model is built !")

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
            model.rotary_emb.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size
        logger.info(model)
        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        if args.disable_wd_emb:
            optimizer, scheduler = evabyte_build_optimizer(model, args.optim, args.steps)
        else:
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
                            assert saved_multi_state["source_to_state"][cur_source]["block_size"] == cur_multi_state["source_to_state"][cur_source]["block_size"], (
                                "Mismatched block size for data source {cur_source}; currently we only support same world size for continual training"
                            )
                            # also copy specified attributes
                            saved_multi_state["source_to_state"][cur_source]["compression_sampling_rate"] = cur_multi_state["source_to_state"][cur_source]["compression_sampling_rate"]
                            saved_multi_state["source_to_state"][cur_source]["raw_compression_mix_option"] = cur_multi_state["source_to_state"][cur_source]["raw_compression_mix_option"]
                
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

            #### also copy maintain some specified arguments
            # PrefetchState
            train_state.data_loader_state["batch_size"] = data_loader_state["batch_size"]
            train_state.data_loader_state["prefetch_size"] = data_loader_state["prefetch_size"]

            # PackTokensState
            train_state.data_loader_state["it_state"]["output_seq_len"] = data_loader_state["it_state"]["output_seq_len"]
            train_state.data_loader_state["it_state"]["n_views"] = data_loader_state["it_state"]["n_views"]

            # TokenizerState
            train_state.data_loader_state["it_state"]["it_state"]["add_bos"] = data_loader_state["it_state"]["it_state"]["add_bos"]
            train_state.data_loader_state["it_state"]["it_state"]["add_eos"] = data_loader_state["it_state"]["it_state"]["add_eos"]
            train_state.data_loader_state["it_state"]["it_state"]["name"] = data_loader_state["it_state"]["it_state"]["name"]
            train_state.data_loader_state["it_state"]["it_state"]["path"] = data_loader_state["it_state"]["it_state"]["path"]
            train_state.data_loader_state["it_state"]["it_state"]["spm_byte_path"] = data_loader_state["it_state"]["it_state"]["spm_byte_path"]
            train_state.data_loader_state["it_state"]["it_state"]["byte_conversion_args"] = data_loader_state["it_state"]["it_state"]["byte_conversion_args"]
            train_state.data_loader_state["it_state"]["it_state"]["compression_alg_config"] = data_loader_state["it_state"]["it_state"]["compression_alg_config"]
            train_state.data_loader_state["it_state"]["it_state"]["separate_embedding"] = data_loader_state["it_state"]["it_state"]["separate_embedding"]
        else:
            logger.info(f"Loading latest checkpoint from {args.checkpoint.path}")
            checkpoint.load(model, optimizer, train_state, world_mesh)

        logger.info(f"[DATA MIXTURE] data_loader_state: {train_state.data_loader_state}")
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

        if args.apply_multibyte_loss_mask:
            multibyte_loss_mask_args = {
                "num_pred_heads": args.model.num_pred_heads,
                "vocab_size": args.model.vocab_size,
                "raw_byte_offset": _tokenizer.offset + 256,
                "bos_id": _tokenizer.bos_id,
                "raw_sentinel_id_start": _tokenizer.raw_sentinel_id_start,
                "compressed_sentinel_id_start": _tokenizer.compressed_sentinel_id_start,
                "disable_cross_byte_prediction": args.disable_cross_byte_prediction,
                "weighting_compressed_prediction": args.weighting_compressed_prediction,
                "compressed_loss_weight": args.compressed_loss_weight,
            }
        else:
            multibyte_loss_mask_args = None
        data_loader = context_stack.enter_context(
            build_eva_dataloader_from_args(
                args.data,
                chunk_size=args.model.chunk_size,
                window_size=args.model.window_size,
                eos_id=_tokenizer.eos_id,
                apply_doc_boundary_mask=args.apply_doc_boundary_mask,
                apply_multibyte_loss_mask=args.apply_multibyte_loss_mask,
                loss_mask_args=multibyte_loss_mask_args,
                state=train_state.data_loader_state,
                enable_compression_rate_schedule=args.enable_compression_rate_schedule,
                warmup_steps=args.compression_warmup_steps,
                steady_steps=args.compression_steady_steps,
                decay_steps=args.compression_decay_steps,
                initial_rate=args.compression_initial_rate,
                peak_rate=args.compression_peak_rate,
                final_rate=args.compression_final_rate,
                warmup_mode=args.compression_initial_mode,
                peak_mode=args.compression_steady_mode,
                final_mode=args.compression_final_mode,
            )
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.log_dump_dir, model, args.profiling)
        )

        if args.apply_fused_linear_chunked_ce_loss:
            if args.model.apply_raw_multibyte_lm_head:
                assert not args.apply_multibyte_loss_mask
                # assert args.model.num_pred_heads > 1
                assert args.model.num_raw_pred_heads >= 1
                chunked_ce_loss_fn = ChunkedFusedLinearwithRawMultibyteCrossEntropy(
                    model.lm_head,
                    model.raw_multibyte_lm_head,
                    args.model.num_pred_heads,
                    args.model.num_raw_pred_heads,
                    args.model.vocab_size,
                    args.model.raw_vocab_size,
                    args.model.hidden_size
                )
                chunked_ce_loss_fn.apply_compile_strategy()
            else:
                assert args.apply_multibyte_loss_mask
                assert args.model.num_pred_heads > 1
                chunked_ce_loss_fn = ChunkedFusedLinearCrossEntropy(
                    model.lm_head,
                    args.model.num_pred_heads,
                    args.model.vocab_size,
                    args.model.hidden_size
                )
                chunked_ce_loss_fn.apply_compile_strategy()
        else:
            chunked_ce_loss_fn = None

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
            (batch, token_types, chunked_token_type_ids, intra_chunk_mask, position_ids, loss_mask), train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(
                batch,
                dtype=torch.long,
            )
            if args.apply_doc_boundary_mask:
                # token_types = torch.tensor(token_types, dtype=torch.long)
                # chunked_token_type_ids = torch.tensor(chunked_token_type_ids, dtype=torch.long)
                # intra_chunk_mask = torch.tensor(intra_chunk_mask, dtype=torch.bool)
                # attention_mask = (token_types.cuda(), chunked_token_type_ids.cuda(), intra_chunk_mask.cuda())
                # position_ids = torch.tensor(position_ids, dtype=torch.long).cuda()
                token_types = torch.from_numpy(token_types).cuda()
                chunked_token_type_ids = torch.from_numpy(chunked_token_type_ids).cuda() 
                intra_chunk_mask = torch.from_numpy(intra_chunk_mask).cuda()
                position_ids = torch.tensor(position_ids, dtype=torch.long).cuda()

                # attention_mask = (token_types.cuda(), chunked_token_type_ids.cuda(), intra_chunk_mask.cuda())
                # position_ids = position_ids.cuda()
            else:
                token_types = None
                chunked_token_type_ids = None
                intra_chunk_mask = None
                position_ids = None

            if args.apply_multibyte_loss_mask:
                loss_mask = torch.from_numpy(loss_mask).cuda()
            else:
                loss_mask = None

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1:].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen, _ = labels.shape

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    _input_ids = input_ids[:probe_bsz, :probe_seq]
                    if args.apply_doc_boundary_mask:
                        # NOTE: disable the boundary mask for probe
                        _token_types = None
                        _chunked_token_type_ids = None
                        _intra_chunk_mask = None
                        _position_ids = None
                    else:
                        _token_types = None
                        _chunked_token_type_ids = None
                        _intra_chunk_mask = None
                        _position_ids = None
                    probe_loss = model(
                        input_ids=input_ids[:probe_bsz, :probe_seq],
                        token_types=_token_types,
                        chunked_token_type_ids=_chunked_token_type_ids,
                        intra_chunk_mask=_intra_chunk_mask,
                        position_ids=_position_ids,
                        labels=labels[:probe_bsz, :probe_seq],
                    )
                    (probe_loss / args.model.num_pred_heads).sum().backward()
                    # We zero grads to cancel this fake step
                    optimizer.zero_grad()

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            if args.apply_fused_linear_chunked_ce_loss:
                hidden_states = model(
                    input_ids=input_ids,
                    token_types=token_types,
                    chunked_token_type_ids=chunked_token_type_ids,
                    intra_chunk_mask=intra_chunk_mask,
                    position_ids=position_ids,
                    labels=None,
                    loss_mask=None,
                    skip_lm_head=True
                )
                losses = chunked_ce_loss_fn(hidden_states, labels, loss_mask)
            else:
                losses = model(
                    input_ids=input_ids,
                    token_types=token_types,
                    chunked_token_type_ids=chunked_token_type_ids,
                    intra_chunk_mask=intra_chunk_mask,
                    position_ids=position_ids,
                    labels=labels,
                    loss_mask=loss_mask,
                )

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            if args.model.apply_raw_multibyte_lm_head:
                # losses are pre-scaled so we only need to divide by grad_acc_steps
                loss = losses.sum() / args.grad_acc_steps
            elif args.apply_multibyte_loss_mask:
                loss = losses.sum() / args.grad_acc_steps
            else:
                # We scale loss with grad_acc_steps so the gradient is the same
                # regardless of grad_acc_steps
                loss = (losses / args.model.num_pred_heads).sum() / args.grad_acc_steps
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
                        model_param_count - args.model.vocab_size * args.model.hidden_size,
                        args.model.num_hidden_layers,
                        args.model.hidden_size,
                        args.data.seq_len,
                    )
                    * wps
                )
                if train_state.data_loader_state.get("compression_sampling_rate", None):
                    compression_rate = train_state.data_loader_state["compression_sampling_rate"]
                else:
                    compression_rate = args.data.compression_sampling_rate
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
                        "compression_rate": compression_rate,
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                if losses.ndim == 1:
                    for head_idx, l in enumerate(losses):
                        to_sync[f"loss/head_{head_idx}"] = l.item()

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
                    f"  compression_rate: {compression_rate:.2e}"
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
        mode: LMMTPArgs

    @dataclass
    class LMMTPArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMMTPArgs
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
