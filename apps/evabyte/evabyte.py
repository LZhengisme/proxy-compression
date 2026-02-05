from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from apps.evabyte.core_evabyte import (
    BaseEvaByteArgs,
    EvaByteDecoderLayer,
    EvaByteRotaryEmbedding,
    EvaByteRMSNorm,
    InitArgs,
    prepare_eva_training_mask,
)
from apps.evabyte.attn_mask_utils import (
    calculate_3d_attention_mask_gpu,
)
from torch.nn import CrossEntropyLoss

class ChunkedFusedLinearCrossEntropy(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/loss/ce_chunked_output_loss.py
    """
    def __init__(self, lm_head, num_pred_heads, vocab_size, hidden_size, chunk_size: int = 1024):
        super().__init__()
        self.lm_head = lm_head
        self.num_pred_heads = num_pred_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size

    def compute_cross_entropy(
        self, 
        hidden_states: torch.Tensor, 
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.
        """

        # chunk_logits = torch.mm(hidden_states, self.lm_head.weight.t())
        logits = self.lm_head(hidden_states)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size).float(), 
            labels.view(-1),
            reduction="none"
        )
        head_losses = (loss.view(-1, self.num_pred_heads) * loss_mask).sum(dim=0)
        return head_losses

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the fkl_loss function."""
        self.compute_cross_entropy = torch.compile(
            self.compute_cross_entropy, *args, **kwargs
        )
        return self

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # hidden_states [B, S, D] -> [B x S, D]
        flat_hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_chunks = flat_hidden_states.shape[0] // self.chunk_size
        chunked_hidden_states = torch.chunk(flat_hidden_states, num_chunks, dim=0)
        # labels [B, S, H] -> [B x S, H]
        flat_labels = labels.reshape(-1, self.num_pred_heads)
        chunked_labels = torch.chunk(flat_labels, num_chunks, dim=0)

        # loss_masks [B x S, D] -> [B x S, D]
        flat_loss_mask = loss_mask.reshape(-1, self.num_pred_heads)
        chunked_loss_mask = torch.chunk(flat_loss_mask, num_chunks, dim=0)
        head_losses = torch.zeros(self.num_pred_heads, dtype=torch.float32, device=hidden_states.device)
        for chunk_hidden_states, chunk_labels, chunk_loss_mask in zip(
            chunked_hidden_states, chunked_labels, chunked_loss_mask
        ):
            head_losses = head_losses + self.compute_cross_entropy(
                chunk_hidden_states,
                chunk_labels,
                chunk_loss_mask
            )

        num_unmasked_tokens = loss_mask.float().sum()
        head_losses = head_losses / num_unmasked_tokens
        return head_losses

class ChunkedFusedLinearwithRawMultibyteCrossEntropy(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/loss/ce_chunked_output_loss.py
    """
    def __init__(
        self, 
        lm_head, 
        raw_multibyte_lm_head,
        num_pred_heads, 
        num_raw_pred_heads,
        vocab_size, 
        raw_vocab_size,
        hidden_size, 
        chunk_size: int = 1024
    ):
        super().__init__()
        self.lm_head = lm_head
        self.raw_multibyte_lm_head = raw_multibyte_lm_head
        self.num_pred_heads = num_pred_heads
        self.num_raw_pred_heads = num_raw_pred_heads
        self.vocab_size = vocab_size
        self.raw_vocab_size = raw_vocab_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size

    def compute_cross_entropy(
        self, 
        hidden_states: torch.Tensor, 
        labels: torch.Tensor,
        flat_pred_head_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.
        """

        # chunk_logits = torch.mm(hidden_states, self.lm_head.weight.t())
        logits = self.lm_head(hidden_states)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size).float(), 
            labels.reshape(-1),
            reduction="none"
        )
        chunk_loss = (loss.reshape(-1, self.num_pred_heads) * flat_pred_head_weights).sum()
        return chunk_loss

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the fkl_loss function."""
        self.compute_cross_entropy = torch.compile(
            self.compute_cross_entropy, *args, **kwargs
        )
        return self

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
        total_num_pred_heads = self.num_pred_heads + self.num_raw_pred_heads

        # [B, S, n_views] -> [B x S, n_views]
        labels_flat = labels.reshape(-1, labels.shape[-1])

        # [B, S, D] -> [B x S, D]
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)

        # [B x S, n_views] -> [B x S]
        all_labels_are_raw = torch.all(labels_flat < self.raw_vocab_size, dim=-1)

        # [B x S, D] -> [N_raw, D]
        raw_hidden_states = hidden_states_flat[all_labels_are_raw]

        # [B x S] -> [B x S, 1]
        flat_pred_head_weights = torch.where(
            all_labels_are_raw, 
            1. / total_num_pred_heads, 
            1. / self.num_pred_heads
        ).float().unsqueeze(-1)

        # [N_raw, D] -> [N_raw, num_raw_pred_heads, raw_vocab_size]
        raw_multibyte_logits = self.raw_multibyte_lm_head(raw_hidden_states).reshape(-1, self.num_raw_pred_heads, self.raw_vocab_size)
        # NOTE: we skip bytes covered by lm_head for raw multibyte predictions
        # [B x S, n_views] -> [N_raw, num_raw_pred_heads]
        raw_multibyte_labels = labels_flat[all_labels_are_raw, self.num_pred_heads : self.num_pred_heads + self.num_raw_pred_heads]
        # [N_raw, num_raw_pred_heads, raw_vocab_size], [N_raw, num_raw_pred_heads] -> [N_raw x num_raw_pred_heads]
        raw_multibyte_ce_loss = F.cross_entropy(
            raw_multibyte_logits.float().view(-1, self.raw_vocab_size), 
            raw_multibyte_labels.reshape(-1), 
            reduction="none"
        ) / total_num_pred_heads
        _total_loss = raw_multibyte_ce_loss.sum()

        num_chunks = hidden_states_flat.shape[0] // self.chunk_size
        chunked_hidden_states = torch.chunk(hidden_states_flat, num_chunks, dim=0)
        chunked_labels = torch.chunk(labels_flat[:, : self.num_pred_heads], num_chunks, dim=0)
        chunked_head_weights = torch.chunk(flat_pred_head_weights, num_chunks, dim=0)

        for chunk_hidden_states, chunk_labels, chunk_head_weights in zip(
            chunked_hidden_states, chunked_labels, chunked_head_weights
        ):
            _total_loss = _total_loss + self.compute_cross_entropy(
                chunk_hidden_states,
                chunk_labels,
                chunk_head_weights
            )

        num_tokens = labels.shape[0] * labels.shape[1]
        total_loss = _total_loss / num_tokens

        return total_loss

def prepare_doc_mask_position_ids(
    input_ids: torch.LongTensor,
    chunk_size: int,
    window_size: int,
    eos_token_id: int,
    pad_token_id: Optional[int] = None,
):
    attention_mask = prepare_eva_training_mask(
        input_ids,
        True,
        chunk_size,
        window_size,
        EOS_TOKEN_TYPE_ID=eos_token_id,
        PAD_TOKEN_TYPE_ID=pad_token_id,
    )

    bs, seq_len = input_ids.shape

    position_ids = []

    for b in range(bs):
        position_id = torch.arange(0, seq_len, dtype=torch.long)
        # Find indecies where EOD token is.
        eos_ind = position_id[input_ids[b] == eos_token_id]

        # Loop through EOD indecies:
        prev_index = 0
        for j in range(eos_ind.shape[0]):
            i = eos_ind[j]
            # Reset positions.
            position_id[(i + 1):] -= (i + 1 - prev_index)
            prev_index = i + 1
        
        position_ids.append(position_id)
    position_ids = torch.stack(position_ids, dim=0)
    return attention_mask, position_ids

@dataclass
class EvaByteModelArgs(BaseEvaByteArgs):
    seed: int = 42

class EvaByte(nn.Module):
    def __init__(self, config: EvaByteModelArgs):
        super().__init__()
        self.config = config
        self.init_args = config.init_args

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings

        self.rotary_emb = EvaByteRotaryEmbedding(self.head_dim,
                                                max_position_embeddings=self.max_position_embeddings,
                                                base=config.rope_theta)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([EvaByteDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = EvaByteRMSNorm(config)

        # define multibyte prediction heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size * config.num_pred_heads, bias=False)
        if config.apply_raw_multibyte_lm_head:
            # we separately allocate a lm_head for multibyte raw predictions to reduce memory usage
            # this head is only used for raw predictions beyond the bytes covered by lm_head
            self.raw_multibyte_lm_head = nn.Linear(config.hidden_size, config.raw_vocab_size * config.num_raw_pred_heads, bias=False)
            

    def forward(
            self,
            input_ids: torch.LongTensor,
            token_types: Optional[torch.Tensor] = None,
            chunked_token_type_ids: Optional[torch.Tensor] = None,
            intra_chunk_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            loss_mask: Optional[torch.Tensor] = None,
            skip_lm_head: Optional[bool] = False
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # assert self.training
        assert seq_len % self.config.window_size == 0
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        assert len(cos.shape) == 2, f"cos should be of shape (max_seq_len, head_dim), got {cos.shape} instead"
        assert sin.shape == cos.shape, f"sin should be of shape (max_seq_len, head_dim), got {sin.shape} instead"

        if position_ids is None:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            assert len(position_ids.shape) == 2, f"position_ids should be of 2D, got {position_ids.shape} instead"
            cos = cos[position_ids, :]
            sin = sin[position_ids, :]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # for training, we need to pass in the attention mask
        # usually calculated by _prepare_training_attn_mask()
        if token_types is not None and chunked_token_type_ids is not None and intra_chunk_mask is not None:
            window_mask, chunk_mask, intra_chunk_mask = calculate_3d_attention_mask_gpu(
                token_types,
                chunked_token_type_ids,
                intra_chunk_mask,
                self.config.chunk_size,
                self.config.window_size,
            )
        else:
            window_mask = None
            chunk_mask = None
            intra_chunk_mask = None

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                window_mask=window_mask,
                chunk_mask=chunk_mask, 
                intra_chunk_mask=intra_chunk_mask,
                cos=cos,
                sin=sin,
            )
        hidden_states = self.norm(hidden_states)

        if skip_lm_head:
            return hidden_states
        if self.config.apply_raw_multibyte_lm_head:
            ###########

            logits = self.lm_head(hidden_states)
            shift_logits = logits.view(logits.shape[0], logits.shape[1], self.config.num_pred_heads, self.config.vocab_size)
            if labels is None:
                return shift_logits
            lm_head_labels = labels[:, :, : self.config.num_pred_heads]
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size).float(), lm_head_labels.reshape(-1), reduction="none"
            )
            loss = loss.reshape(-1, self.config.num_pred_heads)

            ## separate compressed and raw predictions based on labels
            # [B, S, n_views] -> [B x S, n_views]
            labels_flat = labels.reshape(-1, labels.shape[-1])

            # [B, S, D] -> [B x S, D]
            hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)

            # [B x S, n_views] -> [B x S]
            all_labels_are_raw = torch.all(labels_flat < self.config.raw_vocab_size, dim=-1)

            # [B x S, D] -> [N_raw, D]
            raw_hidden_states = hidden_states_flat[all_labels_are_raw]

            # [N_raw, D] -> [N_raw, num_raw_pred_heads, raw_vocab_size]
            raw_multibyte_logits = self.raw_multibyte_lm_head(raw_hidden_states).reshape(-1, self.config.num_raw_pred_heads, self.config.raw_vocab_size)

            # NOTE: we skip bytes covered by lm_head for raw multibyte predictions
            # [B x S, n_views] -> [N_raw, num_raw_pred_heads]
            raw_multibyte_labels = labels_flat[all_labels_are_raw, self.config.num_pred_heads : self.config.num_pred_heads + self.config.num_raw_pred_heads]

            # [N_raw, num_raw_pred_heads, raw_vocab_size], [N_raw, num_raw_pred_heads] -> [N_raw x num_raw_pred_heads]
            raw_multibyte_ce_loss = F.cross_entropy(
                raw_multibyte_logits.float().view(-1, self.config.raw_vocab_size), raw_multibyte_labels.reshape(-1), reduction="none"
            )

            total_num_pred_heads = self.config.num_pred_heads + self.config.num_raw_pred_heads

            # NOTE: we divide by the total number of prediction heads to normalize the multibyte loss
            raw_multibyte_ce_loss = raw_multibyte_ce_loss / total_num_pred_heads
            raw_byte_weighting = torch.where(all_labels_are_raw, 1. / total_num_pred_heads, 1. / self.config.num_pred_heads).float()

            # [B x S, num_pred_heads], [B x S] -> [B x S, num_pred_heads]
            loss = loss * raw_byte_weighting.unsqueeze(-1)

            # NOTE: we scale loss produced by raw bytes in lm_head by the number of prediction heads to normalize the loss
            _total_loss = loss.sum() + raw_multibyte_ce_loss.sum()
            num_tokens = labels.shape[0] * labels.shape[1]
            total_loss = _total_loss / num_tokens
            return total_loss

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            if self.config.num_pred_heads > 1:
                shift_logits = logits.view(logits.shape[0], logits.shape[1], self.config.num_pred_heads, self.config.vocab_size)
                # shift_logits = shift_logits.view(-1, logits.shape[1] * self.config.num_pred_heads, self.config.vocab_size)
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
            else:
                shift_logits = logits.view(-1, self.config.vocab_size)
            shift_labels = labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits.float(), shift_labels)
            if loss_mask is not None:
                # NOTE: should clamp num_unmasked_tokens to 1 to avoid NAN (never happens)
                # NOTE: num_unmasked_tokens is local to each rank, which might be different across ranks;
                #       this might lead to some bias during global reduction. That is,
                #       mean(loss[i] / num_toks[i]) != (sum loss[i]) / (sum num_toks[i]) (should not be a problem if num_toks are roughly equal across ranks)                                                                                                              
                num_unmasked_tokens = loss_mask.float().sum()
                loss = (loss.reshape(labels.shape[0], labels.shape[1], -1) * loss_mask).sum(dim=-2).sum(dim=0) / num_unmasked_tokens
            else:
                loss = loss.reshape(labels.shape[0], labels.shape[1], -1).mean(dim=-2).mean(dim=0)
            return loss
        else:
            if self.config.num_pred_heads > 1:
                logits = logits.reshape(logits.shape[0], logits.shape[1], self.config.num_pred_heads, self.config.vocab_size)
            return logits

    def reset_parameters(self, init_args: InitArgs):
        # initialize embedding
        if init_args.init_fn == "evabyte":
            emb_init_std = 1.0
            nn.init.trunc_normal_(
                self.embed_tokens.weight,
                mean=0.0,
                std=emb_init_std,
                a=-3 * emb_init_std,
                b=3 * emb_init_std,
            )
        elif init_args.init_fn == "default":
            emb_init_std = 0.02
            nn.init.trunc_normal_(
                self.embed_tokens.weight,
                mean=0.0,
                std=emb_init_std,
                a=-3 * emb_init_std,
                b=3 * emb_init_std,
            )
        else:
            assert init_args.init_emb_std is not None
            emb_init_std = init_args.init_emb_std
            nn.init.trunc_normal_(
                self.embed_tokens.weight,
                mean=0.0,
                std=emb_init_std,
                a=-3 * emb_init_std,
                b=3 * emb_init_std,
            )

        # initialize layers
        init_in_std = init_args.init_in_std
        init_out_std = init_args.init_out_std
        for depth, layer in enumerate(self.layers):
            layer.reset_parameters(init_in_std, init_out_std)

        # initialize norm
        self.norm.reset_parameters()

        # initialize rotary embedding
        self.rotary_emb.reset_parameters()

        if self.config.apply_raw_multibyte_lm_head:
            nn.init.trunc_normal_(
                self.raw_multibyte_lm_head.weight,
                mean=0.0,
                std=init_out_std,
                a=-3 * init_out_std,
                b=3 * init_out_std,
            )
        # initialize lm head
        nn.init.trunc_normal_(
            self.lm_head.weight,
            mean=0.0,
            std=init_out_std,
            a=-3 * init_out_std,
            b=3 * init_out_std,
        )

    def init_weights(self):
        self.reset_parameters(self.init_args)

def attention_flops_per_token(n_layers, seq_len, dim, causal, chunk_size=8, window_size=2048):
    return 3.5 * (
        4 * n_layers * dim * 
        (
            (seq_len // chunk_size) * (0.5 if causal else 1) + window_size
        )
    )

def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )

def get_no_recompute_ops():
    return {
        # torch.ops.aten.mm.default,
        # torch.ops.aten._scaled_mm.default,
        torch.ops.c10d_functional.reduce_scatter_tensor.default,
        torch.ops.eva.eva_prep_kv_fwd.default,
        torch.ops.eva.eva_agg_fwd.default,
    }

# Optional and only used for fully shard options (fsdp) is choose. Highly recommended for large models
def build_fsdp_grouping_plan(model_args: EvaByteModelArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("embed_tokens", False))

    # Grouping by layers
    for i in range(model_args.num_hidden_layers):
        if i == model_args.num_hidden_layers - 1:
            group_plan.append((f"layers.{i}", False))
        else:
            group_plan.append((f"layers.{i}", True))

    if model_args.apply_raw_multibyte_lm_head:
        group_plan.append(("raw_multibyte_lm_head", False))
    group_plan.append(("lm_head", False))

    return group_plan
