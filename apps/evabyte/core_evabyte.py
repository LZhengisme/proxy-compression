from typing import List, Optional, Tuple, Union
import math
import torch
from torch import nn
import torch.nn.functional as F
from apps.evabyte.component.eva_agg_kernel_triton_op import eva_agg_func_triton
from apps.evabyte.component.eva_prep_kv_kernel_triton_op import eva_prep_kv_func_triton
from dataclasses import dataclass, field
from lingua.probe import log_stats

@dataclass
class InitArgs:
    init_fn: str = "evabyte"
    init_emb_std: Optional[float] = None
    init_in_std: float = 0.02
    init_out_std: float = 0.02
    init_out_factor: float = 1.0

@dataclass
class BaseEvaByteArgs:
    vocab_size: int = 320

    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32

    max_position_embeddings: int = 32768
    norm_add_unit_offset: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100000.0

    init_args: InitArgs = field(default_factory=InitArgs)

    window_size: int = 2048
    num_chunks: Optional[int] = None
    attention_class: str = "eva"
    chunk_size: int = 16
    num_pred_heads: int = 8

    apply_raw_multibyte_lm_head: bool = False
    raw_vocab_size: int = 320
    num_raw_pred_heads: int = 1

def prepare_eva_attention_mask(
        seq_len, 
        device, 
        chunk_size, 
        window_size,
        use_cache=False, 
        cache=None
    ):
    """
    Prepare attention masks for EVA.
    
    """
    chunk_causal_mask  = None
    window_causal_mask = None
    if use_cache:
        cached_seq_len = cache.get_seq_length()
        total_seq_len = seq_len + cached_seq_len
        # cached_seq_len will be 0 during prefilling
        # padded_seq_len = chunk_size * math.ceil(total_seq_len / chunk_size)
        padded_seq_len = window_size * math.ceil(total_seq_len / window_size)
        num_chunks = padded_seq_len // chunk_size
    else:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        assert seq_len % chunk_size == 0
        num_chunks = seq_len // chunk_size

        assert seq_len % window_size == 0

    # create causal mask
    ################################
    # generate chunked causal masks
    ################################
    # [b, h, j, c, c]
    chunks_per_window = window_size // chunk_size
    if num_chunks >= chunks_per_window:
        chunk_causal_mask = torch.ones(
            (chunk_size, num_chunks, num_chunks), 
            device=device,
            dtype=torch.bool
        ).triu(0)
        
        num_blocks = num_chunks // chunks_per_window
        chunk_causal_mask = chunk_causal_mask.reshape(
            chunk_size,
            num_blocks, 
            chunks_per_window, 
            num_blocks, 
            chunks_per_window
        ).transpose(-2, -3)

        block_diag_zero = (
            torch.eye(num_blocks, device=device, dtype=torch.bool)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        # Set diagonal blocks to zero
        chunk_causal_mask = chunk_causal_mask.masked_fill(block_diag_zero, True)

        # Reshape back to original size
        chunk_causal_mask = (
            chunk_causal_mask
            .transpose(-2, -3)
            .reshape(chunk_size, num_chunks, num_chunks)
            .transpose(-2, -3)
            .reshape(chunk_size * num_chunks, num_chunks)
            .unsqueeze(0)
            .unsqueeze(0)
        )
    else:
        chunk_causal_mask = torch.ones(
            (1, 1, chunk_size, num_chunks, num_chunks), 
            device=device,
            dtype=torch.bool,
        ).triu(0).transpose(-2, -3) # [1, 1, c, j, c]
        chunk_causal_mask = chunk_causal_mask.reshape(
            1, 1, chunk_size * num_chunks, num_chunks
        ) # [1, 1, n, c]

    if use_cache:
        chunk_causal_mask = chunk_causal_mask[..., cached_seq_len : cached_seq_len + seq_len, :]

    window_causal_mask = torch.ones(
        (1, 1, 1, window_size, window_size), 
        device=device
    ).triu(1).to(torch.bool)
    return (chunk_causal_mask, window_causal_mask)

def prepare_eva_training_mask(
    target_token_type_ids,
    use_doc_boundary_attention, 
    chunk_size,
    window_size,
    EOS_TOKEN_TYPE_ID=None,
    PAD_TOKEN_TYPE_ID=None,
):
    '''
    This function prepares the attention mask for training EvaByte.
        target_token_type_ids:
            Tensor of shape (batch_size, seq_len), marking the token type ids 
            for the target sequence. In particular, this function expects
                - target_token_type_ids[i, j] = EOS_TOKEN_TYPE_ID 
                    if the j-th token in the i-th sequence is the end of an article.
                - target_token_type_ids[i, j] = PAD_TOKEN_TYPE_ID 
                    if the j-th token in the i-th sequence is the padding token.
        use_doc_boundary_attention: bool, 
            whether to enable doc boundary attention.
        EOS_TOKEN_TYPE_ID: int, 
            the token type id for the end of an article.
        PAD_TOKEN_TYPE_ID: int, 
            the token type id for the padding token.
    '''
    batch_size, num_tokens = target_token_type_ids.shape

    chunk_causal_mask, window_causal_mask = prepare_eva_attention_mask(
        num_tokens, 
        target_token_type_ids.device, 
        chunk_size=chunk_size, 
        window_size=window_size,
        use_cache=False,
        cache=None
    )
    if use_doc_boundary_attention:
        #### step 1: mark each document with a unique id
        end_token_ids = {EOS_TOKEN_TYPE_ID, PAD_TOKEN_TYPE_ID}
        token_types = torch.zeros(batch_size, num_tokens)
        for sequence_idx, sequence in enumerate(target_token_type_ids):
            num_articles = 0
            start_index = 0
            # for each sample in the batch, the collapsed attention mask looks like:
            # [1, 1, .... 1, 0, 2, 2, ... 2, 0, ... n, n ..... n], assuming there are n articles in the sequence.
            # Each of the n articles are separated by 0.
            for token_idx, token_type_id in enumerate(sequence):
                if start_index is not None and token_type_id.item() in end_token_ids:
                    num_articles += 1
                    end_index = token_idx if token_type_id == PAD_TOKEN_TYPE_ID else token_idx
                    token_types[sequence_idx][start_index:end_index] = num_articles
                    start_index = None
                elif start_index is None and token_type_id.item() not in end_token_ids:
                    start_index = token_idx

            if start_index is not None:
                num_articles += 1
                token_types[sequence_idx][start_index:] = num_articles
        assert num_tokens % chunk_size == 0, "Number of tokens must be divisible by chunk size"
        assert num_tokens % window_size == 0, "Number of tokens must be divisible by window size"
        num_chunks = num_tokens // chunk_size
        num_windows = num_tokens // window_size

        article_separator = 0

        #### step 2: generate attention masks for each window
        #### NOTE: we perform exact attention within each window, 
        ####       so we only need to mask out different documents
        ####       for each window.
        token_types_windows = token_types.reshape(batch_size, num_windows, window_size, 1)
        token_types_windows_t = token_types_windows.transpose(-1, -2)
        # replace all elements in TOKEN_SEPS with -1
        token_types_windows = torch.where(token_types_windows == article_separator, -1, token_types_windows)
        window_3d_mask = (token_types_windows == token_types_windows_t)
        window_3d_mask = ~window_3d_mask

        #### step 3: generate chunk-level 3D masks
        #### NOTE: this is a bit tricky, as we aim to mask out different 
        ####       documents to avoid cross-doc attention across chunks.
        #### Example: suppose we have a sequence of length 12 with 3 documents:
        ####       [1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3].
        ####       The chunk-size and window-size are both 4.
        ####       The chunk-level mask of shape (batch_size, seq_len, num_chunks) is:
        ####       [
        ####         [0, 0, 0],
        ####         [0, 0, 0],
        ####         [0, 0, 0],
        ####         [0, 0, 0],
        ####
        ####         [1, 0, 0],
        ####         [0, 0, 0],
        ####         [0, 0, 0],
        ####         [0, 0, 0],
        ####
        ####         [0, 1, 0],
        ####         [0, 1, 0],
        ####         [0, 1, 0],
        ####         [0, 1, 0],
        ####       ]
        ####       Explanation:
        ####       - Tokens will not attend to their own and future chunks.
        ####         (as tokens within a chunk are captured by the window-level exact attention)
        ####       - Tokens will attend to a chunk only if there are tokens 
        ####         from the same document in that chunk.
        ####       The mask within each chunk of shape (batch_size, num_chunks, chunk_size) is:
        ####       [
        ####         [1, 1, 1, 1],
        ####         [0, 0, 0, 1],
        ####         [1, 1, 1, 1],
        ####       ]
        ####       Explanation:
        ####       - If all tokens in a chunk are from the same document, 
        ####         no tokens will be masked out.
        ####       - If there are tokens from different documents in a chunk, 
        ####         only tokens from the rightmost document will be kept.
        ####         (b/c the future chunks might contain tokens from the rightmost document,
        ####         but all the remaining docs will never get attended by other docs)
        token_types_chunks = token_types.reshape(batch_size, num_chunks, chunk_size)
        inter_chunk_mask = torch.zeros((batch_size, num_tokens, num_chunks), dtype=torch.bool)
        intra_chunk_mask = torch.ones_like(token_types_chunks, dtype=torch.bool)
        
        for chunk_idx in range(num_chunks):
            for batch_idx in range(batch_size):
                # Identify tokens in the current chunk belonging to each sequence
                chunk = token_types_chunks[batch_idx, chunk_idx]
                unique_elements = torch.unique(chunk, sorted=True).tolist()
                
                # Create a mask for whether each token can attend to the current chunk
                for token_type in unique_elements:
                    if token_type == article_separator:
                        continue
                    token_mask = (token_types[batch_idx] == token_type)
                    inter_chunk_mask[batch_idx, :, chunk_idx] |= token_mask

                # Create a mask within each chunk
                unique_elements = [x for x in unique_elements if x != article_separator]
                if len(unique_elements) >= 1 and chunk[-1] != article_separator:
                    intra_chunk_mask[batch_idx, chunk_idx] = (chunk == unique_elements[-1])
        
        inter_chunk_mask = ~inter_chunk_mask
        intra_chunk_mask = ~intra_chunk_mask

        window_mask = torch.logical_or(window_causal_mask, window_3d_mask.unsqueeze(1))
        inter_chunk_mask = torch.logical_or(chunk_causal_mask, inter_chunk_mask.unsqueeze(1))
        intra_chunk_mask = intra_chunk_mask

        attention_mask = (
            window_mask.reshape(batch_size, 1, num_tokens, window_size), 
            inter_chunk_mask, 
            intra_chunk_mask.reshape(batch_size, 1, num_tokens, 1)
        )
    else:
        attention_mask = None
    return attention_mask

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims (last dim) of the input.
    Args:
        x: Rotary embedded tensor
    Return:
        Tensor with half of last dim negated and rotated to the front.
    """
    x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
        q: torch.Tensor, 
        k: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding (cos, sin) to the query and key tensor on the sequence dimension.

    The legends for dimensions are defined as:
    num_heads: number of attention heads
    current_seq_len: the current batch's sequence length, should be either 1 or max_seq_len
    max_seq_len: the static sequence length, different from current_seq_len in cached inference case where it is always
                 maximum lenghth, e.g. the length of static sequence length of KV cache

                 
    Args:
        q: Query tensor, of size (batch_size, num_heads, current_seq_len, head_dim)
        k: Key tensor, of size (batch_size, num_key_value_heads, current_seq_len, head_dim)
        cos: Cosine base of rotary embedding, of size (max_seq_len, head_dim)
        sin: Sine base of rotary embedding, of size (max_seq_len, head_dim)
        position_ids: The position indices of the tokens corresponding to the query and key tensors. It has a size of
                      (batch_size, current_seq_len).

    Returns:
        Embedded query and key tensor of same size as input.
    
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class EvaAttention(nn.Module):
    """
        Causal EVA for language modeling.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.head_dim_scaling = self.head_dim ** -0.5

        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.window_size = config.window_size
        
        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        if self.chunk_size is not None:
            assert self.window_size >= self.chunk_size and self.window_size % self.chunk_size == 0
            # chunk_size overrides the number of landmarks
            self.num_chunks = None

        self.chunks_per_window = int(self.window_size // self.chunk_size)
        self.adaptive_phi = nn.Parameter(
            torch.empty(
                1,
                self.num_heads,
                1,
                1,
                self.head_dim
            )
        )
        self.adaptive_mu_k = nn.Parameter(
            torch.empty(
                1,
                self.num_heads,
                1,
                1,
                self.head_dim
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        window_mask: Optional[torch.Tensor],
        chunk_mask: Optional[torch.Tensor], 
        intra_chunk_mask: Optional[torch.Tensor],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        ############################################
        # compute q, k, v from hidden states
        ############################################
        # [b, h, q_len, d]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, h, kv_len, d]
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, h, kv_len, d]
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        ############################################
        # apply rotary positional embeddings to q, k
        ############################################
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        ############################################
        # update and get cached singleton tokens
        # update and cache k and v for calculating chunk-level RFAs
        ############################################
        s_k, s_v = k, v
        dump_k, dump_v = k, v

        singleton_mask = window_mask
        dump_rf_mask = intra_chunk_mask

        adaptive_mu_k = log_stats(self.adaptive_mu_k, "param_psi")
        adaptive_phi = log_stats(self.adaptive_phi, "param_phi")

        rfa_k, rfa_v = eva_prep_kv_func_triton(
            dump_k, dump_v, 
            adaptive_mu_k, adaptive_phi, 
            dump_rf_mask, self.head_dim_scaling, self.chunk_size
        )
        # rfa_mask = get_rfa_chunk_mask(dump_rf_mask)

        rfa_k = log_stats(rfa_k, "rfa_k")
        rfa_v = log_stats(rfa_v, "rfa_v")
        ############################################
        # compute the full attention output
        ############################################
        attn_output = eva_agg_func_triton(
            q, s_k, s_v, 
            rfa_k, rfa_v, 
            singleton_mask, chunk_mask,
            self.head_dim_scaling, self.window_size, self.chunks_per_window
        )
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = log_stats(attn_output, "attn_output")
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output
    
    def reset_parameters(self, init_in_std, init_out_std):
        nn.init.trunc_normal_(
            self.adaptive_mu_k,
            mean=0.0,
            std=self.head_dim_scaling,
            a=-2.0 * self.head_dim_scaling,
            b=2.0 * self.head_dim_scaling,
        )
        nn.init.trunc_normal_(
            self.adaptive_phi,
            mean=0.0,
            std=self.head_dim_scaling,
            a=-2.0 * self.head_dim_scaling,
            b=2.0 * self.head_dim_scaling,
        )
        for w in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_in_std,
                a=-3 * init_in_std,
                b=3 * init_in_std,
            )

        nn.init.trunc_normal_(
            self.o_proj.weight,
            mean=0.0,
            std=init_out_std,
            a=-3 * init_out_std,
            b=3 * init_out_std,
        )

class EvaByteRMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.variance_epsilon = config.rms_norm_eps
        self.add_unit_offset = config.norm_add_unit_offset
        if self.add_unit_offset:
            self.weight = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        hidden_states = log_stats(hidden_states, "resid")
        x = hidden_states.float()

        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        if self.add_unit_offset:
            weight = (1 + self.weight.float())
        else:
            weight = self.weight.float()
        return (weight * output).type_as(hidden_states)

    def reset_parameters(self):
        if self.add_unit_offset:
            torch.nn.init.zeros_(self.weight)
        else:
            torch.nn.init.ones_(self.weight)

class EvaByteRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", self._precompute_freqs(), persistent=False)

        cos_cached, sin_cached = self._precompute_cos_sin_cache(
            self.inv_freq,
            max_position_embeddings,
            self.inv_freq.device,
            torch.get_default_dtype()
        )
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def _precompute_freqs(self):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

    def _precompute_cos_sin_cache(self, inv_freq, seq_len, device, dtype=None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if dtype is None:
            return emb.cos(), emb.sin()
        else:
            return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len < self.max_seq_len_cached:
            cos_slice = self.cos_cached[:seq_len]
            sin_slice = self.sin_cached[:seq_len]
        else:
            cos_slice = self.cos_cached
            sin_slice = self.sin_cached

        return (
            cos_slice.to(dtype=x.dtype),
            sin_slice.to(dtype=x.dtype)
        )

    def reset_parameters(self):
        inv_freq = self._precompute_freqs()
        self.inv_freq[...] = inv_freq
        cos_cached, sin_cached = self._precompute_cos_sin_cache(
            inv_freq,
            self.max_position_embeddings,
            inv_freq.device,
        )
        self.cos_cached[...] = cos_cached
        self.sin_cached[...] = sin_cached

class EvaByteMLP(nn.Module):
    def __init__(self, config, layer_idx: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.layer_idx = layer_idx
        self.config = config

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def reset_parameters(self, init_in_std, init_out_std):
        for w in [self.gate_proj, self.up_proj]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_in_std,
                a=-3 * init_in_std,
                b=3 * init_in_std,
            )
        nn.init.trunc_normal_(
            self.down_proj.weight,
            mean=0.0,
            std=init_out_std,
            a=-3 * init_out_std,
            b=3 * init_out_std,
        )

class EvaByteDecoderLayer(nn.Module):
    def __init__(self, config: BaseEvaByteArgs, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = EvaAttention(config=config, layer_idx=layer_idx)
        self.mlp = EvaByteMLP(config, layer_idx=layer_idx)
        self.input_layernorm = EvaByteRMSNorm(config)
        self.post_attention_layernorm = EvaByteRMSNorm(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            window_mask: Optional[torch.Tensor],
            chunk_mask: Optional[torch.Tensor], 
            intra_chunk_mask: Optional[torch.Tensor],
            cos: Optional[torch.Tensor] = None,
            sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            window_mask=window_mask,
            chunk_mask=chunk_mask, 
            intra_chunk_mask=intra_chunk_mask,
            cos=cos,
            sin=sin,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def reset_parameters(self, init_in_std, init_out_std):
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.reset_parameters(init_in_std, init_out_std)
        self.mlp.reset_parameters(init_in_std, init_out_std)
