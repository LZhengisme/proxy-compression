from typing import Optional, Tuple, Union
import torch
from torch import nn

MASK_MIN_VALUE = -10e10

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

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         ) -> torch.Tensor:
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

def attention_op(
        q,
        k,
        v,
        attn_mask,
        mixedp_attn,
        head_dim_scaling,
        attn_impl="native"
    ):
    if attn_impl == "sdpa":
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            ~attn_mask,
            is_causal=False,
            scale=head_dim_scaling
        )
    elif attn_impl == "native":
        attn = torch.matmul(q, k.transpose(-2, -1))
        if mixedp_attn:
            attn = attn.to(torch.float)
        attn = attn * head_dim_scaling
        if attn_mask is not None:
            # _attn_mask = attn_mask.to(attn.dtype).masked_fill(attn_mask, MASK_MIN_VALUE)
            # attn = attn + _attn_mask
            attn = attn.masked_fill(attn_mask, MASK_MIN_VALUE)
        
        attn_weights = torch.softmax(attn, dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output

def prm_projection(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    mixedp_attn: bool = False
    ):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    x: input for which features are computed
    projection_matrix: random matrix used to compute features
    Returns:
    Random features for fast attention.
    """
    # x : [..., m, d]
    # proj : [..., r, d]
    scaling_factor = (x.shape[-1] ** -0.5)
    proj_x = torch.matmul(projection_matrix, x.transpose(-1, -2)) # [..., r, m]
    norm = torch.sum(x ** 2, dim=-1).unsqueeze(-2) * 0.5 # [..., 1]
    if mixedp_attn:
        proj_x = proj_x.to(torch.float)
        norm = norm.to(torch.float)
    phi_x =  scaling_factor * (proj_x - norm)
    return phi_x

class RefEvaAttentionMixin:
    def _generate_feature_map(self, rf_q, rf_k, rf_v, rf_mask=None):
        rf_k_logits = torch.sum(self.adaptive_mu_k.to(rf_k.dtype) * rf_k, dim=-1, keepdim=True) # b h c m 1
        if self.config.mixedp_attn:
            rf_k_logits = rf_k_logits.to(torch.float)

        if rf_mask is not None:
            rf_k_logits = rf_k_logits.masked_fill(rf_mask, MASK_MIN_VALUE)

        rf_k_weights = torch.softmax(rf_k_logits, dim=-2).to(rf_k.dtype)
        rf_k_bar = torch.sum(rf_k_weights * rf_k, dim=-2)
        weights = self.adaptive_phi.to(rf_k.dtype)
        return weights, rf_k_bar

    def _calculate_chunk_rfa_cache(self, rf_q, rf_k, rf_v, weights, rf_mask=None):
        proj_x = torch.sum(weights * rf_k, dim=-1, keepdim=True)
        norm = torch.sum(rf_k ** 2, dim=-1, keepdim=True) * 0.5 # [..., 1]
        if self.config.mixedp_attn:
            proj_x = proj_x.to(torch.float)
            norm = norm.to(torch.float)
        log_phi_k = self.head_dim_scaling * (proj_x - norm)

        if rf_mask is not None:
            _rf_mask = rf_mask.to(log_phi_k.dtype).masked_fill(rf_mask, MASK_MIN_VALUE)
            log_phi_k = log_phi_k + _rf_mask
            # log_phi_k = log_phi_k.masked_fill(rf_mask, float("-inf"))

        # [b, h, c, m, r]
        softmax_phi_k = torch.softmax(log_phi_k, dim=-2).to(rf_k.dtype)
        softmax_phi_k_v = torch.sum(softmax_phi_k * rf_v, dim=-2)
        # [b, h, c, r, m] [b, h, c, m, d] -> [b, h, c, r, d]
        # softmax_phi_k_v = torch.matmul(softmax_phi_k.transpose(-1, -2), rf_v).squeeze(-2)
        log_sum_phi_k = None
        return softmax_phi_k_v, log_sum_phi_k

    def _calculate_chunk_rfa(self, q, softmax_phi_k_v, log_sum_phi_k, weights):
        return softmax_phi_k_v
    
    def window_partition(self, x, window_size=None):
        window_size = window_size if window_size is not None else self.window_size

        gw, d = x.shape[-2:]
        leading_dims = x.shape[:-2]
        n_groups = gw // window_size
        return x.reshape(*leading_dims, n_groups, window_size, d)
    
    def window_merge(self, x, window_size=None):
        g, w, d = x.shape[-3:]
        leading_dims = x.shape[:-3]
        return x.reshape(*leading_dims, g * w, d)
    
    def _eva_prep_kv(self, k, v, param_mu, param_phi, intra_chunk_mask):
        ############################################
        # compute q, k, v stats for chunk-level RFAs
        ############################################
        dump_k, dump_v = k, v

        dump_rf_mask = intra_chunk_mask

        if (
            dump_k is not None and
            dump_v is not None
        ):
            # [b, h, c, j, d]
            rf_k = self.window_partition(dump_k, window_size=self.chunk_size)
            # [b, h, c, j, d]
            rf_v = self.window_partition(dump_v, window_size=self.chunk_size)

            if dump_rf_mask is not None:
                rf_mask = self.window_partition(dump_rf_mask, window_size=self.chunk_size)
            else:
                rf_mask = None
        else:
            rf_k = None
            rf_v = None
            rf_mask = None

        rf_k_logits = torch.sum(param_mu.to(rf_k.dtype) * rf_k, dim=-1, keepdim=True) # b h c m 1
        if self.config.mixedp_attn:
            rf_k_logits = rf_k_logits.to(torch.float)

        if rf_mask is not None:
            _rf_mask = rf_mask.to(rf_k_logits.dtype).masked_fill(rf_mask, MASK_MIN_VALUE)
            rf_k_logits = rf_k_logits + _rf_mask
            # rf_k_logits = rf_k_logits.masked_fill(rf_mask, float("-inf"))

        rf_k_weights = torch.softmax(rf_k_logits, dim=-2).to(rf_k.dtype)
        rf_k_bar = torch.sum(rf_k_weights * rf_k, dim=-2)
        weights = param_phi.to(rf_k.dtype)
        softmax_phi_k_v, log_sum_phi_k = self._calculate_chunk_rfa_cache(None, rf_k, rf_v, weights, rf_mask=rf_mask)

        if rf_k_bar is not None:
            rfa_per_chunk = self._calculate_chunk_rfa(None, softmax_phi_k_v, log_sum_phi_k, weights)
        return rf_k_bar, rfa_per_chunk

    def _eva_aggregate(self, q, k, v, rf_k_bar, rfa_per_chunk, window_causal_mask, chunk_causal_mask):
        prev_s_mask = window_causal_mask # [1, 1, w, i, j]
        prev_chunk_mask = self.window_partition(chunk_causal_mask)
        if prev_s_mask.shape[-3] == 1:
            # need to expand
            prev_s_mask = prev_s_mask.expand(-1, -1, prev_chunk_mask.shape[-3], -1, -1)

        prev_w_q = self.window_partition(q) # [b, h, w, i, d]
        prev_w_k = self.window_partition(k) # [b, h, w, j, d]
        prev_w_v = self.window_partition(v) # [b, h, w, j, d]

        if prev_w_k is not None:
            if rf_k_bar is not None:
                num_windows = prev_w_k.shape[-3]
                # rf_k_bar and rfa_per_chunk take the shape [b, h, c, d]
                # -> [b, h, 1, c, d] -> [b, h, w, c, d]
                prev_rf_k_bar = rf_k_bar.unsqueeze(-3).expand(-1, -1, num_windows, -1, -1)
                prev_rfa_per_chunk = rfa_per_chunk.unsqueeze(-3).expand(-1, -1, num_windows, -1, -1)
                prev_agg_k = torch.cat([prev_w_k, prev_rf_k_bar], dim=-2)
                prev_agg_v = torch.cat([prev_w_v, prev_rfa_per_chunk], dim=-2)

                prev_attn_mask = torch.cat([prev_s_mask, prev_chunk_mask], dim=-1)
            else:
                prev_agg_k = prev_w_k
                prev_agg_v = prev_w_v
                prev_attn_mask = prev_s_mask

            prev_attn_output = attention_op(
                q=prev_w_q,
                k=prev_agg_k,
                v=prev_agg_v,
                attn_mask=prev_attn_mask,
                mixedp_attn=self.config.mixedp_attn,
                head_dim_scaling=self.head_dim_scaling,
                attn_impl=self.config.attn_impl if hasattr(self.config, "attn_impl") else "native"
            )
            prev_attn_output = self.window_merge(prev_attn_output)

        return prev_attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        if len(attention_mask) == 3:
            window_causal_mask, chunk_causal_mask, intra_chunk_mask = attention_mask
        else:
            raise NotImplementedError("Only attention-mask tuple with length 2 or 3 is supported")

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
        # compute q, k, v stats for the local window
        ############################################
        prev_w_q = self.window_partition(q) # [b, h, w, i, d]
        prev_w_k = self.window_partition(k) # [b, h, w, j, d]
        prev_w_v = self.window_partition(v) # [b, h, w, j, d]
        # during training, we assume window_size divides seq_len so no remainders
        cur_w_q = cur_w_k = cur_w_v = None

        ############################################
        # compute q, k, v stats for chunk-level RFAs
        ############################################
        dump_q, dump_k, dump_v = q, k, v

        prev_s_mask = self.window_partition(window_causal_mask) # [1, 1, w, i, j]
        cur_s_mask = None
        prev_chunk_mask = self.window_partition(chunk_causal_mask)
        cur_chunk_mask = None
        dump_rf_mask = intra_chunk_mask
        if prev_s_mask.shape[-3] == 1:
            # need to expand
            prev_s_mask = prev_s_mask.expand(-1, -1, prev_chunk_mask.shape[-3], -1, -1)

        if (
            dump_q is not None and
            dump_k is not None and
            dump_v is not None
        ):
            # [b, h, c, j, d]
            rf_q = self.window_partition(dump_q, window_size=self.chunk_size)
            # [b, h, c, j, d]
            rf_k = self.window_partition(dump_k, window_size=self.chunk_size)
            # [b, h, c, j, d]
            rf_v = self.window_partition(dump_v, window_size=self.chunk_size)

            if dump_rf_mask is not None:
                rf_mask = self.window_partition(dump_rf_mask, window_size=self.chunk_size)
            else:
                rf_mask = None
        else:
            rf_q = None
            rf_k = None
            rf_v = None
            rf_mask = None


        if rf_q is not None:
            # import pdb; pdb.set_trace()
            weights, rf_k_bar = self._generate_feature_map(rf_q, rf_k, rf_v, rf_mask=rf_mask)
            softmax_phi_k_v, log_sum_phi_k = self._calculate_chunk_rfa_cache(rf_q, rf_k, rf_v, weights, rf_mask=rf_mask)
        else:
            weights = None
            softmax_phi_k_v = None
            log_sum_phi_k = None
            rf_k_bar = None

        if rf_k_bar is not None:
            rfa_per_chunk = self._calculate_chunk_rfa(q, softmax_phi_k_v, log_sum_phi_k, weights)
        ############################################
        # compute meta-attention weights for 
        # - group-wise RFAs and 
        # - singletons (equivalent to exact local attention)
        ############################################
        if prev_w_k is not None:
            if rf_k_bar is not None:
                num_windows = prev_w_k.shape[-3]
                # rf_k_bar and rfa_per_chunk take the shape [b, h, c, d]
                # -> [b, h, 1, c, d] -> [b, h, w, c, d]
                prev_rf_k_bar = rf_k_bar.unsqueeze(-3).expand(-1, -1, num_windows, -1, -1)
                prev_rfa_per_chunk = rfa_per_chunk.unsqueeze(-3).expand(-1, -1, num_windows, -1, -1)
                prev_agg_k = torch.cat([prev_w_k, prev_rf_k_bar], dim=-2)
                prev_agg_v = torch.cat([prev_w_v, prev_rfa_per_chunk], dim=-2)

                prev_attn_mask = torch.cat([prev_s_mask, prev_chunk_mask], dim=-1)
            else:
                prev_agg_k = prev_w_k
                prev_agg_v = prev_w_v
                prev_attn_mask = prev_s_mask

            prev_attn_output = attention_op(
                q=prev_w_q,
                k=prev_agg_k,
                v=prev_agg_v,
                attn_mask=prev_attn_mask,
                mixedp_attn=self.config.mixedp_attn,
                head_dim_scaling=self.head_dim_scaling,
                attn_impl=self.config.attn_impl if hasattr(self.config, "attn_impl") else "native"
            )
            prev_attn_output = self.window_merge(prev_attn_output)

        if cur_w_k is not None:
            if rf_k_bar is not None:
                # rf_k_bar and rfa_per_chunk take the shape [b, h, c, d]
                # cur_w_k and cur_w_v also has shape [b, h, m, d]
                cur_agg_k = torch.cat([cur_w_k, rf_k_bar], dim=-2)
                cur_agg_v = torch.cat([cur_w_v, rfa_per_chunk], dim=-2)

                cur_attn_mask = torch.cat([cur_s_mask, cur_chunk_mask], dim=-1)
            else:
                cur_agg_k = cur_w_k
                cur_agg_v = cur_w_v
                cur_attn_mask = cur_s_mask

            cur_attn_output = attention_op(
                q=cur_w_q,
                k=cur_agg_k,
                v=cur_agg_v,
                attn_mask=cur_attn_mask,
                mixedp_attn=self.config.mixedp_attn,
                head_dim_scaling=self.head_dim_scaling,
                attn_impl=self.config.attn_impl if hasattr(self.config, "attn_impl") else "native"
            )

        if prev_w_k is not None and cur_w_k is not None:
            attn_output = torch.cat([prev_attn_output, cur_attn_output], dim=-2)
        elif prev_w_k is not None:
            attn_output = prev_attn_output
        elif cur_w_k is not None:
            attn_output = cur_attn_output
        else:
            raise ValueError("There must be some bug")

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output