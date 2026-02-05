"""
NumPy implementations of document-boundary attention mask functions for EVA.
These functions replace PyTorch implementations to avoid multiprocessing issues.
"""
import torch
import numpy as np
import random
from typing import Optional, Tuple, Union
from apps.evabyte.core_evabyte import prepare_eva_attention_mask

def prepare_multibyte_loss_weight(
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    num_pred_heads: int,
    vocab_size: int,
    raw_byte_offset: int,
    bos_id: int,
    raw_sentinel_id_start: int,
    compressed_sentinel_id_start: int,
    disable_cross_byte_prediction: bool = False,
    weighting_compressed_prediction: bool = False,
    compressed_loss_weight: float = 0.33333,
):
    # input_ids: (batch_size, seq_len)
    # labels: (batch_size, seq_len, num_pred_heads)
    # num_pred_heads: int
    # vocab_size: int

    pred_head_weight = torch.ones_like(labels, dtype=torch.float32)

    if weighting_compressed_prediction:
        # apply exponential downweighting over prediction heads
        _coeff = 0.75 ** torch.arange(num_pred_heads, dtype=torch.float32).reshape(1, 1, num_pred_heads)
        all_labels_are_raw = torch.all(labels < raw_byte_offset, dim=-1, keepdim=True)
        pred_head_weight = torch.where(all_labels_are_raw, pred_head_weight, _coeff)
        # downweight each compressed byte prediction by 1/3 to balance raw bytes
        compressed_byte_weights = torch.ones_like(all_labels_are_raw, dtype=torch.float32)
        compressed_byte_weights = torch.where(all_labels_are_raw, compressed_byte_weights, compressed_loss_weight)
    else:
        compressed_byte_weights = 1.0

    cross_sample_mask = (labels == bos_id) | (labels == raw_sentinel_id_start) | (labels == compressed_sentinel_id_start)
    cross_sample_mask = torch.cumsum(cross_sample_mask, dim=-1) <= 0
    pred_head_weight = pred_head_weight * cross_sample_mask.to(pred_head_weight.dtype)

    if disable_cross_byte_prediction:
        inputs_are_raw = (input_ids < raw_byte_offset).unsqueeze(-1)
        ignored_label_mask = torch.cumsum(labels >= raw_byte_offset, dim=-1) > 0
        weight_to_keep = ~(inputs_are_raw & ignored_label_mask)
        pred_head_weight = pred_head_weight * weight_to_keep

    normalizing_constant = pred_head_weight.sum(dim=-1, keepdim=True)
    normalizing_constant_removing_all_zeros = normalizing_constant.masked_fill(normalizing_constant <= 0.0, 1.0)

    pred_head_weight = compressed_byte_weights * pred_head_weight / normalizing_constant_removing_all_zeros

    return pred_head_weight

def prepare_multibyte_loss_weight_numpy(
    input_ids: np.ndarray,
    labels: np.ndarray,
    num_pred_heads: int,
    vocab_size: int,
    raw_byte_offset: int,
    bos_id: int,
    raw_sentinel_id_start: int,
    compressed_sentinel_id_start: int,
    disable_cross_byte_prediction: bool = False,
    weighting_compressed_prediction: bool = False,
    compressed_loss_weight: float = 0.33333,
) -> np.ndarray:
    """
    Compute per-head loss weights for multi-byte prediction using NumPy.

    For each token position, assigns a weight to each prediction head based on whether all labels are "raw" (label < raw_byte_offset)
    or if any label is a special token (BOS, raw sentinel, compressed sentinel). If all labels are raw, all heads get weight 1.
    Otherwise, weights decay by 0.5 per head (i.e., [1.0, 0.5, 0.25, ...]). Heads after the first special token in a position are masked out.
    Optionally, if `disable_cross_byte_prediction` is True, further mask out heads after the first non-raw label for raw input tokens.

    Args:
        input_ids: np.ndarray of shape (batch_size, seq_len)
            Input token ids.
        labels: np.ndarray of shape (batch_size, seq_len, num_pred_heads)
            Target labels for each prediction head.
        num_pred_heads: int
            Number of prediction heads.
        vocab_size: int
            Vocabulary size.
        raw_byte_offset: int
            All ids < raw_byte_offset are considered "raw" bytes.
        bos_id: int
            BOS token id; marks the start of a new document.
        raw_sentinel_id_start: int
            Raw sentinel token id; marks the start of a new raw document.
        compressed_sentinel_id_start: int
            Compressed sentinel token id; marks the start of a new compressed document.
        disable_cross_byte_prediction: bool, default False
            If True, disables cross-byte prediction for raw input tokens by masking out heads after the first non-raw label.

    Returns:
        np.ndarray of shape (batch_size, seq_len, num_pred_heads), dtype float32
            Normalized per-head weights for each token position.
    """
    pred_head_weight = np.ones_like(labels, dtype=np.float32)
    if weighting_compressed_prediction:
        coeff = (0.75 ** np.arange(num_pred_heads, dtype=np.float32)).reshape(1, 1, num_pred_heads)
        all_labels_are_raw = np.all(labels < raw_byte_offset, axis=-1, keepdims=True)
        pred_head_weight = np.where(all_labels_are_raw, pred_head_weight, coeff)
        compressed_byte_weights = np.ones_like(all_labels_are_raw, dtype=np.float32)
        compressed_byte_weights = np.where(all_labels_are_raw, compressed_byte_weights, compressed_loss_weight)
    else:
        compressed_byte_weights = 1.0

    specials = (
        (labels == bos_id)
        | (labels == raw_sentinel_id_start)
        | (labels == compressed_sentinel_id_start)
    )
    cross_sample_mask = np.cumsum(specials.astype(np.int64), axis=-1) <= 0
    pred_head_weight = pred_head_weight * cross_sample_mask.astype(np.float32)

    if disable_cross_byte_prediction:
        inputs_are_raw = (input_ids < raw_byte_offset)[:, :, None]
        ignored_label_mask = np.cumsum(labels >= raw_byte_offset, axis=-1) > 0
        weight_to_keep = ~(inputs_are_raw & ignored_label_mask)
        pred_head_weight = pred_head_weight * weight_to_keep

    normalizing_constant = pred_head_weight.sum(axis=-1, keepdims=True)
    normalizing_constant_removing_all_zeros = np.where(normalizing_constant <= 0.0, 1.0, normalizing_constant)
    pred_head_weight = compressed_byte_weights * pred_head_weight / normalizing_constant_removing_all_zeros

    return pred_head_weight.astype(np.float32)

def prepare_eva_token_type_ids(
    input_ids,
    chunk_size,
    window_size,
    EOS_TOKEN_TYPE_ID=None,
    PAD_TOKEN_TYPE_ID=None,
):
    '''
    This function prepares the attention mask for training EvaByte.
        input_ids:
            Tensor of shape (batch_size, seq_len), marking the token type ids 
            for the target sequence. In particular, this function expects
                - input_ids[i, j] = EOS_TOKEN_TYPE_ID 
                    if the j-th token in the i-th sequence is the end of an article.
                - input_ids[i, j] = PAD_TOKEN_TYPE_ID 
                    if the j-th token in the i-th sequence is the padding token.
        EOS_TOKEN_TYPE_ID: int, 
            the token type id for the end of an article.
        PAD_TOKEN_TYPE_ID: int, 
            the token type id for the padding token.
    '''
    batch_size, num_tokens = input_ids.shape
    #### step 1: mark each document with a unique id
    end_token_ids = {EOS_TOKEN_TYPE_ID, PAD_TOKEN_TYPE_ID}
    token_types = torch.zeros(batch_size, num_tokens)
    for sequence_idx, sequence in enumerate(input_ids):
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

    article_separator = 0
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
    chunked_token_type_ids = torch.zeros((2, batch_size, num_chunks), dtype=torch.int64)
    intra_chunk_mask = torch.ones_like(token_types_chunks, dtype=torch.bool)
    
    for chunk_idx in range(num_chunks):
        for batch_idx in range(batch_size):
            # Identify tokens in the current chunk belonging to each sequence
            chunk = token_types_chunks[batch_idx, chunk_idx]

            chunked_token_type_ids[0, batch_idx, chunk_idx] = chunk[0]
            chunked_token_type_ids[1, batch_idx, chunk_idx] = chunk[-1]

            # Create a mask within each chunk
            unique_elements = torch.unique(chunk, sorted=True).tolist()
            unique_elements = [x for x in unique_elements if x != article_separator]
            if len(unique_elements) >= 1 and chunk[-1] != article_separator:
                intra_chunk_mask[batch_idx, chunk_idx] = (chunk == unique_elements[-1])
    intra_chunk_mask = ~intra_chunk_mask
    return token_types, chunked_token_type_ids, intra_chunk_mask


def prepare_eva_token_type_ids_numpy(
    input_ids: np.ndarray,
    chunk_size: int,
    window_size: int,
    EOS_TOKEN_TYPE_ID: Optional[int] = None,
    PAD_TOKEN_TYPE_ID: Optional[int] = None,
) -> Union[Tuple, None]:
    '''
    This function prepares the attention mask for training EvaByte.
        input_ids:
            Numpy Array of shape (batch_size, seq_len), marking the token type ids 
            for the target sequence. In particular, this function expects
                - input_ids[i, j] = EOS_TOKEN_TYPE_ID 
                    if the j-th token in the i-th sequence is the end of an article.
                - input_ids[i, j] = PAD_TOKEN_TYPE_ID 
                    if the j-th token in the i-th sequence is the padding token.
        EOS_TOKEN_TYPE_ID: int, 
            the token type id for the end of an article.
        PAD_TOKEN_TYPE_ID: int, 
            the token type id for the padding token.
    '''
    batch_size, num_tokens = input_ids.shape
    # Create document boundary masks
    end_token_ids = {EOS_TOKEN_TYPE_ID, PAD_TOKEN_TYPE_ID}
    
    token_types = np.zeros((batch_size, num_tokens))
    for sequence_idx in range(batch_size):
        sequence = input_ids[sequence_idx]
        num_articles = 0
        start_index = 0
        
        for token_idx in range(len(sequence)):
            token_type_id = sequence[token_idx]
            if start_index is not None and token_type_id in end_token_ids:
                num_articles += 1
                end_index = token_idx if token_type_id == PAD_TOKEN_TYPE_ID else token_idx
                token_types[sequence_idx][start_index:end_index] = num_articles
                start_index = None
            elif start_index is None and token_type_id not in end_token_ids:
                start_index = token_idx

        if start_index is not None:
            num_articles += 1
            token_types[sequence_idx][start_index:] = num_articles
    assert num_tokens % chunk_size == 0, "Number of tokens must be divisible by chunk size"
    assert num_tokens % window_size == 0, "Number of tokens must be divisible by window size"
    num_chunks = num_tokens // chunk_size
    article_separator = 0

    # Generate chunk-level masks
    token_types_chunks = token_types.reshape(batch_size, num_chunks, chunk_size)
    chunked_token_type_ids = np.zeros((2, batch_size, num_chunks), dtype=np.int64)
    intra_chunk_mask = np.ones_like(token_types_chunks, dtype=bool)

    for chunk_idx in range(num_chunks):
        for batch_idx in range(batch_size):
            chunk = token_types_chunks[batch_idx, chunk_idx]

            chunked_token_type_ids[0, batch_idx, chunk_idx] = chunk[0]
            chunked_token_type_ids[1, batch_idx, chunk_idx] = chunk[-1]

            unique_elements = np.unique(chunk)
            # Intra-chunk mask: if multiple docs in chunk, keep only rightmost
            unique_elements = unique_elements[unique_elements != article_separator]
            if len(unique_elements) >= 1 and chunk[-1] != article_separator:
                intra_chunk_mask[batch_idx, chunk_idx] = (chunk == unique_elements[-1])
    
    intra_chunk_mask = ~intra_chunk_mask
    return token_types, chunked_token_type_ids, intra_chunk_mask



def prepare_token_types_position_ids(
    input_ids: torch.LongTensor,
    chunk_size: int,
    window_size: int,
    eos_token_id: int,
    pad_token_id: Optional[int] = None,
):
    token_types, chunked_token_type_ids, intra_chunk_mask = prepare_eva_token_type_ids(
        input_ids,
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
    return token_types, chunked_token_type_ids, intra_chunk_mask, position_ids

def prepare_token_types_position_ids_numpy(
    input_ids: np.ndarray,
    chunk_size: int,
    window_size: int,
    eos_token_id: int,
    pad_token_id: Optional[int] = None,
) -> Tuple[Union[Tuple, None], np.ndarray]:
    token_types, chunked_token_type_ids, intra_chunk_mask = prepare_eva_token_type_ids_numpy(
        input_ids,
        chunk_size,
        window_size,
        EOS_TOKEN_TYPE_ID=eos_token_id,
        PAD_TOKEN_TYPE_ID=pad_token_id,
    )

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
    return token_types, chunked_token_type_ids, intra_chunk_mask, position_ids

def calculate_3d_attention_mask_gpu(
    token_type_ids,
    chunked_token_type_ids,
    intra_chunk_mask,
    chunk_size,
    window_size,
):
    '''
    This function calculates the 3D attention mask for EvaByte.
    token_type_ids:
        Tensor of shape (batch_size, num_tokens), marking the token type ids 
        for the target sequence.
        # [1, 1, .... 1, 0, 2, 2, ... 2, 0, ... n, n ..... n], assuming there are n articles in the sequence.
        # Each of the n articles are separated by 0.
    chunked_token_type_ids:
        Tensor of shape (2, batch_size, num_chunks), marking the left and right end token type ids 
        for each chunk. We only need to mark the left and right end token type ids.
    chunk_size: int,
        The size of each chunk.
    window_size: int,
        The size of each window.
    '''
    batch_size, num_tokens = token_type_ids.shape
    num_chunks = num_tokens // chunk_size
    num_windows = num_tokens // window_size
    article_separator = 0
    #### step 2: generate attention masks for each window
    #### NOTE: we perform exact attention within each window, 
    ####       so we only need to mask out different documents
    ####       for each window.
    token_types_windows = token_type_ids.reshape(batch_size, num_windows, window_size, 1)
    token_types_windows_t = token_types_windows.transpose(-1, -2)
    # replace all elements in TOKEN_SEPS with -1
    token_types_windows = torch.where(token_types_windows == article_separator, -1, token_types_windows)
    window_3d_mask = (token_types_windows == token_types_windows_t)
    window_3d_mask = ~window_3d_mask

    chunked_token_type_ids = torch.where(chunked_token_type_ids == article_separator, -1, chunked_token_type_ids)

    token_type_ids_chunks = token_type_ids.unsqueeze(-1) # (batch_size, num_tokens, 1)
    chunk_3d_mask = (
        (token_type_ids_chunks == chunked_token_type_ids[0, :, :].unsqueeze(-2)) | 
        (token_type_ids_chunks == chunked_token_type_ids[1, :, :].unsqueeze(-2))
    )
    chunk_3d_mask = ~chunk_3d_mask

    chunk_causal_mask, window_causal_mask = prepare_eva_attention_mask(
        num_tokens, 
        token_type_ids.device, 
        chunk_size=chunk_size, 
        window_size=window_size,
        use_cache=False,
        cache=None
    )
    window_mask = torch.logical_or(window_causal_mask, window_3d_mask.unsqueeze(1))
    inter_chunk_mask = torch.logical_or(chunk_causal_mask, chunk_3d_mask.unsqueeze(1))

    return window_mask.reshape(batch_size, 1, num_tokens, window_size), inter_chunk_mask, intra_chunk_mask.reshape(batch_size, 1, num_tokens, 1)

def prepare_doc_mask_position_ids_for_test(
    input_ids: torch.LongTensor,
    chunk_size: int,
    window_size: int,
    eos_token_id: int,
    pad_token_id: Optional[int],
):
    token_types, chunked_token_type_ids, intra_chunk_mask = prepare_eva_token_type_ids(
        input_ids,
        chunk_size,
        window_size,
        EOS_TOKEN_TYPE_ID=eos_token_id,
        PAD_TOKEN_TYPE_ID=pad_token_id,
    )
    print("[DEBUG] token_types:", input_ids[0])
    print("[DEBUG] token_types:", token_types[0])
    print("[DEBUG] chunked_token_type_ids:", chunked_token_type_ids[:, 0, :])
    print("[DEBUG] intra_chunk_mask:", intra_chunk_mask[0])
    for i in range(input_ids.shape[1]):
        print("[DEBUG] token id: {}, token: {} || token_type: {}".format(i, input_ids[0][i], token_types[0][i]))
    chunks = input_ids.reshape(input_ids.shape[0], -1, chunk_size)
    num_chunks = chunks.shape[1]
    for i in range(num_chunks):
        print("[DEBUG] chunk id: {}, chunk: {} || intra_chunk_mask: {} || chunked_token_types L: {} R: {}".format(
                i, 
                chunks[0][i], 
                intra_chunk_mask[0][i], 
                chunked_token_type_ids[0][0][i], 
                chunked_token_type_ids[1][0][i]
            )
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

    attention_mask = calculate_3d_attention_mask_gpu(
        token_types.cuda(),
        chunked_token_type_ids.cuda(),
        intra_chunk_mask.cuda(),
        chunk_size,
        window_size,
    )
    attention_mask = tuple(mask.cpu() for mask in attention_mask)
    return attention_mask, position_ids

if __name__ == "__main__":
    from apps.evabyte.evabyte import prepare_doc_mask_position_ids as prepare_doc_mask_position_ids_pt
    torch.set_printoptions(threshold=1024)

    class TestDocumentBoundaryMasks:
        """Test suite for document boundary attention mask implementations."""

        def generate_random_input_ids(
            self,
            batch_size: int,
            seq_len: int,
            vocab_size: int,
            eos_token_id: int,
            pad_token_id: int,
            chunk_size: int,
            min_doc_len: int = 10,
            max_doc_len: int = 50,
            seed: int = None,
            add_padding: bool = False,
        ) -> np.ndarray:
            """
            Generate random input_ids with varied document lengths packed together.
            
            Args:
                batch_size: Number of sequences in batch
                seq_len: Total sequence length
                vocab_size: Size of vocabulary
                eos_token_id: Token ID for end of sequence
                min_doc_len: Minimum document length
                max_doc_len: Maximum document length
                seed: Random seed for reproducibility
                
            Returns:
                input_ids: Shape (batch_size, seq_len) with documents separated by EOS tokens
            """
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
            
            input_ids = np.zeros((batch_size, seq_len), dtype=np.int64)
            
            for b in range(batch_size):
                pos = 0
                while pos < seq_len:
                    pad_len = 0
                    # Generate random document length
                    remaining = seq_len - pos
                    if remaining <= min_doc_len:
                        doc_len = remaining
                    else:
                        doc_len = min(
                            random.randint(min_doc_len, max_doc_len),
                            remaining
                        )
                        if add_padding:
                            if doc_len % chunk_size != 0:
                                pad_len = chunk_size - (doc_len % chunk_size)
                            if doc_len + pad_len >= remaining:
                                pad_len = remaining - doc_len
                            print("[DEBUG] doc_len {} pad_len: {} remaining: {}".format(doc_len, pad_len, remaining))
                    # Generate random tokens for this document (excluding EOS)
                    valid_tokens = list(range(1, vocab_size))
                    if eos_token_id in valid_tokens:
                        valid_tokens.remove(eos_token_id)
                    if pad_token_id in valid_tokens:
                        valid_tokens.remove(pad_token_id)
                    
                    if doc_len == remaining:
                        # last chunk
                        for i in range(doc_len):
                            if pos + i < seq_len:
                                input_ids[b, pos + i] = random.choice(valid_tokens)
                    else:
                        for i in range(doc_len - 1):
                            if pos + i < seq_len:
                                input_ids[b, pos + i] = random.choice(valid_tokens)
                        
                        # Add EOS token if there's space
                        if pos + doc_len - 1 < seq_len:
                            input_ids[b, pos + doc_len - 1] = eos_token_id
                    pos += doc_len
                    if pos < seq_len and pad_len > 0:
                        input_ids[b, pos:pos+pad_len] = pad_token_id
                        pos += pad_len
            return input_ids

        def compare_attention_masks(
            self,
            mask_pt: Union[Tuple, None],
            mask_v2: Union[Tuple, None],
            tolerance: float = 1e-6
        ) -> bool:
            """
            Compare attention masks from PyTorch and NumPy implementations.
            
            Args:
                mask_pt: PyTorch attention mask (tuple or None)
                mask_np: NumPy attention mask (tuple or None)
                tolerance: Numerical tolerance for comparison
                
            Returns:
                True if masks are equivalent
            """
            if mask_pt is None and mask_v2 is None:
                return True
            
            if mask_pt is None or mask_v2 is None:
                return False
            
            if not isinstance(mask_pt, tuple) or not isinstance(mask_v2, tuple):
                return False
            
            if len(mask_pt) != len(mask_v2):
                return False
            
            for i, (pt_mask, v2_mask) in enumerate(zip(mask_pt, mask_v2)):
                pt_array = pt_mask.cpu()
                v2_array = v2_mask.cpu()
                
                if not torch.allclose(pt_array, v2_array, atol=tolerance):
                    print(f"Mask {i} differs:")
                    print(f"PyTorch shape: {pt_array.shape}, NumPy shape: {v2_array.shape}")
                    print(f"Max difference: {torch.max(torch.abs(pt_array.float() - v2_array.float()))}")
                    return False
            
            return True
        
        def compare_position_ids(
            self,
            pos_pt: torch.Tensor,
            pos_v2: torch.Tensor,
            tolerance: float = 1e-6
        ) -> bool:
            """
            Compare position IDs from PyTorch and NumPy implementations.
            
            Args:
                pos_pt: PyTorch position IDs
                pos_np: NumPy position IDs
                tolerance: Numerical tolerance for comparison
                
            Returns:
                True if position IDs are equivalent
            """
            pt_array = pos_pt.cpu()
            v2_array = pos_v2.cpu()
            return torch.allclose(pt_array, v2_array, atol=tolerance)
        
        def test_basic_functionality(self, test_params):
            """Test basic functionality with simple input."""
            batch_size, seq_len = test_params['batch_size'], test_params['seq_len']
            
            # Create simple input with known EOS positions
            input_ids_np = np.random.randint(3, test_params['vocab_size'], (batch_size, seq_len))
            input_ids_np[0, 32] = test_params['eos_token_id']  # EOS at position 32
            input_ids_np[0, 96] = test_params['eos_token_id']  # EOS at position 96
            input_ids_np[1, 64] = test_params['eos_token_id']  # EOS at position 64
            
            input_ids_pt = torch.from_numpy(input_ids_np)
            
            # Test PyTorch implementation
            mask_pt, pos_pt = prepare_doc_mask_position_ids_pt(
                input_ids_pt,
                test_params['chunk_size'],
                test_params['window_size'],
                test_params['eos_token_id'],
                test_params['pad_token_id']
            )
            
            # Test NumPy implementation
            mask_pt_v2, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                input_ids_pt,
                test_params['chunk_size'],
                test_params['window_size'],
                test_params['eos_token_id'],
                test_params['pad_token_id']
            )
            
            # Compare results
            assert self.compare_attention_masks(mask_pt, mask_pt_v2), "Attention masks don't match"
            assert self.compare_position_ids(pos_pt, pos_pt_v2), "Position IDs don't match"
        
        def test_numpy_pt_impl(self, test_params):
            """Test with randomly generated documents of varied lengths."""
            batch_size, seq_len = test_params['batch_size'], test_params['seq_len']
            
            for seed in [42, 123, 999]:
                input_ids_np = self.generate_random_input_ids(
                    batch_size, seq_len, test_params['vocab_size'],
                    test_params['eos_token_id'], test_params['pad_token_id'], chunk_size=test_params["chunk_size"],
                    seed=seed, add_padding=True
                )
                input_ids_pt = torch.from_numpy(input_ids_np)
                
                # Test PyTorch implementation
                token_types_pt, chunked_types_pt, intra_chunk_mask_pt, pos_pt = prepare_token_types_position_ids(
                    input_ids_pt,
                    test_params['chunk_size'],
                    test_params['window_size'],
                    test_params['eos_token_id'],
                    test_params['pad_token_id']
                )
                
                # Test NumPy implementation
                token_types_np, chunked_types_np, intra_chunk_mask_np, pos_np = prepare_token_types_position_ids_numpy(
                    input_ids_np,
                    test_params['chunk_size'],
                    test_params['window_size'],
                    test_params['eos_token_id'],
                    test_params['pad_token_id']
                )
                
                # Compare results
                assert np.allclose(token_types_pt.numpy(), token_types_np), f"Token types don't match for seed {seed}: {token_types_pt} vs {token_types_np}"
                assert np.allclose(chunked_types_pt.numpy(), chunked_types_np), f"Chunked types don't match for seed {seed}: {chunked_types_pt} vs {chunked_types_np}"
                assert np.allclose(intra_chunk_mask_pt.numpy(), intra_chunk_mask_np), f"Intra chunk masks don't match for seed {seed}: {intra_chunk_mask_pt} vs {intra_chunk_mask_np}"
                assert np.allclose(pos_pt.numpy(), pos_np), f"Position ids don't match for seed {seed}: {pos_pt} vs {pos_np}"

        def test_random_documents(self, test_params):
            """Test with randomly generated documents of varied lengths."""
            batch_size, seq_len = test_params['batch_size'], test_params['seq_len']
            
            for seed in [42, 123, 999]:
                input_ids_np = self.generate_random_input_ids(
                    batch_size, seq_len, test_params['vocab_size'],
                    test_params['eos_token_id'], test_params['pad_token_id'], chunk_size=test_params["chunk_size"],
                    seed=seed, add_padding=True
                )
                input_ids_pt = torch.from_numpy(input_ids_np)
                
                # Test PyTorch implementation
                mask_pt, pos_pt = prepare_doc_mask_position_ids_pt(
                    input_ids_pt,
                    test_params['chunk_size'],
                    test_params['window_size'],
                    test_params['eos_token_id'],
                    test_params['pad_token_id']
                )
                
                # Test NumPy implementation
                mask_pt_v2, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                    input_ids_pt,
                    test_params['chunk_size'],
                    test_params['window_size'],
                    test_params['eos_token_id'],
                    test_params['pad_token_id']
                )
                
                # Compare results
                assert self.compare_attention_masks(mask_pt, mask_pt_v2), f"Attention masks don't match for seed {seed}"
                assert self.compare_position_ids(pos_pt, pos_pt_v2), f"Position IDs don't match for seed {seed}"
        
        def test_edge_cases(self, test_params):
            """Test edge cases."""
            
            # Test 1: No EOS tokens
            batch_size, seq_len = test_params['batch_size'], test_params['seq_len']
            input_ids_np = np.random.randint(3, test_params['vocab_size'], (batch_size, seq_len))
            input_ids_pt = torch.from_numpy(input_ids_np)
            
            mask_pt, pos_pt = prepare_doc_mask_position_ids_pt(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            mask_pt_v2, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            
            assert self.compare_attention_masks(mask_pt, mask_pt_v2), "No EOS case: Masks don't match"
            assert self.compare_position_ids(pos_pt, pos_pt_v2), "No EOS case: Position IDs don't match"
            
            # Test 2: Only EOS tokens
            input_ids_np = np.full((batch_size, seq_len), test_params['eos_token_id'])
            input_ids_pt = torch.from_numpy(input_ids_np)
            
            mask_pt, pos_pt = prepare_doc_mask_position_ids_pt(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            mask_pt_v2, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            
            assert self.compare_attention_masks(mask_pt, mask_pt_v2), "All EOS case: Masks don't match"
            assert self.compare_position_ids(pos_pt, pos_pt_v2), "All EOS case: Position IDs don't match"
            
            # Test 3: EOS at first position
            input_ids_np = np.random.randint(3, test_params['vocab_size'], (batch_size, seq_len))
            input_ids_np[:, 0] = test_params['eos_token_id']
            input_ids_pt = torch.from_numpy(input_ids_np)
            
            mask_pt, pos_pt = prepare_doc_mask_position_ids_pt(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            mask_pt_v2, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            
            assert self.compare_attention_masks(mask_pt, mask_pt_v2), "EOS at start: Masks don't match"
            assert self.compare_position_ids(pos_pt, pos_pt_v2), "EOS at start: Position IDs don't match"
            
            # Test 4: EOS at last position
            input_ids_np = np.random.randint(3, test_params['vocab_size'], (batch_size, seq_len))
            input_ids_np[:, -1] = test_params['eos_token_id']
            input_ids_pt = torch.from_numpy(input_ids_np)
            
            mask_pt, pos_pt = prepare_doc_mask_position_ids_pt(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            mask_pt_v2, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            
            assert self.compare_attention_masks(mask_pt, mask_pt_v2), "EOS at end: Masks don't match"
            assert self.compare_position_ids(pos_pt, pos_pt_v2), "EOS at end: Position IDs don't match"
        
        def test_different_sizes(self, test_params):
            """Test with different sequence lengths and batch sizes."""
            
            test_configs = [
                (16, 512),  # Large batch
                (2, 1024),  # Long sequence
                (1, 4096),  # Very long sequence
                (1, 16384),  # Very long sequence
                (1, 32768),  # Very long sequence
                (1, 65536),  # Very long sequence
            ]
            
            for batch_size, seq_len in test_configs:
                # Skip if sequence length doesn't match window size requirements
                if seq_len % test_params['window_size'] != 0:
                    continue
                    
                input_ids_np = self.generate_random_input_ids(
                    batch_size, seq_len, test_params['vocab_size'],
                    test_params['eos_token_id'], test_params['pad_token_id'], chunk_size=test_params["chunk_size"], seed=42
                )
                input_ids_pt = torch.from_numpy(input_ids_np)
                
                mask_pt, pos_pt = prepare_doc_mask_position_ids_pt(
                    input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
                )
                mask_pt_v2, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                    input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
                )
                
                assert self.compare_attention_masks(mask_pt, mask_pt_v2), f"Size {batch_size}x{seq_len}: Masks don't match"
                assert self.compare_position_ids(pos_pt, pos_pt_v2), f"Size {batch_size}x{seq_len}: Position IDs don't match"
        
        def test_position_id_correctness(self, test_params):
            """Test that position IDs are correctly reset at document boundaries."""
            batch_size, seq_len = 1, test_params['seq_len']
            
            # Create input with known EOS positions
            input_ids_np = np.full((batch_size, seq_len), 5)  # Fill with token 5
            doc_lens = [16, 32, 22, 40]
            assert sum(doc_lens) < seq_len, "Sum of doc lengths must be less than seq_len"
            doc_lens.append(seq_len - sum(doc_lens))

            prev_index = 0
            for doc_len in doc_lens[:-1]:
                input_ids_np[0, prev_index + doc_len - 1] = test_params['eos_token_id']
                prev_index += doc_len
            
            input_ids_pt = torch.from_numpy(input_ids_np)
            # Test NumPy implementation
            _, pos_pt_v2 = prepare_doc_mask_position_ids_for_test(
                input_ids_pt, test_params['chunk_size'], test_params['window_size'], test_params['eos_token_id'], test_params['pad_token_id']
            )
            
            prev_index = 0
            # Check position resets
            for i in range(len(doc_lens)):
                assert torch.all(pos_pt_v2[0, prev_index:prev_index + doc_lens[i]] == torch.arange(doc_lens[i])), (
                    f"Document {i} positions incorrect"
                    f"pos_pt_v2: {pos_pt_v2[0, prev_index:prev_index + doc_lens[i]]}"
                    f"expected: {torch.arange(doc_lens[i])}"
                )
                prev_index += doc_lens[i]

        def test_multibyte_loss_weight_numpy(self):
            """Simple parity tests for prepare_multibyte_loss_weight vs NumPy version."""
            num_heads = 8
            vocab_size = 1024
            raw_byte_offset = 64
            bos_id = 0
            raw_sentinel_id_start = 1
            compressed_sentinel_id_start = 2


            def generate_input_ids_and_labels(inputs):
                temp = np.lib.stride_tricks.sliding_window_view(inputs, num_heads + 1, axis=0)
                input_ids_np = temp[None, :, 0]
                labels_np = temp[None, :, 1:]
                return input_ids_np, labels_np

            # Case 1: all raw labels, no specials -> expect weights all ones (no normalization due to impl semantics)
            input_ids_np_1, labels_np_1 = generate_input_ids_and_labels(
                [0, 1, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
            )
            torch_out = prepare_multibyte_loss_weight(
                torch.from_numpy(input_ids_np_1),
                torch.from_numpy(labels_np_1),
                num_heads,
                vocab_size,
                raw_byte_offset,
                bos_id,
                raw_sentinel_id_start,
                compressed_sentinel_id_start,
            ).numpy()

            np_out = prepare_multibyte_loss_weight_numpy(
                input_ids_np_1,
                labels_np_1,
                num_heads,
                vocab_size,
                raw_byte_offset,
                bos_id,
                raw_sentinel_id_start,
                compressed_sentinel_id_start,
            )
            print("========= CASE 1 ==========")
            print("========= input_ids_np_1 ==========")
            print(input_ids_np_1)
            print("========= labels_np_1 ==========")
            print(labels_np_1)
            print("========= torch_out ==========")
            print(torch_out)
            print("========= np_out ==========")
            print(np_out)
            assert np.allclose(torch_out, np_out)

            # Case 2: introduce a non-raw label (> raw_byte_offset) to trigger coeff
            input_ids_np_2, labels_np_2  = generate_input_ids_and_labels(
                [0, 1, 32, 33, 34, 35, 36, 37, 2, 39, 40, 41, 0, 1, 43, 44, 45, 46, 47, 48, 49, 50]
            )
            torch_out2 = prepare_multibyte_loss_weight(
                torch.from_numpy(input_ids_np_2),
                torch.from_numpy(labels_np_2),
                num_heads,
                vocab_size,
                raw_byte_offset,
                bos_id,
                raw_sentinel_id_start,
                compressed_sentinel_id_start,
            ).numpy()
            np_out2 = prepare_multibyte_loss_weight_numpy(
                input_ids_np_2,
                labels_np_2,
                num_heads,
                vocab_size,
                raw_byte_offset,
                bos_id,
                raw_sentinel_id_start,
                compressed_sentinel_id_start,
            )
            print("========= CASE 2 ==========")
            print("========= input_ids_np_2 ==========")
            print(input_ids_np_2)
            print("========= labels_np_2 ==========")
            print(labels_np_2)
            print("========= torch_out ==========")
            print(torch_out2)
            print("========= np_out ==========")
            print(np_out2)
            assert np.allclose(torch_out2, np_out2)


            for disable_cross_byte_prediction in [True, False]:
                for weighting_compressed_prediction in [True, False]:
                    # Case 3: add a special token at head 1 -> only heads before first special kept and normalized across kept heads
                    input_ids_np_3, labels_np_3  = generate_input_ids_and_labels(
                        [0, 1, 32, 33, 34, 35, 80, 90, 100, 110, 44, 45, 46, 47, 1, 2, 2, 2, 2, 2, 2]
                    )
                    torch_out3 = prepare_multibyte_loss_weight(
                        torch.from_numpy(input_ids_np_3),
                        torch.from_numpy(labels_np_3),
                        num_heads,
                        vocab_size,
                        raw_byte_offset,
                        bos_id,
                        raw_sentinel_id_start,
                        compressed_sentinel_id_start,
                        weighting_compressed_prediction=weighting_compressed_prediction,
                        disable_cross_byte_prediction=disable_cross_byte_prediction,
                    ).numpy()
                    np_out3 = prepare_multibyte_loss_weight_numpy(
                        input_ids_np_3,
                        labels_np_3,
                        num_heads,
                        vocab_size,
                        raw_byte_offset,
                        bos_id,
                        raw_sentinel_id_start,
                        compressed_sentinel_id_start,
                        weighting_compressed_prediction=weighting_compressed_prediction,
                        disable_cross_byte_prediction=disable_cross_byte_prediction,
                    )
                    print("========= CASE 3 ==========")
                    print(f"========= input_ids_np_3 with disable_cross_byte_prediction={disable_cross_byte_prediction} and weighting_compressed_prediction={weighting_compressed_prediction} ==========")
                    print(input_ids_np_3)
                    print("========= labels_np_3 ==========")
                    print(labels_np_3)
                    print("========= torch_out ==========")
                    print(torch_out3)
                    print("========= np_out ==========")
                    print(np_out3)
                    assert np.allclose(torch_out3, np_out3)


                    # Case 4: general case
                    input_ids_np_4, labels_np_4  = generate_input_ids_and_labels(
                        [0, 1, 32, 33, 34, 80, 90, 100, 110, 44, 45, 46, 47, 80, 80, 80, 90, 100, 110, 3, 2, 21, 22, 23, 24, 25, 26, 27, 28, 29, 4, 5]
                    )
                    torch_out4 = prepare_multibyte_loss_weight(
                        torch.from_numpy(input_ids_np_4),
                        torch.from_numpy(labels_np_4),
                        num_heads,
                        vocab_size,
                        raw_byte_offset,
                        bos_id,
                        raw_sentinel_id_start,
                        compressed_sentinel_id_start,
                        weighting_compressed_prediction=weighting_compressed_prediction,
                        disable_cross_byte_prediction=disable_cross_byte_prediction,
                    ).numpy()
                    np_out4 = prepare_multibyte_loss_weight_numpy(
                        input_ids_np_4,
                        labels_np_4,
                        num_heads,
                        vocab_size,
                        raw_byte_offset,
                        bos_id,
                        raw_sentinel_id_start,
                        compressed_sentinel_id_start,
                        weighting_compressed_prediction=weighting_compressed_prediction,
                        disable_cross_byte_prediction=disable_cross_byte_prediction,
                    )
                    print("========= CASE 4 ==========")
                    print(f"========= input_ids_np_4 with disable_cross_byte_prediction={disable_cross_byte_prediction} and weighting_compressed_prediction={weighting_compressed_prediction} ==========")
                    print(input_ids_np_4)
                    print("========= labels_np_4 ==========")
                    print(labels_np_4)
                    print("========= torch_out ==========")
                    print(torch_out4)
                    print("========= np_out ==========")
                    print(np_out4)
                    assert np.allclose(torch_out4, np_out4)


    def run_tests():
        """Run all tests manually if pytest is not available."""
        test_instance = TestDocumentBoundaryMasks()
        test_params = {
            'chunk_size': 8,
            'window_size': 32,
            'eos_token_id': 1001,
            'pad_token_id': 1002,
            'vocab_size': 1000,
            'batch_size': 2,
            'seq_len': 128,
        }
        
        print("Running document boundary mask comparison tests...")
        
        try:
            # Lightweight parity test for multibyte loss weight
            test_instance.test_multibyte_loss_weight_numpy()
            print("✓ Tested multibyte loss weight NumPy parity...")

            test_instance.test_numpy_pt_impl(test_params)
            print("✓ Tested NumPy vs PyTorch implementation...")

            test_instance.test_basic_functionality(test_params)
            print("✓ Tested basic functionality...")
            
            test_instance.test_random_documents(test_params)
            print("✓ Tested random documents...")
            
            test_instance.test_edge_cases(test_params)
            print("✓ Tested edge cases...")
            
            test_instance.test_different_sizes(test_params)
            print("✓ Tested different sizes...")
            
            test_instance.test_position_id_correctness(test_params)
            print("✓ Tested position ID correctness...")
            
            print("\nAll tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise

    run_tests()
