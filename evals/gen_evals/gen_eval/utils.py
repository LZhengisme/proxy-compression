import torch
from transformers import (
    StoppingCriteria, 
    LogitsProcessor
)
from typing import List, Union, Literal
from collections import namedtuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from lingua.tokenizer import (
    SimpleBPEByteTokenizer, 
    SimpleBPEDoubleByteTokenizer,
    SimpleTokenIDPlusByteTokenizer,
    BytePairEncodingTokenizer,
    HalfByteCodec,
    BitCodec,
    DoubleBitCodec,
    DoubleByteCodec
)
import re
from transformers import AutoTokenizer

tokenizer_call_output = namedtuple(
    "tokenizer_call_output",
    ["input_ids", "attention_mask"],
)


class Trie:
    def __init__(self, *args, **kwargs):
        self.root = {}
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        """Update the trie with key-value pairs."""
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __setitem__(self, key, value):
        """Set the value at the node for the given key."""
        crawler = self.root
        for char in key:
            crawler = crawler.setdefault(char, {})
        crawler['value'] = value

    def extensions(self, prefix: str) -> list:
        """Retrieve values starting with a given prefix."""
        crawler = self.root
        for char in prefix:
            if char not in crawler:
                break
            crawler = crawler[char]
        return self._collect_values(crawler)

    def _collect_values(self, node: dict) -> list:
        """Recursively collect all values under the given node."""
        values = [node['value']] if 'value' in node else []
        for child in node.values():
            if isinstance(child, dict):
                values.extend(self._collect_values(child))
        return values

@torch.inference_mode()
def heal_tokens(prompt, model, tokenizer, spm_tokenizer, device):
    bos_token_id, pad_token_id = tokenizer.bos_token_id, tokenizer.pad_token_id
    vocab_trie = Trie(spm_tokenizer.get_vocab())

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    ).input_ids[0].to(device)

    """
    the latter code assumes the input_ids is not empty,
    input_id has to be checked if contains elements
    """
    if input_ids.numel() == 0:
        return input_ids

    tail_id = input_ids[-tokenizer.sp_byte_length:].tolist()
    tail_tok = tokenizer.decode(tail_id)

    # space_id = tokenizer.token_to_id(" ")
    # if space_id is not None:
    #     space_tok = tokenizer.id_to_token(space_id)[0]
    #     # tail tokens are used for a prefix search, thus, whitespaces are replaced with
    #     # their tokenization (e.g. 'Ġ') to enable search for tokens prefixed with a whitespace
    #     tail_tok = tail_tok.replace(" ", space_tok)

    if torch.all(input_ids == pad_token_id).item():
        return prompt  # skip empty sequences (all pad ids)

    # apply bias for alternatives (extensions) to the tail token
    # seq_bias = {
    #     (self.tokenizer.convert_tokens_to_ids(alt_tok),): 10.0 for alt_tok in vocab_trie.extensions(prefix=tail_tok)
    # }
    # seq_bias = {
    #     (alt_tok,): 10.0 for alt_tok in vocab_trie.extensions(prefix=tail_tok)
    # }
    seq_biases = [{} for _ in range(tokenizer.sp_byte_length)]
    for alt_tok in vocab_trie.extensions(prefix=tail_tok):
        token_bytes = alt_tok.to_bytes(tokenizer.sp_byte_length, byteorder='big')

        for i in range(tokenizer.sp_byte_length):
            token_val = int(token_bytes[i]) + tokenizer.offset
            seq_biases[i][(token_val,)] = 10.0
    
    for i in range(tokenizer.sp_byte_length):
        if (tail_id[i],) not in seq_biases[i]:
            seq_biases[i][(tail_id[i],)] = 10.0

        seq_biases[i][(tail_id[i],)] += 1.0

    if len(seq_biases[0]) == 1:
        return prompt  # skip if there are no token alternatives to heal with
    # slightly favor original token to limit aggressive healing e.g. 'http' -> 'https'

    trimmed_ids = input_ids[:-tokenizer.sp_byte_length]

    """
    the latter code assumes trimmed_ids is not empty
    so have to check the its element count
    """
    if trimmed_ids.numel() == 0:
        return prompt

    input_ids = trimmed_ids.unsqueeze(0)
    for i in range(tokenizer.sp_byte_length):
        seq_bias = seq_biases[i]
        healed_input_ids = model.generate(
            input_ids, 
            sequence_bias=seq_bias,
            max_new_tokens=1,
        )
        input_ids = healed_input_ids
    healed_prompt = tokenizer.decode(healed_input_ids.squeeze())
    return healed_prompt

class TokenizerWrapper:
    """Wrapper class for tokenizer to support batched inputs."""
    def __init__(
            self, 
            tokenizer,
            tokenizer_path, 
            tokenizer_mode, 
            spm_path: str, 
            byte_conversion_args: dict,
            separate_embedding: bool,
        ):
        if byte_conversion_args is not None:
            assert "byte_converter_type" in byte_conversion_args
        self.tokenizer = tokenizer
        assert self.tokenizer.bos_token_id is not None
        assert self.tokenizer.eos_token_id is not None
        assert hasattr(self.tokenizer, 'offset')
        self.n_words = len(self.tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token
        self.name_or_path = self.tokenizer.name_or_path
        self.offset = self.tokenizer.offset

        if separate_embedding:
            self.compressed_offset = self.offset + 256
        else:
            self.compressed_offset = None

        self.raw_sentinel_id_start = self.tokenizer.convert_tokens_to_ids("<extra_id_12>")
        self.raw_sentinel_id_end = self.tokenizer.convert_tokens_to_ids("<extra_id_13>")
        self.compressed_sentinel_id_start = self.tokenizer.convert_tokens_to_ids("<extra_id_14>")
        self.compressed_sentinel_id_end = self.tokenizer.convert_tokens_to_ids("<extra_id_15>")

        self.byte_scrambler = None
        self.lingua_tokenizer = None
        self.halfbyte_coder = None
        self.doublebyte_coder = None
        self.bit_coder = None
        self.doublebit_coder = None
        self.subbyte_coder = None
        if tokenizer_mode == "spm_byte":
            self.lingua_tokenizer = SimpleBPEByteTokenizer(tokenizer_path, spm_path, byte_conversion_args, separate_embedding)
            self.sp_byte_length = self.lingua_tokenizer.sp_byte_length
        elif tokenizer_mode == "token_plus_byte":
            self.lingua_tokenizer = SimpleTokenIDPlusByteTokenizer(tokenizer_path, spm_path, separate_embedding)
        elif tokenizer_mode == "spm_doublebyte":
            self.lingua_tokenizer = SimpleBPEDoubleByteTokenizer(tokenizer_path, spm_path, byte_conversion_args, separate_embedding)
            self.sp_byte_length = self.lingua_tokenizer.sp_byte_length
        elif tokenizer_mode == "halfbyte":
            self.halfbyte_coder = HalfByteCodec()
        elif tokenizer_mode == "doublebyte":
            self.doublebyte_coder = DoubleByteCodec()
        elif tokenizer_mode == "doublebit":
            self.doublebit_coder = DoubleBitCodec()
        elif tokenizer_mode == "bit":
            self.bit_coder = BitCodec()
        elif tokenizer_mode == "hf_tokenizer":
            # use HF tokenizers
            self.regular_tokenizer = AutoTokenizer.from_pretrained(spm_path, trust_remote_code=True)
        elif tokenizer_mode == "hf_spm":
            # use (usually locally trained) HF tokenizers
            self.regular_tokenizer = BytePairEncodingTokenizer(spm_path)
            self.bos_token_id = self.regular_tokenizer.bos_id
            self.eos_token_id = self.regular_tokenizer.eos_id
            self.pad_token_id = self.regular_tokenizer.pad_id
            self.eos_token = "<eos>"
        else:
            self.byte_scrambler = None
            self.lingua_tokenizer = None

        self.tokenizer_mode = tokenizer_mode

    def encode_with_spm(self, s: str):
        return self.lingua_tokenizer.encode_with_spm(s)

    def decode_with_spm(self, byte_list: list):
        return self.lingua_tokenizer.decode_with_spm(byte_list)

    def encode_bytes_with_sentinels(
            self, s: bytes, 
            bytes_type: Literal["raw", "compressed"], 
            add_bos: bool = False, 
            add_eos: bool = False,
            add_sentinel_start: bool = True,
            add_sentinel_end: bool = True,
        ):
        if bytes_type == "raw":
            sentinel_id_start = self.raw_sentinel_id_start
            sentinel_id_end = self.raw_sentinel_id_end
            _offset = self.offset
        elif bytes_type == "compressed":
            sentinel_id_start = self.compressed_sentinel_id_start
            sentinel_id_end = self.compressed_sentinel_id_end
            if self.compressed_offset:
                _offset = self.compressed_offset
            else:
                _offset = self.offset
        else:
            raise ValueError(f"Unknown bytes_type: {bytes_type}")
        tokens = (
            [self.bos_token_id] * add_bos +
            [sentinel_id_start] * add_sentinel_start +
            [b + _offset for b in s] +
            [sentinel_id_end] * add_sentinel_end +
            [self.eos_token_id] * add_eos
        )
        return tokens

    def __call__(self, text, **kwargs):
        """Tokenize text and return token ids."""
        if self.tokenizer_mode == "raw_sentinel":
            compressed_bytes = text.encode("utf-8")
            token_ids = self.encode_bytes_with_sentinels(compressed_bytes, bytes_type="raw", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode == "compressed_sentinel":
            compressed_bytes = text.encode("utf-8")
            token_ids = self.encode_bytes_with_sentinels(compressed_bytes, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode in ["spm_byte", "spm_doublebyte", "token_plus_byte"]:
            spm_bytes = self.encode_with_spm(text)
            token_ids = self.encode_bytes_with_sentinels(spm_bytes, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode.startswith("scramble"):
            scrambled_bytes = self.byte_scrambler.encode(text.encode("utf-8"))
            token_ids = self.encode_bytes_with_sentinels(scrambled_bytes, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode == "halfbyte":
            halfbyte_bytes = self.halfbyte_coder.encode(text.encode("utf-8"))
            token_ids = self.encode_bytes_with_sentinels(halfbyte_bytes, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode == "subbyte":
            subbyte_bytes = self.subbyte_coder.encode(text.encode("utf-8"))
            token_ids = self.encode_bytes_with_sentinels(subbyte_bytes, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode == "doublebit":
            doublebits = self.doublebit_coder.encode(text.encode("utf-8"))
            token_ids = self.encode_bytes_with_sentinels(doublebits, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode == "bit":
            bits = self.bit_coder.encode(text.encode("utf-8"))
            token_ids = self.encode_bytes_with_sentinels(bits, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode == "doublebyte":
            doublebyte_bytes = self.doublebyte_coder.encode(text.encode("utf-8"))
            token_ids = self.encode_bytes_with_sentinels(doublebyte_bytes, bytes_type="compressed", add_bos=True, add_sentinel_end=False)
        elif self.tokenizer_mode == "default":
            token_ids = self.tokenizer(text, **kwargs)["input_ids"][0].tolist()
        elif self.tokenizer_mode == "hf_tokenizer":
            token_ids = self.regular_tokenizer(text, **kwargs)["input_ids"][0].tolist()
        elif self.tokenizer_mode == "hf_spm":
            token_ids = self.regular_tokenizer.encode(text, add_bos=True, add_eos=False)
        else:
            raise ValueError(f"Unknown tokenizer mode: {self.tokenizer_mode}")

        token_ids = torch.tensor([token_ids], dtype=torch.long)
        return tokenizer_call_output(token_ids, torch.ones_like(token_ids, dtype=torch.bool))

    def _preprocess_compressed_tokens(self, token_ids):
        if token_ids[0] == self.compressed_sentinel_id_start:
            token_ids = token_ids[1:]

        if self.compressed_offset:
            _offset = self.compressed_offset
        else:
            _offset = self.offset
        token_ids = [token - _offset for token in token_ids if token != self.compressed_sentinel_id_end]
        return token_ids

    def decode(self, token_ids, **kwargs):
        if token_ids[0] == self.bos_token_id:
            token_ids = token_ids[1:]
        if token_ids[-1] == self.eos_token_id:
            token_ids = token_ids[:-1]
        if len(token_ids) == 0:
            return ""
        if self.tokenizer_mode == "raw_sentinel":
            if token_ids[0] == self.raw_sentinel_id_start:
                token_ids = token_ids[1:]
            
            token_ids = [
                token for token in token_ids 
                if (token != self.raw_sentinel_id_end) and (token < 256 + self.offset)
            ]
        elif self.tokenizer_mode == "compressed_sentinel":
            token_ids = self._preprocess_compressed_tokens(token_ids)
        elif self.tokenizer_mode in ["spm_byte", "spm_doublebyte", "token_plus_byte"]:
            token_ids = self._preprocess_compressed_tokens(token_ids)
            return self.decode_with_spm(token_ids)
        elif self.tokenizer_mode.startswith("scramble"):
            token_ids = self._preprocess_compressed_tokens(token_ids)
            utf8_bytes = self.byte_scrambler.decode(token_ids)
            return utf8_bytes.decode("utf-8", errors="ignore")
        elif self.tokenizer_mode == "halfbyte":
            token_ids = self._preprocess_compressed_tokens(token_ids)
            utf8_bytes = self.halfbyte_coder.decode(token_ids)
            return utf8_bytes.decode("utf-8", errors="ignore")
        elif self.tokenizer_mode == "subbyte":
            token_ids = self._preprocess_compressed_tokens(token_ids)
            utf8_bytes = self.subbyte_coder.decode(token_ids)
            return utf8_bytes.decode("utf-8", errors="ignore")
        elif self.tokenizer_mode == "doublebit":
            token_ids = self._preprocess_compressed_tokens(token_ids)
            utf8_bytes = self.doublebit_coder.decode(token_ids)
            return utf8_bytes.decode("utf-8", errors="ignore")
        elif self.tokenizer_mode == "bit":
            token_ids = self._preprocess_compressed_tokens(token_ids)
            utf8_bytes = self.bit_coder.decode(token_ids)
            return utf8_bytes.decode("utf-8", errors="ignore")
        elif self.tokenizer_mode == "doublebyte":
            token_ids = self._preprocess_compressed_tokens(token_ids)
            utf8_bytes = self.doublebyte_coder.decode(token_ids)
            return utf8_bytes.decode("utf-8", errors="ignore")
        elif self.tokenizer_mode == "default":
            pass
        elif self.tokenizer_mode == "hf_spm":
            return self.regular_tokenizer.decode(token_ids)
        elif self.tokenizer_mode == "hf_tokenizer":
            return self.regular_tokenizer.decode(token_ids, add_bos=False, add_eos=False)
        else:
            raise ValueError(f"Unknown tokenizer mode: {self.tokenizer_mode}")

        # tokenizer.decode will do the offset stuff
        return self.tokenizer.decode(
            token_ids, **kwargs
        )

    def batch_decode(self, byte_tensor: torch.Tensor, **kwargs):
        strs = []
        for bytes in byte_tensor.tolist():
            strs.append(self.decode(bytes, **kwargs))
        return strs

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)

class StopWordsCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if a stop token is met."""

    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        should_stop = False
        for stop_token in self.stop_tokens:
            should_stop = torch.any(input_ids[:, -1:] == stop_token)
        return should_stop

class UTF8ByteOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, utf8_byte_offset: int):
        self.utf8_byte_offset = utf8_byte_offset

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        byte_value_mask = torch.arange(scores.shape[-1], device=scores.device) < self.utf8_byte_offset
        scores = torch.where(byte_value_mask, scores, -float("inf"))
        return scores
