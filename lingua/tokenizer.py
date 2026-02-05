# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, ByteString, Dict
import logging
import os
import math
from lingua.byte_converter import (
    SimpleTokenBytes, 
    GrayTokenBytes
)
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

@dataclass
class ByteConverterConfig:
    byte_converter_type: str = "simple"

# @dataclass
# class ByteTransformConfig:

@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None
    spm_byte_path: Optional[str] = None
    separate_embedding: bool = False
    byte_converter_config: ByteConverterConfig = field(default_factory=ByteConverterConfig)

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

class BytePairEncodingTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        from tokenizers import Tokenizer as HF_Tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_tokenizer = HF_Tokenizer.from_file(model_path)
        self.sp_tokenizer.add_special_tokens(
            [
                "<bos>", 
                "<eos>", 
                "<pad>",
                "<unk>",
                "<mask>",
                "<sep>",
            ]
        )
        # BOS / EOS token IDs
        self.bos_id: int = self.sp_tokenizer.token_to_id("<bos>")
        self.eos_id: int = self.sp_tokenizer.token_to_id("<eos>")
        self.pad_id: int = self.sp_tokenizer.token_to_id("<pad>")
        self.n_words: int = self.sp_tokenizer.get_vocab_size(with_added_tokens=True)
        print("n_words: ", self.n_words, flush=True)
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        tokens = (
            [self.bos_id] * add_bos + self.sp_tokenizer.encode(s).ids + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int]):
        return self.sp_tokenizer.decode(tokens)

class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 0
        self.eos_id = 255
        self.n_words = 256

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        tokens = [t for t in tokens if t != self.bos_id and t != self.eos_id]
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int]):
        return self.sp_model.decode(tokens)

class VanillaHFTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        assert self.tokenizer.bos_token_id is not None
        assert self.tokenizer.eos_token_id is not None
        self.n_words = len(self.tokenizer)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

    def _strip_encode(self, string):
        tok = self.tokenizer.encode(string)
        if (
            tok[0] == self.bos_id
            and len(tok) >= 1
        ):
            tok = tok[1:]
        # If there is an EOS token at the end of completion then remove it
        if (
            tok[-1] == self.eos_id
            and len(tok) >= 1
        ):
            tok = tok[:-1]
        return tok

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + self._strip_encode(s) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)

class HFTokenizer(Tokenizer):
    def __init__(self, model_path: str, separate_embedding: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        assert self.tokenizer.bos_token_id is not None
        assert self.tokenizer.eos_token_id is not None
        assert hasattr(self.tokenizer, 'offset')
        self.n_words = len(self.tokenizer)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.offset = self.tokenizer.offset

        self.raw_sentinel_id_start = self.tokenizer.convert_tokens_to_ids("<extra_id_12>")
        self.raw_sentinel_id_end = self.tokenizer.convert_tokens_to_ids("<extra_id_13>")
        self.compressed_sentinel_id_start = self.tokenizer.convert_tokens_to_ids("<extra_id_14>")
        self.compressed_sentinel_id_end = self.tokenizer.convert_tokens_to_ids("<extra_id_15>")

        self.separate_embedding = separate_embedding
        if self.separate_embedding:
            # when separating the embedding between compressed and raw bytes,
            # we offset compressed embedding indices by 256
            self.n_words = self.n_words + 256
            self.compressed_offset = self.offset + 256
        else:
            self.compressed_offset = None

    def _strip_encode(self, string):
        tok = self.tokenizer.encode(string)
        if (
            tok[0] == self.bos_id
            and len(tok) >= 1
        ):
            tok = tok[1:]
        # If there is an EOS token at the end of completion then remove it
        if (
            tok[-1] == self.eos_id
            and len(tok) >= 1
        ):
            tok = tok[:-1]
        return tok

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + self._strip_encode(s) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)
   
    def encode_bytes(self, s: bytes, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + [b + self.offset for b in s] + [self.eos_id] * add_eos
        return tokens
    
    def encode_bytes_with_sentinels(self, s: bytes, bytes_type: Literal["raw", "compressed"], add_bos: bool = False, add_eos: bool = False):
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
            [self.bos_id] * add_bos +
            [sentinel_id_start] +
            [b + _offset for b in s] +
            [sentinel_id_end] +
            [self.eos_id] * add_eos
        )
        return tokens

    def decode_bytes(self, tokens: List[int]):
        return bytes([b - self.offset for b in tokens])

class SimpleTokenIDPlusByteTokenizer(HFTokenizer):
    def __init__(
        self, 
        model_path: str, 
        spm_path: str, 
        separate_embedding: bool,
    ) -> None:
        super().__init__(model_path, separate_embedding)
        self.token_tokenizer = AutoTokenizer.from_pretrained(spm_path, trust_remote_code=True)
        self.token_bos_id = self.tokenizer.bos_token_id
        self.token_eos_id = self.tokenizer.eos_token_id

    def encode_with_spm(self, s: str):
        token_ids = self.token_tokenizer.encode(s)
        if (
            token_ids[0] == self.token_bos_id
            and len(token_ids) >= 1
        ):
            token_ids = token_ids[1:]
        if (
            token_ids[-1] == self.token_eos_id
            and len(token_ids) >= 1
        ):
            token_ids = token_ids[:-1]
        return token_ids

    def decode_with_spm(self, byte_list: list):
        return self.token_tokenizer.decode(byte_list)

class DoubleByteTokenizer(HFTokenizer):
    def __init__(
        self,
        model_path: str,
        separate_embedding: bool,
    ) -> None:
        super().__init__(model_path, separate_embedding)
        self.n_words = 65536 + self.offset
        if self.separate_embedding:
            # when separating the embedding between compressed and raw bytes,
            # we offset compressed embedding indices by 256
            self.n_words = self.n_words + 256
            self.compressed_offset = self.offset + 256
        else:
            self.compressed_offset = None

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        raise NotImplementedError

    def decode(self, tokens: List[int]):
        raise NotImplementedError
   
    def encode_bytes(self, s: bytes, add_bos: bool = False, add_eos: bool = False):
        raise NotImplementedError
    
    def encode_bytes_with_sentinels(self, s: List[int], bytes_type: Literal["raw", "compressed"], add_bos: bool = False, add_eos: bool = False):
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
            [self.bos_id] * add_bos +
            [sentinel_id_start] +
            [b + _offset for b in s] +
            [sentinel_id_end] +
            [self.eos_id] * add_eos
        )
        return tokens

    def decode_bytes(self, tokens: List[int]):
        raise NotImplementedError

class SimpleBPEByteTokenizer(HFTokenizer):
    def __init__(
        self, 
        model_path: str, 
        spm_path: str, 
        byte_converter_args: Dict,
        separate_embedding: bool,
    ) -> None:
        super().__init__(model_path, separate_embedding)
        if spm_path.endswith(".model"):
            import sentencepiece as spm
            self.sp_type = "spm"
            self.sp_tokenizer = spm.SentencePieceProcessor(model_file=spm_path)
            self.sp_byte_length = max(
                1,
                math.ceil(
                    math.log2(
                        self.sp_tokenizer.vocab_size()
                    ) / 8
                )
            )
        elif spm_path.endswith(".json"):
            from tokenizers import Tokenizer as HF_Tokenizer
            self.sp_type = "hf"
            self.sp_tokenizer = HF_Tokenizer.from_file(spm_path)
            self.sp_byte_length = max(
                1, 
                math.ceil(
                    math.log2(
                        len(self.sp_tokenizer.get_vocab())
                    ) / 8
                )
            )
        else:
            raise ValueError

        if (
            byte_converter_args is not None and
            byte_converter_args.get("byte_converter_type", None) == "gray"
        ):
            self.bytes_encoder = GrayTokenBytes(
                sp_tokenizer=self.sp_tokenizer,
                max_bytelength=self.sp_byte_length
            )
        else:
            if byte_converter_args is not None:
                assert byte_converter_args.get("byte_converter_type", None) in ["simple", None]
            self.bytes_encoder = SimpleTokenBytes(
                max_bytelength=self.sp_byte_length
            )

    def encode_with_spm(self, s: str):
        if self.sp_type == "hf":
            token_ids = self.sp_tokenizer.encode(s).ids
        elif self.sp_type == "spm":
            token_ids = self.sp_tokenizer.encode(s, out_type=int)
        
        bytes_stream = self.bytes_encoder.encode_token_ids_to_bytes(token_ids)
        return bytes_stream

    def decode_with_spm(self, byte_list: list):
        token_ids = self.bytes_encoder.decode_bytes_to_token_ids(byte_list)
        return self.sp_tokenizer.decode(token_ids)

class SimpleBPEDoubleByteTokenizer(SimpleBPEByteTokenizer):
    def __init__(
        self,
        model_path: str,
        spm_path: str,
        byte_converter_args: Dict,
        separate_embedding: bool,
    ) -> None:
        super().__init__(model_path, spm_path, byte_converter_args, separate_embedding)
        self.n_words = 65536 + self.offset
        if self.separate_embedding:
            # when separating the embedding between compressed and raw bytes,
            # we offset compressed embedding indices by 256
            self.n_words = self.n_words + 256
            self.compressed_offset = self.offset + 256
        else:
            self.compressed_offset = None

        self.double_byte_codec = DoubleByteCodec()
    
    def encode_with_spm(self, s: str):
        bytes_stream = super().encode_with_spm(s)
        token_ids = self.double_byte_codec.encode(bytes_stream)
        return token_ids

    def decode_with_spm(self, token_ids: list):
        byte_list = self.double_byte_codec.decode(token_ids)
        return super().decode_with_spm(byte_list)

class HalfByteCodec:
    """
    Split-per-nibble encoder/decoder.
    • encode(b'ABC') -> bytearray([0x4,0x1, 0x4,0x2, 0x4,0x3])
    • decode(bytearray([0x4,0x1, 0x4,0x2, 0x4,0x3])) -> b'ABC'
    """

    HALF_MASK = 0x0F  # 4-bit mask

    @classmethod
    def encode(cls, data: ByteString) -> bytearray:
        """
        Turn a byte string into an even-length stream of half-bytes.
        Each output byte ∈ {0x00 … 0x0F}.
        """
        out = bytearray(len(data) * 2)
        j = 0
        for b in data:
            out[j]   = b >> 4
            out[j+1] = b & cls.HALF_MASK
            j += 2
        return out

    @classmethod
    def decode(cls, encoded: ByteString) -> bytearray:
        """
        Reassemble a half-byte stream into the original bytes.

        Raises
        ------
        ValueError
            * odd number of halfbytes
            * value outside 0x00-0x0F encountered
        """
        n = len(encoded)
        if n & 1:
            n = n - 1

        out = bytearray(n // 2)
        j = 0
        for i in range(0, n, 2):
            hi, lo = encoded[i], encoded[i + 1]
            if (hi | lo) & ~cls.HALF_MASK:
                print(
                    f"Invalid halfbyte value(s) at pos {i//2}: {hi:#x}, {lo:#x}"
                )
                continue

            out[j] = (hi << 4) | lo
            j += 1
        return out

class DoubleByteCodec:
    """
    Group two bytes into one 16-bit integer token and back again.

        >>> tokens = DoubleByteCodec.encode(b"ABCD")
        >>> tokens          # big-endian by default
        [0x4142, 0x4344]

        >>> DoubleByteCodec.decode(tokens)
        b'ABCD'
    """

    def __init__(self, byteorder: str = "big", pad_byte: int = 0x00):
        if byteorder not in ("big", "little"):
            raise ValueError("byteorder must be 'big' or 'little'")
        if not 0 <= pad_byte <= 0xFF:
            raise ValueError("pad_byte must be 0-255")
        self.byteorder = byteorder
        self.pad_byte = pad_byte

    # -------- encode -------------------------------------------------
    def encode(self, data: ByteString) -> List[int]:
        """
        Return a list of 16-bit ints.
        """
        if len(data) & 1:
            data = data + bytes([self.pad_byte])

        tokens: List[int] = []
        for i in range(0, len(data), 2):
            if self.byteorder == "big":
                token = (data[i] << 8) | data[i + 1]
            else:  # little-endian
                token = data[i] | (data[i + 1] << 8)
            tokens.append(token)
        return tokens

    # -------- decode -------------------------------------------------
    def decode(self, tokens: List[int]) -> bytearray:
        """
        Re-create the original byte stream from a list of 0-65535 ints.
        """
        out = bytearray()
        for t in tokens:
            if not 0 <= t <= 0xFFFF:
                raise ValueError(f"Token out of range 0-65535: {t}")
            if self.byteorder == "big":
                out.extend([(t >> 8) & 0xFF, t & 0xFF])
            else:  # little-endian
                out.extend([t & 0xFF, (t >> 8) & 0xFF])
        
        if out and out[-1] == self.pad_byte:
            out = out[:-1]
        return out

class BitCodec:
    """

    Convert any byte into bit streams
    """

    @classmethod
    def encode(cls, data: ByteString) -> List[int]:
        """
        Encode a byte string into a bit stream.
        """
        bits = []
        
        for b in data:
            _bits = [(b >> i) & 1 for i in range(8)]
            bits.extend(_bits[::-1])

        return bits

    @classmethod
    def decode(cls, bits: List[int]) -> bytearray:
        """
        Decode a bit stream into a byte string.
        """
        if len(bits) < 8:
            return b""
        if len(bits) % 8 != 0:
            bits = bits[:len(bits) - len(bits) % 8]
        byte_out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            byte_out.append(byte)
        return byte_out

class DoubleBitCodec:
    """
    Split a byte into 4 2-bit chunks.
    """

    QUAD_MASK = 0x03  # 2-bit mask

    @classmethod
    def encode(cls, data: ByteString) -> bytearray:
        """
        Turn a byte string into an even-length stream of half-bytes.
        Each output byte ∈ {0x00 … 0x0F}.
        """
        out = bytearray(len(data) * 4)
        j = 0
        for b in data:
            out[j]   = (b >> 6) & cls.QUAD_MASK
            out[j+1] = (b >> 4) & cls.QUAD_MASK
            out[j+2] = (b >> 2) & cls.QUAD_MASK
            out[j+3] = b & cls.QUAD_MASK
            j += 4
        return out

    @classmethod
    def decode(cls, encoded: ByteString) -> bytearray:
        """
        Reassemble a half-byte stream into the original bytes.

        Raises
        ------
        ValueError
            * odd number of halfbytes
            * value outside 0x00-0x0F encountered
        """
        n = len(encoded)
        if n < 4:
            return b""
        if n % 4 != 0:
            n = n - (n % 4)

        out = bytearray(n // 4)
        j = 0
        for i in range(0, n, 4):
            out[j] = (
                (encoded[i] << 6) |
                (encoded[i + 1] << 4) |
                (encoded[i + 2] << 2) |
                encoded[i + 3]
            )
            j += 1
        return out

def build_tokenizer(
        name: str, 
        path: Optional[str] = None, 
        spm_byte_path: Optional[str] = None, 
        byte_conversion_args: Optional[Dict] = None,
        separate_embedding: bool = False
    ) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "sp":
        return SentencePieceTokenizer(path)
    elif name == "vanilla":
        return BytePairEncodingTokenizer(path)
    elif name == "vanilla_hf":
        return VanillaHFTokenizer(path)
    elif name == 'hf':
        if spm_byte_path is not None:
            return SimpleBPEByteTokenizer(path, spm_byte_path, byte_conversion_args, separate_embedding)
        else:
            return HFTokenizer(path, separate_embedding)
    elif name == "token_plus_byte":
        return SimpleTokenIDPlusByteTokenizer(path, spm_byte_path, separate_embedding)
    elif name == "doublebyte":
        if spm_byte_path is not None:
            return SimpleBPEDoubleByteTokenizer(path, spm_byte_path, byte_conversion_args, separate_embedding)
        else:
            return DoubleByteTokenizer(path, separate_embedding)
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
