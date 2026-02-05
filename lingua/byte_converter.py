class TokenBytes:
    def __init__(self):
        pass

    def encode_token_ids_to_bytes(self, token_ids):
        pass
    
    def decode_bytes_to_token_ids(self, bytes_stream):
        pass

class GrayTokenBytes(TokenBytes):
    def __init__(self, sp_tokenizer, max_bytelength: int):

        vocab_dict = sp_tokenizer.get_vocab()
        token_bytes_id = []
        for idx, (token, token_id) in enumerate(vocab_dict.items()):
            token_bytes = sp_tokenizer.decode([token_id]).encode("utf-8")
            token_bytes_id.append((token_bytes, token_id))

        self.max_bytelength = max_bytelength

        token_bytes_id_sorted = sorted(token_bytes_id, key=lambda x: x[0])

        token_id_to_bytes = {}
        for rank, (token_bytes, token_id) in enumerate(token_bytes_id_sorted):
            encoded_id = (rank ^ (rank >> 1))
            encoded_bytes = encoded_id.to_bytes(self.max_bytelength, byteorder='big')
            token_id_to_bytes[token_id] = encoded_bytes
        self.token_id_to_bytes = token_id_to_bytes
        self.bytes_to_token_id = {v: k for k, v in token_id_to_bytes.items()}

    def encode_token_ids_to_bytes(self, token_ids):
        encoded_bytes = bytearray()
        for token_id in token_ids:
            encoded_bytes.extend(self.token_id_to_bytes[token_id])
        return bytes(encoded_bytes)

    def decode_bytes_to_token_ids(self, bytes_stream):
        token_ids = []
        for i in range(0, len(bytes_stream) - self.max_bytelength + 1, self.max_bytelength):
            token_bytes = bytes_stream[i:i + self.max_bytelength]
            token_ids.append(self.bytes_to_token_id[bytes(token_bytes)])

        return token_ids

class SimpleTokenBytes(TokenBytes):
    def __init__(self, max_bytelength: int):
        self.max_bytelength = max_bytelength

    def encode_token_ids_to_bytes(self, token_ids):
        byte_array = bytearray()
        for token_id in token_ids:
            token_bytes = token_id.to_bytes(self.max_bytelength, byteorder='big')
            for b in token_bytes:
                byte_array.append(b)
        
        return bytes(byte_array)

    def decode_bytes_to_token_ids(self, bytes_stream):
        token_ids = []
        for i in range(0, len(bytes_stream) - self.max_bytelength + 1, self.max_bytelength):
            token_bytes = bytes_stream[i:i + self.max_bytelength]
            token_id = 0
            for b in token_bytes:
                token_id = (token_id << 8) | b
            token_ids.append(int(token_id))

        return token_ids
