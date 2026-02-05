def pack_compressed_spans(data, bits_per_compressed: int, compression_bit_threshold: int, compression_offset: int = 256):
    """
    Convert consecutive compressed values into larger integers.
    
    Args:
        data: List of integers where 0-255 are raw bytes, compression_offset+ are compressed bytes
        bits_per_compressed: Number of bits to use for each packed value
        compression_bit_threshold: Number of bits each compressed value actually uses
        compression_offset: Offset that marks start of compressed values (default 256)
    
    Returns:
        List with consecutive compressed spans packed into larger integers
    """
    if not data:
        return []
    
    result = []
    i = 0

    assert compression_bit_threshold % bits_per_compressed == 0, "compression_bit_threshold must be divisible by bits_per_compressed"
    packing_mask = (1 << bits_per_compressed) - 1
    compression_mask = (1 << compression_bit_threshold) - 1
    
    # Calculate byte-aligned padded size
    padded_compression_bit_threshold = ((compression_bit_threshold + 7) // 8) * 8
    padded_mask = (1 << padded_compression_bit_threshold) - 1

    padding_bits = padded_compression_bit_threshold - compression_bit_threshold
    
    while i < len(data):
        if data[i] >= compression_offset:
            # Find the end of consecutive compressed bytes
            span_start = i
            while i < len(data) and data[i] >= compression_offset:
                i += 1
            
            # Extract the span of compressed bytes
            compressed_span = data[span_start:i]
            
            base_values = [x - compression_offset for x in compressed_span]
            
            # Process bytes incrementally to avoid large numbers
            bit_buffer = 0
            bits_in_buffer = 0
            packed_values = []
            
            for val in base_values:
                # Add this byte to bit buffer
                bit_buffer = (bit_buffer << 8) | val
                bits_in_buffer += 8
                
                # Extract padded chunks as soon as we have enough bits
                while bits_in_buffer >= padded_compression_bit_threshold:
                    shift_amount = bits_in_buffer - padded_compression_bit_threshold
                    padded_val = (bit_buffer >> shift_amount) & padded_mask
                    
                    # Remove the extracted bits from buffer
                    bit_buffer &= (1 << shift_amount) - 1
                    bits_in_buffer -= padded_compression_bit_threshold
                    
                    # Strip padding by extracting only the meaningful bits
                    extracted_val = (padded_val >> padding_bits) & compression_mask
                    
                    pack_buffer = extracted_val
                    pack_bits = compression_bit_threshold
                    
                    # Pack values as soon as we have enough bits
                    while pack_bits >= bits_per_compressed:
                        pack_shift = pack_bits - bits_per_compressed
                        packed_val = (pack_buffer >> pack_shift) & packing_mask
                        packed_values.append(packed_val + compression_offset)
                        
                        # Remove packed bits from pack buffer
                        pack_buffer &= (1 << pack_shift) - 1
                        pack_bits -= bits_per_compressed
            
                    assert bits_in_buffer == 0, "bits_in_buffer must be 0 after processing compressed span"
                    assert pack_bits == 0, "pack_bits must be 0 after packing"
                    
            result.extend(packed_values)
        else:
            # Raw byte (0-255), keep as is
            result.append(data[i])
            i += 1
    
    return result

def pseudo_to_packed_bytes(lst: list[int]) -> bytes:
    out = bytearray()
    acc = bits = 0
    for v in lst:
        acc |= (v & 0x1FF) << bits
        bits += 9
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:                        # flush tail
        out.append(acc)
    return bytes(out)

def packed_bytes_to_pseudo(b: bytes) -> list[int]:
    out, acc, bits = [], 0, 0
    for byte in b:
        acc |= byte << bits
        bits += 8
        while bits >= 9:
            out.append(acc & 0x1FF)
            acc >>= 9
            bits -= 9
    return out
