# Vendored from https://github.com/99991/pygguf
# If it is to be shipped, correct attribution should be given.

# GGUF specification
# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
import struct
import warnings
import numpy as np

GGML_TYPES = {
    "F32": 0,
    "Q4_0": 2,
    "Q8_0": 8,
    "Q4_K": 12,
    "Q6_K": 14,
}

Q8_0_BLOCK_SIZE = 2 + 32
Q4_K_BLOCK_SIZE = 144
Q4_0_BLOCK_SIZE = 2 + 16
Q6_K_BLOCK_SIZE = 210

DATA_TYPES = {
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
}

for key, value in list(DATA_TYPES.items()):
    DATA_TYPES[value] = key

def read_value(f, data_type):
    if data_type == DATA_TYPES["string"]:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    elif data_type == DATA_TYPES["uint32"]:
        return struct.unpack("<I", f.read(4))[0]

    elif data_type == DATA_TYPES["uint64"]:
        return struct.unpack("<Q", f.read(8))[0]

    elif data_type == DATA_TYPES["int32"]:
        return struct.unpack("<i", f.read(4))[0]

    elif data_type == DATA_TYPES["float32"]:
        return struct.unpack("<f", f.read(4))[0]

    elif data_type == DATA_TYPES["array"]:
        data_type, count = struct.unpack("<IQ", f.read(4+8))
        return [read_value(f, data_type) for _ in range(count)]
    elif data_type == DATA_TYPES["bool"]:
       # This should correspond to `GGUF_METADATA_VALUE_TYPE_BOOL`
       # 1-byte value where 0 is false and 1 is true.
       return struct.unpack("<b", f.read(1))[0]
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def load_gguf(f):
    f.seek(0)
    assert f.read(4) == b"GGUF"
    values = struct.unpack("<IQQ", f.read(4+8+8))
    version, n_tensors, n_kv = values
    if version != 3:
        warnings.warn(f"Version {version} has never been tested, might not work")

    # import pdb; pdb.set_trace()

    info = {}
    for _ in range(n_kv):
        name = read_value(f, DATA_TYPES["string"])
        print(name)

        data_type = struct.unpack("<I", f.read(4))[0]

        info[name] = read_value(f, data_type)

    tensorinfo = {}
    for _ in range(n_tensors):
        name = read_value(f, DATA_TYPES["string"])
        shape_len = read_value(f, DATA_TYPES["uint32"])
        shape = [read_value(f, DATA_TYPES["uint64"]) for _ in range(shape_len)]
        ggml_type = read_value(f, DATA_TYPES["uint32"])
        bad_offset = read_value(f, DATA_TYPES["uint64"])

        tensorinfo[name] = {
            "ggml_type": ggml_type,
            "shape": shape,
            "bad_offset": bad_offset,
        }

    start = f.tell()

    # Inconveniently, the offset defined in gguf files is relative to the
    # end of the header and is unaligned.
    # We need to compute the absolute file offset ourselves instead.
    for t in tensorinfo.values():
        offset = start + t["bad_offset"]

        # Whey 32? See: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#required
        # TODO: infer that from the config?
        alignment = 32
        offset += (alignment - offset % alignment) % alignment

        t["offset"] = offset

    return info, tensorinfo

def dequantize_q4_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    num_blocks = len(data) // Q4_K_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q4_K_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q4_K_BLOCK_SIZE)

    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)

    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
    offsets = scale_offsets * np.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)

    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    return factors * qs2 - offsets

def dequantize_q4_0(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1086
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11
    num_blocks = len(data) // Q4_0_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q4_0_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q4_0_BLOCK_SIZE)

    # The scales are stored on the first 2 bytes and the rest corresponds to the quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    # scales = np.nan_to_num(scales)
    # the rest of the bytes corresponds to the quants - we discard the first two bytes
    quants = data_u8[:, 2:]

    ql = (quants[:, :] & 0xf).astype(np.int8) - 8
    qr = (quants[:, :] >> 4).astype(np.int8) - 8

    # Use hstack 
    quants = np.hstack([ql, qr])

    return (scales * quants).astype(np.float32)

def dequantize_q6_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    num_blocks = len(data) // Q6_K_BLOCK_SIZE

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, Q6_K_BLOCK_SIZE // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, Q6_K_BLOCK_SIZE)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, Q6_K_BLOCK_SIZE)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)

    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:,   :32 ] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64 ] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:,   :32 ] >>  4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64 ] >>  4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96 ] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96 ] >>  4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >>  4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    return scales * np.concatenate([
        sc[:,  0] * q1[:, :16],
        sc[:,  1] * q1[:, 16:],
        sc[:,  2] * q2[:, :16],
        sc[:,  3] * q2[:, 16:],
        sc[:,  4] * q3[:, :16],
        sc[:,  5] * q3[:, 16:],
        sc[:,  6] * q4[:, :16],
        sc[:,  7] * q4[:, 16:],
        sc[:,  8] * q5[:, :16],
        sc[:,  9] * q5[:, 16:],
        sc[:, 10] * q6[:, :16],
        sc[:, 11] * q6[:, 16:],
        sc[:, 12] * q7[:, :16],
        sc[:, 13] * q7[:, 16:],
        sc[:, 14] * q8[:, :16],
        sc[:, 15] * q8[:, 16:],
    ], axis=1)

def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    num_blocks = len(data) // Q8_0_BLOCK_SIZE

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]

    return scales * qs

def load_gguf_tensor(f, tensorinfo, name):
    t = tensorinfo[name]
    offset = t["offset"]
    shape = t["shape"]
    ggml_type = t["ggml_type"]
    num_elements = np.prod(shape)
    f.seek(offset)

    print(ggml_type)
    # import pdb; pdb.set_trace()

    if ggml_type == GGML_TYPES["F32"]:
        size = num_elements * 4
        values = np.frombuffer(f.read(size), dtype=np.float32)

    elif ggml_type == GGML_TYPES["Q8_0"]:
        size = num_elements * Q8_0_BLOCK_SIZE // 32
        data = f.read(size)

        values = dequantize_q8_0(data)

    elif ggml_type == GGML_TYPES["Q4_0"]:
        size = num_elements * Q4_0_BLOCK_SIZE // 32
        data = f.read(size)

        values = dequantize_q4_0(data)

    elif ggml_type == GGML_TYPES["Q4_K"]:
        size = num_elements * Q4_K_BLOCK_SIZE // 256
        data = f.read(size)

        values = dequantize_q4_k(data)

    elif ggml_type == GGML_TYPES["Q6_K"]:
        size = num_elements * Q6_K_BLOCK_SIZE // 256
        data = f.read(size)

        values = dequantize_q6_k(data)

    else:
        raise NotImplementedError(f"ggml_type {ggml_type} not implemented")

    return values.reshape(shape[::-1])