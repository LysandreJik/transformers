# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TODO intro """
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Dict

from gguf import GGUFValueType

from .utils.gguf_utils import load_gguf_tensor, load_gguf

from .utils.logging import get_logger

logger = get_logger(__name__)



@dataclass
class GGUFSupportedArchitecture:
    architecture_name: str
    model_prefix_name: str
    key_mapping: Dict[str, str]


llama_mapping = GGUFSupportedArchitecture('llama', 'llama', {

})
GGUF_SUPPORTED_ARCHITECTURES = {
    'llama': llama_mapping,
    'qwen2': llama_mapping,
    'gemma': llama_mapping,
}

renames = {
    "ignore": {
        'GGUF': {
            'version': 'version',
            'tensor_count': 'tensor_count',
            'kv_count': 'kv_count',
        },
        'general': {
            'file_type': 'file_type',
            'quantization_version': 'quantization_version'
        }
    },
    'config': {
        'general': {
            'architecture': 'model_type',
            'name': '_model_name_or_path',
        },
        'llama': {
            'context_length': 'max_position_embeddings',
            'block_count': 'num_hidden_layers',
            'feed_forward_length': 'intermediate_size',
            'embedding_length': 'hidden_size',
            'rope.dimension_count': None,
            'rope.freq_base': 'rope_theta',
            'attention.head_count': 'num_attention_heads',
            'attention.head_count_kv': None,
            'attention.layer_norm_rms_epsilon': 'rms_norm_eps',
            'vocab_size': 'vocab_size'
        },
        'gemma': {
            'context_length': 'max_position_embeddings',
            'block_count': 'num_hidden_layers',
            'feed_forward_length': 'intermediate_size',
            'embedding_length': 'hidden_size',
            'rope.dimension_count': None,
            'rope.freq_base': 'rope_theta',
            'attention.head_count': 'num_attention_heads',
            'attention.head_count_kv': None,
            'attention.layer_norm_rms_epsilon': 'rms_norm_eps',
        },
        'qwen2': {
            'context_length': 'max_position_embeddings',
            'block_count': 'num_hidden_layers',
            'feed_forward_length': 'intermediate_size',
            'embedding_length': 'hidden_size',
            'rope.dimension_count': None,
            'rope.freq_base': 'rope_theta',
            'attention.head_count': 'num_attention_heads',
            'attention.head_count_kv': None,
            'attention.layer_norm_rms_epsilon': 'rms_norm_eps',
            'use_parallel_residual': False
        },
        'tokenizer': {
            'ggml.model': 'model_type',
            'ggml.bos_token_id': 'bos_token_id',
            'ggml.eos_token_id': 'eos_token_id',
            'ggml.unknown_token_id': 'unk_token_id',
            'ggml.padding_token_id': 'pad_token_id',
        }
    },
    'tensors': {
        "llama": {
            "token_embd": "model.embed_tokens",
            "blk": "model.layers",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "attn_norm": "input_layernorm",
            "attn_q": "self_attn.q_proj",
            "attn_v": "self_attn.v_proj",
            "attn_k": "self_attn.k_proj",
            "attn_output": "self_attn.o_proj",
            "output.weight": "lm_head.weight",
            "output_norm": "model.norm",
        },
        "gemma": {
            "token_embd": "model.embed_tokens",
            "blk": "model.layers",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "attn_norm": "input_layernorm",
            "attn_q": "self_attn.q_proj",
            "attn_v": "self_attn.v_proj",
            "attn_k": "self_attn.k_proj",
            "attn_output": "self_attn.o_proj",
            "output.weight": "lm_head.weight",
            "output_norm": "model.norm",
        },
        "qwen2": {
            "token_embd": "model.embed_tokens",
            "blk": "model.layers",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "attn_norm": "input_layernorm",
            "attn_q": "self_attn.q_proj",
            "attn_v": "self_attn.v_proj",
            "attn_k": "self_attn.k_proj",
            "attn_output": "self_attn.o_proj",
            "output.weight": "lm_head.weight",
            "output_norm": "model.norm",
        }
    },
    'tokenizer': {
        'tokenizer': {
            'ggml.model': 'tokenizer_type',
            'ggml.tokens': 'tokens',
            'ggml.scores': 'scores',
            'ggml.token_type': 'token_type',
            'ggml.merges': 'merges',
            'ggml.bos_token_id': 'bos_token_id',
            'ggml.eos_token_id': 'eos_token_id',
            'ggml.unknown_token_id': 'unk_token_id',
            'ggml.padding_token_id': 'pad_token_id',
        }
    },
    'tokenizer_config': {
        'tokenizer': {
            'chat_template': 'chat_template',
            'ggml.model': 'model_type',
            'ggml.bos_token_id': 'bos_token_id',
            'ggml.eos_token_id': 'eos_token_id',
            'ggml.unknown_token_id': 'unk_token_id',
            'ggml.padding_token_id': 'pad_token_id',
        }
    },
}

def read_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        assert data_type[0] == 9, "Received multiple types, but therefore expect the first type to indicate an array."
        data_type, array_data_type = data_type


    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = ''.join([chr(a) for a in _value])
    elif data_type in [9]:
        _value = read_value(_value, array_data_type)

    return _value


def load_gguf_checkpoint_in_pytorch_model(
    gguf_checkpoint_path, output_loading_info=False
):
    """
    TODO Docs
    """
    try:
        from gguf import GGUFReader
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise


    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())


    parsed_parameters = {k: {} for k in renames}

    # List all key-value pairs in a columnized format
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        gguf_key = key
        split = gguf_key.split('.')
        prefix = split[0]
        config_key = '.'.join(split[1:])

        value = [read_value(field.parts[_data_index], field.types) for _data_index in field.data]
        if len(value) == 1:
            value = value[0]

        for parameter in renames:
            parameter_renames = renames[parameter]
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames.get(prefix, {}).get(config_key)
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            print("Not added", gguf_key, value)

    # List all tensors
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        renamed_tensor_name = tensor.name

        for tensor_name_mapping in renames["tensors"]:
            if tensor_name_mapping in renamed_tensor_name:
                renamed_tensor_name = renamed_tensor_name.replace(tensor_name_mapping, renames["tensors"][tensor_name_mapping])

        parsed_parameters['tensors'][tensor.name] = {'tensor_type': tensor.tensor_type, 'data': tensor.data, "transformers_key": renamed_tensor_name}

    print(f"Remaining keys: {reader_keys}")
    return parsed_parameters

def load_and_convert_gguf_file(gguf_checkpoint_path):
    """
    TODO Docs
    """

    converted_state_dict = {}

    with open(gguf_checkpoint_path, "rb") as f:
        # Load metadata
        info, tensorinfo = load_gguf(f)
        architecture = info["general.architecture"]

        # import pdb; pdb.set_trace()

        if architecture not in GGUF_SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {architecture} not supported"
            )

        tensor_key_mapping = renames["tensors"][architecture]

        for name in tensorinfo:
            weights = load_gguf_tensor(f, tensorinfo, name)
            shape = tensorinfo[name]["shape"]

            if architecture == "llama" and (".attn_k." in name or ".attn_q." in name):
                num_heads = info[f"{architecture}.attention.head_count"]
                tmp_shape = (shape[-1] // num_heads // 2, num_heads, 2, shape[0])
                weights = weights.reshape(tmp_shape)
                weights = weights.transpose(0, 2, 1, 3)
                weights = weights.reshape(shape[::-1])

            for tensor_name in tensor_key_mapping:
                if tensor_name in name:
                    name = name.replace(tensor_name, tensor_key_mapping[tensor_name])

            converted_state_dict[name] = torch.from_numpy(np.copy(weights))
    
    return converted_state_dict