# coding=utf-8
# Copyright 2018 {{cookiecutter.authors}} and The HuggingFace Inc. team.
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
"""PyTorch {{cookiecutter.model_name}} model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .activations import gelu, gelu_new, swish
from .configuration_{{cookiecutter.lowercase_model_name}} import {{cookiecutter.model_name}}Config
from .file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_outputs import (
    {{cookiecutter.model_outputs}}
)
from .modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "{{cookiecutter.model_name}}Config"
_TOKENIZER_FOR_DOC = "{{cookiecutter.model_name}}Tokenizer"

{{cookiecutter.uppercase_model_name}}_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "{{cookiecutter.checkpoint_identifier}}"
    # See all {{cookiecutter.model_name}} models at https://huggingface.co/models?filter={{cookiecutter.lowercase_model_name}}
]
