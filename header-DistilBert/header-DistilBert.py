# coding=utf-8
# Copyright 2018 The HuggingFace Team and The HuggingFace Inc. team.
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
"""PyTorch DistilBert model. """


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
from .configuration_distilbert import DistilBertConfig
from .file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_outputs import (
    QuestionAnsweringModelOutput,
    TokenClassificationModelOutput
)
from .modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DistilBertConfig"
_TOKENIZER_FOR_DOC = "DistilBertTokenizer"

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "('distilbert-base-uncased,distilbert-base-cased',)"
    # See all DistilBert models at https://huggingface.co/models?filter=distilbert
]
