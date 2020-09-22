# coding=utf-8
# Copyright {{cookiecutter.authors}} and The HuggingFace Inc. team.
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
""" {{cookiecutter.uppercase_modelname}} model configuration """

from configuration_utils import PretrainedConfig
from utils import logging


logger = logging.get_logger(__name__)

{{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "{{cookiecutter.checkpoint_identifier}}": "https://s3.amazonaws.com/models.huggingface.co/bert/{{cookiecutter.checkpoint_identifier}}/config.json",
    # See all {{cookiecutter.uppercase_modelname}} models at https://huggingface.co/models?filter={{cookiecutter.lowercase_modelname}}
}


class {{cookiecutter.modelname}}Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.{{cookiecutter.modelname}}Model`.
    It is used to instantiate an {{cookiecutter.modelname}} model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the {{cookiecutter.modelname}} `{{cookiecutter.checkpoint_identifier}} <https://huggingface.co/{{cookiecutter.checkpoint_identifier}}>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.


    Args:
        vocab_size (:obj:`int`, optional, defaults to 30522):
            Vocabulary size of the {{cookiecutter.modelname}} model. Defines the different tokens that
            can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.{{cookiecutter.modelname}}Model`.
        hidden_size (:obj:`int`, optional, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, optional, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, "gelu", "relu", "swish" and "gelu_new" are supported.
        hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, optional, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, optional, defaults to 2):
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.{{cookiecutter.modelname}}Model`.
        initializer_range (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, optional, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        >>> from transformers import {{cookiecutter.modelname}}Model, {{cookiecutter.modelname}}Config

        >>> # Initializing a {{cookiecutter.modelname}} {{cookiecutter.checkpoint_identifier}} style configuration
        >>> configuration = {{cookiecutter.modelname}}Config()

        >>> # Initializing a model from the {{cookiecutter.checkpoint_identifier}} style configuration
        >>> model = {{cookiecutter.modelname}}Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "{{cookiecutter.lowercase_modelname}}"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
