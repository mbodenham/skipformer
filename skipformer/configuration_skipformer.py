import os
from typing import Any, List
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

root_dir = os.path.join(os.path.dirname(__file__))
SKIPFORMER_PRETRAINED_CONFIG = {
    "gpt2-small":   os.path.join(root_dir, "models", "gpt2-small", "config.json"),
    "gpt2-medium":   os.path.join(root_dir, "models", "gpt2-medium", "config.json"),
    "gpt2-large":   os.path.join(root_dir, "models", "gpt2-large", "config.json"),
    "gpt2-small-w": os.path.join(root_dir, "models", "gpt2-small-w", "config.json"),
    "skipformer-a": os.path.join(root_dir, "models", "skipformer-a", "config.json"),
    "skipformer-a-medium": os.path.join(root_dir, "models", "skipformer-a-medium", "config.json"),
    "skipformer-b": os.path.join(root_dir, "models", "skipformer-b", "config.json")
}

class SkipformerConfig(PretrainedConfig):
    model_type = "skipformer"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50259,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50257,
        sep_token_id=50258,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        in_dim=768,
        out_dim=768,
        attention_window=False,
        model_subtype="skipformer-a",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attention_window = attention_window
        self.model_subtype = model_subtype

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, sep_token_id=sep_token_id, **kwargs)
