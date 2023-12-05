import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parameter import Parameter

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_skipformer import SkipformerConfig
from . import attention_window_matmul
# from .blocks import SkipformerBlock1, SkipformerBlock2, SkipformerBlock3, SkipformerBlock4

# from .sparse_attention.WindowBatchMatMul import WindowBatchMatMul

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "skipformer"
_CONFIG_FOR_DOC = "SkipformerConfig"
_TOKENIZER_FOR_DOC = "SkipformerTokenizer"

SKIPFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2-small",
    "gpt2-small-W",
    "skipformer-a",
    "skipformer-b",
]

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class SkipformerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, embed_dim=None, num_heads=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        if embed_dim is None:
            self.embed_dim = config.hidden_size
        else:
            self.embed_dim = embed_dim

        if num_heads is None:
            self.num_heads = config.num_attention_heads
        else:
            self.num_heads = num_heads

        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)

        self.pruned_heads = set()

        if config.attention_window:
            self.attention_window = AdaptiveWindow(
                                       # num_heads=self.num_heads,
                                       num_heads=1,
                                       max_window_size=1024,
                                       window_ramp_size=16,
                                       loss_coeff=0.0001,
                                       ramp_func='tanh')
        else:
            self.attention_window = None
            self.register_buffer("window_size", torch.tensor([1024.]))

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if self.attention_window:
            attn_weights = self.attention_window(query, key.transpose(-1, -2))
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class SkipformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SkipformerConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SkipformerModel):
            module.gradient_checkpointing = value


class SkipformerModel(SkipformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        # SKIPFORMERS = {'gpt-2_small':  {'module': GPT2Small,
        #                                'in_dim': 768,
        #                                'out_dim': 768,
        #                                'attention_window': False},
        #                'gpt-2_small-w': {'module': GPT2Small,
        #                                 'in_dim': 768,
        #                                 'out_dim': 768,
        #                                 'attention_window': True},
        #                 'skipformer_a': {'module': SkipformerA,
        #                                  'in_dim': 768,
        #                                  'out_dim': 768,
        #                                  'attention_window': True},
        #                'skipformer_b': {'module': SkipformerB,
        #                                 'in_dim': 768,
        #                                 'out_dim': 768,
        #                                 'attention_window': True}
        #                }

        SKIPFORMERS = {'gpt2-small':   GPT2Small,
                       'gpt2-medium':  GPT2Medium,
                       'gpt2-large':   GPT2Large,
                       'gpt2-small-w': GPT2Small,
                       'skipformer-a': SkipformerA,
                       'skipformer-a-medium': SkipformerAMedium,
                       'skipformer-b': SkipformerB}
        #
        # self.embed_dim = config.hidden_size
        # self.in_dim = SKIPFORMERS[config.model_name]['in_dim']
        # self.out_dim = SKIPFORMERS[config.model_name]['out_dim']
        # self.attention_window = SKIPFORMERS[config.model_name]['attention_window']

        # self.embed_dim = config.hidden_size
        # self.in_dim = config.in_dim
        # self.out_dim = config.out_dim
        # self.attention_window = config.attention_window

        self.wte = nn.Embedding(config.vocab_size, config.in_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.in_dim)

        # self.h = nn.ModuleList([SKIPFORMERS[config.model_type]['module'](config, layer_idx=None)])
        self.h = nn.ModuleList([SKIPFORMERS[config.model_subtype](config, layer_idx=None)])

        self.drop = None

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        self.attention_window = config.attention_window
        self.attention_window_layers = []
        if config.attention_window:
            for block in self.h:
                for layer in block.modules():
                    if hasattr(layer, 'attention_window') and layer.attention_window:
                        self.attention_window_layers.append(layer)

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds


        # print(input_shape)
        # print(hidden_states.size(-1))

        output_shape = input_shape + (hidden_states.size(-1),)
        # output_shape = input_shape + (768,)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))


        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class SkipformerLMHeadModel(SkipformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"window_size"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = SkipformerModel(config)
        self.lm_head = nn.Linear(config.in_dim, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()


    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        pass
        # self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # return (lm_logits,)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )



class SkipformerForSequenceClassification(SkipformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = SkipformerModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), f"Cannot handle batch sizes > 1 if no padding token is defined. {self.config.pad_token_id}_{batch_size}"
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class AdaptiveWindow(nn.Module):
    def __init__(self, num_heads, max_window_size, window_ramp_size, loss_coeff, masked=True,
                 init_val=1.0, ramp_func='linear', **factory_kwargs):
        super(AdaptiveWindow, self).__init__()
        self.num_heads = num_heads
        self.masked = masked

        max_window_size = torch.tensor(max_window_size, **factory_kwargs)
        self.register_buffer('max_window_size', max_window_size)
        self.window_ramp_size = window_ramp_size
        self.loss_coeff = loss_coeff

        mask_template = self.gen_mask_template(max_window_size)
        self.register_buffer('mask_template', mask_template)


        if ramp_func == 'linear':
            self.mask_func = lambda mt, z, S, R: ((mt + z*S)/R + 1).clamp(0, 1)
        elif ramp_func == 'exp':
            self.mask_func = lambda x, mt, z, S, R: (1  / (1 + torch.exp( ( 20/R ) * (-mt-z*S-0.5*R ) ))).clamp(0, 1)
        elif ramp_func == 'tanh':
            self.mask_func = lambda mt, z, S, R: torch.tanh( (4/R) * (mt+z*S+R) ).clamp(0, 1)
        elif ramp_func == 'tanhv':
            self.mask_func = lambda  x, mt, z, S, R: torch.tanh( ( (S-(0.5*z*S)) / ((10*z*S)+100*(z+0.01)) ) * (mt+z*S+R) ).clamp(0, 1)
        elif ramp_func == '1/x':
            self.mask_func = lambda x, mt, z, S, R: ( R / ( 10*(-mt-z*S) ) ).clamp(0, 1)


        self.current_val = Parameter(torch.empty((num_heads, 1), **factory_kwargs))
        nn.init.constant_(self.current_val, init_val)

        z = self.current_val.view(1, self.num_heads, 1, 1)
        mask = self.mask_func(self.mask_template, z, self.max_window_size, self.window_ramp_size)
        self.register_buffer('mask', mask)
        self.window_size = None


    def forward(self, q, k, masked=True):
        if self.training:
            z = self.current_val.view(self.num_heads, 1, 1)
            self.mask = self.mask_func(self.mask_template, z, self.max_window_size, self.window_ramp_size)

            x = torch.matmul(q, k) * self.mask
        else:
            if self.window_size is None:
                self.init_mask()

            x = attention_window_matmul.apply(q,
                                                k,
                                                self.window_size,
                                                self.mask,
                                                self.masked)

        return x


    def init_mask(self):
        # self.current_val[0] = 1.
        z = self.current_val.view(self.num_heads, 1, 1)
        # z = self.current_val.view(1, self.num_heads, 1, 1)
        # self.current_val[0] = 1.
        self.mask = self.mask_func(self.mask_template, z, self.max_window_size, self.window_ramp_size)
        self.window_size = self.get_current_size()
        if self.window_size.shape[0] == 1:
            self.window_size = self.window_size.repeat([12])

        # print(self.window_size[0])
        # self.window_size_block = torch.ceil( (1024//128) * (self.window_size/1024) ) + 1;
        # self.window_size_thread = thread_ws = torch.ceil( (1024//8)  * (self.window_size/1024) ) + 1;
        # print(self.window_size, self.window_size_block, self.window_size_thread)
        return z.item()
        # print(z)

    def gen_mask_template(self, size):
        mask_template = torch.zeros([size, size])
        c = torch.cat((torch.arange(-size, 1, 1), torch.arange(-1, -size-1, -1)))

        for i in range(size):
            mask_template[i,:] = c[size-i:2*size-i]

        return mask_template

    def get_current_size(self, include_ramp=True):
        current_size = torch.ceil(torch.flatten(self.current_val) * self.max_window_size)
        if include_ramp:
            current_size += self.window_ramp_size
        current_size = current_size.clamp(0, self.max_window_size)
        return current_size


    def clamp_param(self):
        self.current_val.data.clamp_(0, 1)


    def get_loss(self):
        loss = self.max_window_size * self.current_val.mean()
        # loss = self.current_val.mean()

        return loss


class ReLUSquared(nn.Module):
    def __init__(self):
        super(ReLUSquared, self).__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


class GPT2Small(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(GPT2Small, self).__init__()
        #78173
        #encoder layers
        self.dropout0 = nn.Dropout(config.resid_pdrop)
        self.layernorm0 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention0 = SkipformerAttention(config, layer_idx=0, embed_dim=768,  num_heads=12)
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.layernorm1 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear0 = nn.Linear(768, 3072)
        self.gelu0 = nn.GELU()
        self.linear1 = nn.Linear(3072, 768)
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.layernorm2 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention1 = SkipformerAttention(config, layer_idx=1, embed_dim=768,  num_heads=12)
        self.dropout3 = nn.Dropout(config.resid_pdrop)
        self.layernorm3 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear2 = nn.Linear(768, 3072)
        self.gelu1 = nn.GELU()
        self.linear3 = nn.Linear(3072, 768)
        self.dropout4 = nn.Dropout(config.resid_pdrop)
        self.layernorm4 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention2 = SkipformerAttention(config, layer_idx=2, embed_dim=768,  num_heads=12)
        self.dropout5 = nn.Dropout(config.resid_pdrop)
        self.layernorm5 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear4 = nn.Linear(768, 3072)
        self.gelu2 = nn.GELU()
        self.linear5 = nn.Linear(3072, 768)
        self.dropout6 = nn.Dropout(config.resid_pdrop)
        self.layernorm6 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention3 = SkipformerAttention(config, layer_idx=3, embed_dim=768,  num_heads=12)
        self.dropout7 = nn.Dropout(config.resid_pdrop)
        self.layernorm7 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear6 = nn.Linear(768, 3072)
        self.gelu3 = nn.GELU()
        self.linear7 = nn.Linear(3072, 768)
        self.dropout8 = nn.Dropout(config.resid_pdrop)
        self.layernorm8 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention4 = SkipformerAttention(config, layer_idx=4, embed_dim=768,  num_heads=12)
        self.dropout9 = nn.Dropout(config.resid_pdrop)
        self.layernorm9 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear8 = nn.Linear(768, 3072)
        self.gelu4 = nn.GELU()
        self.linear9 = nn.Linear(3072, 768)
        self.dropout10 = nn.Dropout(config.resid_pdrop)
        self.layernorm10 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention5 = SkipformerAttention(config, layer_idx=5, embed_dim=768,  num_heads=12)
        self.dropout11 = nn.Dropout(config.resid_pdrop)
        self.layernorm11 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear10 = nn.Linear(768, 3072)
        self.gelu5 = nn.GELU()
        self.linear11 = nn.Linear(3072, 768)
        self.dropout12 = nn.Dropout(config.resid_pdrop)
        self.layernorm12 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention6 = SkipformerAttention(config, layer_idx=6, embed_dim=768,  num_heads=12)
        self.dropout13 = nn.Dropout(config.resid_pdrop)
        self.layernorm13 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear12 = nn.Linear(768, 3072)
        self.gelu6 = nn.GELU()
        self.linear13 = nn.Linear(3072, 768)
        self.dropout14 = nn.Dropout(config.resid_pdrop)
        self.layernorm14 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention7 = SkipformerAttention(config, layer_idx=7, embed_dim=768,  num_heads=12)
        self.dropout15 = nn.Dropout(config.resid_pdrop)
        self.layernorm15 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear14 = nn.Linear(768, 3072)
        self.gelu7 = nn.GELU()
        self.linear15 = nn.Linear(3072, 768)
        self.dropout16 = nn.Dropout(config.resid_pdrop)
        self.layernorm16 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention8 = SkipformerAttention(config, layer_idx=8, embed_dim=768,  num_heads=12)
        self.dropout17 = nn.Dropout(config.resid_pdrop)
        self.layernorm17 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear16 = nn.Linear(768, 3072)
        self.gelu8 = nn.GELU()
        self.linear17 = nn.Linear(3072, 768)
        self.dropout18 = nn.Dropout(config.resid_pdrop)
        self.layernorm18 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention9 = SkipformerAttention(config, layer_idx=9, embed_dim=768,  num_heads=12)
        self.dropout19 = nn.Dropout(config.resid_pdrop)
        self.layernorm19 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear18 = nn.Linear(768, 3072)
        self.gelu9 = nn.GELU()
        self.linear19 = nn.Linear(3072, 768)
        self.dropout20 = nn.Dropout(config.resid_pdrop)
        self.layernorm20 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention10 = SkipformerAttention(config, layer_idx=10, embed_dim=768,  num_heads=12)
        self.dropout21 = nn.Dropout(config.resid_pdrop)
        self.layernorm21 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear20 = nn.Linear(768, 3072)
        self.gelu10 = nn.GELU()
        self.linear21 = nn.Linear(3072, 768)
        self.dropout22 = nn.Dropout(config.resid_pdrop)
        self.layernorm22 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention11 = SkipformerAttention(config, layer_idx=11, embed_dim=768,  num_heads=12)
        self.dropout23 = nn.Dropout(config.resid_pdrop)
        self.layernorm23 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear22 = nn.Linear(768, 3072)
        self.gelu11 = nn.GELU()
        self.linear23 = nn.Linear(3072, 768)
        self.dropout24 = nn.Dropout(config.resid_pdrop)
        self.layernorm24 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        #encoder forward
        hidden_states = self.dropout0(hidden_states)
        skip0 = hidden_states #skip to self.dropout1

        hidden_states = self.layernorm0(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention0(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout1(hidden_states)
        hidden_states += skip0
        skip1 = hidden_states #skip to self.dropout2

        hidden_states = self.layernorm1(hidden_states)
        hidden_states = self.linear0(hidden_states)
        hidden_states = self.gelu0(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states += skip1
        skip2 = hidden_states #skip to self.dropout3

        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention1(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout3(hidden_states)
        hidden_states += skip2
        skip3 = hidden_states #skip to self.dropout4

        hidden_states = self.layernorm3(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.gelu1(hidden_states)
        hidden_states = self.linear3(hidden_states)
        hidden_states = self.dropout4(hidden_states)
        hidden_states += skip3
        skip4 = hidden_states #skip to self.dropout5

        hidden_states = self.layernorm4(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention2(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout5(hidden_states)
        hidden_states += skip4
        skip5 = hidden_states #skip to self.dropout6

        hidden_states = self.layernorm5(hidden_states)
        hidden_states = self.linear4(hidden_states)
        hidden_states = self.gelu2(hidden_states)
        hidden_states = self.linear5(hidden_states)
        hidden_states = self.dropout6(hidden_states)
        hidden_states += skip5
        skip6 = hidden_states #skip to self.dropout7

        hidden_states = self.layernorm6(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention3(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout7(hidden_states)
        hidden_states += skip6
        skip7 = hidden_states #skip to self.dropout8

        hidden_states = self.layernorm7(hidden_states)
        hidden_states = self.linear6(hidden_states)
        hidden_states = self.gelu3(hidden_states)
        hidden_states = self.linear7(hidden_states)
        hidden_states = self.dropout8(hidden_states)
        hidden_states += skip7
        skip8 = hidden_states #skip to self.dropout9

        hidden_states = self.layernorm8(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention4(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout9(hidden_states)
        hidden_states += skip8
        skip9 = hidden_states #skip to self.dropout10

        hidden_states = self.layernorm9(hidden_states)
        hidden_states = self.linear8(hidden_states)
        hidden_states = self.gelu4(hidden_states)
        hidden_states = self.linear9(hidden_states)
        hidden_states = self.dropout10(hidden_states)
        hidden_states += skip9
        skip10 = hidden_states #skip to self.dropout11

        hidden_states = self.layernorm10(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention5(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout11(hidden_states)
        hidden_states += skip10
        skip11 = hidden_states #skip to self.dropout12

        hidden_states = self.layernorm11(hidden_states)
        hidden_states = self.linear10(hidden_states)
        hidden_states = self.gelu5(hidden_states)
        hidden_states = self.linear11(hidden_states)
        hidden_states = self.dropout12(hidden_states)
        hidden_states += skip11
        skip12 = hidden_states #skip to self.dropout13

        hidden_states = self.layernorm12(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention6(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout13(hidden_states)
        hidden_states += skip12
        skip13 = hidden_states #skip to self.dropout14

        hidden_states = self.layernorm13(hidden_states)
        hidden_states = self.linear12(hidden_states)
        hidden_states = self.gelu6(hidden_states)
        hidden_states = self.linear13(hidden_states)
        hidden_states = self.dropout14(hidden_states)
        hidden_states += skip13
        skip14 = hidden_states #skip to self.dropout15

        hidden_states = self.layernorm14(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention7(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout15(hidden_states)
        hidden_states += skip14
        skip15 = hidden_states #skip to self.dropout16

        hidden_states = self.layernorm15(hidden_states)
        hidden_states = self.linear14(hidden_states)
        hidden_states = self.gelu7(hidden_states)
        hidden_states = self.linear15(hidden_states)
        hidden_states = self.dropout16(hidden_states)
        hidden_states += skip15
        skip16 = hidden_states #skip to self.dropout17

        hidden_states = self.layernorm16(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention8(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout17(hidden_states)
        hidden_states += skip16
        skip17 = hidden_states #skip to self.dropout18

        hidden_states = self.layernorm17(hidden_states)
        hidden_states = self.linear16(hidden_states)
        hidden_states = self.gelu8(hidden_states)
        hidden_states = self.linear17(hidden_states)
        hidden_states = self.dropout18(hidden_states)
        hidden_states += skip17
        skip18 = hidden_states #skip to self.dropout19

        hidden_states = self.layernorm18(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention9(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout19(hidden_states)
        hidden_states += skip18
        skip19 = hidden_states #skip to self.dropout20

        hidden_states = self.layernorm19(hidden_states)
        hidden_states = self.linear18(hidden_states)
        hidden_states = self.gelu9(hidden_states)
        hidden_states = self.linear19(hidden_states)
        hidden_states = self.dropout20(hidden_states)
        hidden_states += skip19
        skip20 = hidden_states #skip to self.dropout21

        hidden_states = self.layernorm20(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention10(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout21(hidden_states)
        hidden_states += skip20
        skip21 = hidden_states #skip to self.dropout22

        hidden_states = self.layernorm21(hidden_states)
        hidden_states = self.linear20(hidden_states)
        hidden_states = self.gelu10(hidden_states)
        hidden_states = self.linear21(hidden_states)
        hidden_states = self.dropout22(hidden_states)
        hidden_states += skip21
        skip22 = hidden_states #skip to self.dropout23

        hidden_states = self.layernorm22(hidden_states)
        attn_output = self.maskedautomaticsparsemultiheadattention11(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)
        hidden_states = attn_output[0]
        outputs = attn_output[1:]
        hidden_states = self.dropout23(hidden_states)
        hidden_states += skip22
        skip23 = hidden_states #skip to self.dropout24

        hidden_states = self.layernorm23(hidden_states)
        hidden_states = self.linear22(hidden_states)
        hidden_states = self.gelu11(hidden_states)
        hidden_states = self.linear23(hidden_states)
        hidden_states = self.dropout24(hidden_states)
        hidden_states += skip23
        hidden_states = self.layernorm24(hidden_states)


        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs

class GPT2Medium(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(GPT2Medium, self).__init__()
        #78173
        #encoder layers

        self.gpt2_small_1 = GPT2Small(config)
        self.gpt2_small_2 = GPT2Small(config)



    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        #encoder forward

        hidden_states, _ = self.gpt2_small_1(hidden_states, 
                                     layer_past,
                                    attention_mask,
                                    head_mask,
                                    encoder_hidden_states,
                                    encoder_attention_mask,
                                    use_cache,
                                    output_attentions)
        
        

        output = self.gpt2_small_2(hidden_states, 
                                layer_past,
                            attention_mask,
                            head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            use_cache,
                            output_attentions) 
        return output


class GPT2Large(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(GPT2Large, self).__init__()
        #78173
        #encoder layers
        self.dropout0 = nn.Dropout(config.resid_pdrop)
        self.layernorm0 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention0 = SkipformerAttention(config, layer_idx=0, embed_dim=1280,  num_heads=20)
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.layernorm1 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear0 = nn.Linear(1280, 3072)
        self.gelu0 = nn.GELU()
        self.linear1 = nn.Linear(3072, 1280)
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.layernorm2 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention1 = SkipformerAttention(config, layer_idx=1, embed_dim=1280,  num_heads=20)
        self.dropout3 = nn.Dropout(config.resid_pdrop)
        self.layernorm3 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear2 = nn.Linear(1280, 3072)
        self.gelu1 = nn.GELU()
        self.linear3 = nn.Linear(3072, 1280)
        self.dropout4 = nn.Dropout(config.resid_pdrop)
        self.layernorm4 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention2 = SkipformerAttention(config, layer_idx=2, embed_dim=1280,  num_heads=20)
        self.dropout5 = nn.Dropout(config.resid_pdrop)
        self.layernorm5 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear4 = nn.Linear(1280, 3072)
        self.gelu2 = nn.GELU()
        self.linear5 = nn.Linear(3072, 1280)
        self.dropout6 = nn.Dropout(config.resid_pdrop)
        self.layernorm6 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention3 = SkipformerAttention(config, layer_idx=3, embed_dim=1280,  num_heads=20)
        self.dropout7 = nn.Dropout(config.resid_pdrop)
        self.layernorm7 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear6 = nn.Linear(1280, 3072)
        self.gelu3 = nn.GELU()
        self.linear7 = nn.Linear(3072, 1280)
        self.dropout8 = nn.Dropout(config.resid_pdrop)
        self.layernorm8 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention4 = SkipformerAttention(config, layer_idx=4, embed_dim=1280,  num_heads=20)
        self.dropout9 = nn.Dropout(config.resid_pdrop)
        self.layernorm9 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear8 = nn.Linear(1280, 3072)
        self.gelu4 = nn.GELU()
        self.linear9 = nn.Linear(3072, 1280)
        self.dropout10 = nn.Dropout(config.resid_pdrop)
        self.layernorm10 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention5 = SkipformerAttention(config, layer_idx=5, embed_dim=1280,  num_heads=20)
        self.dropout11 = nn.Dropout(config.resid_pdrop)
        self.layernorm11 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear10 = nn.Linear(1280, 3072)
        self.gelu5 = nn.GELU()
        self.linear11 = nn.Linear(3072, 1280)
        self.dropout12 = nn.Dropout(config.resid_pdrop)
        self.layernorm12 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention6 = SkipformerAttention(config, layer_idx=6, embed_dim=1280,  num_heads=20)
        self.dropout13 = nn.Dropout(config.resid_pdrop)
        self.layernorm13 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear12 = nn.Linear(1280, 3072)
        self.gelu6 = nn.GELU()
        self.linear13 = nn.Linear(3072, 1280)
        self.dropout14 = nn.Dropout(config.resid_pdrop)
        self.layernorm14 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention7 = SkipformerAttention(config, layer_idx=7, embed_dim=1280,  num_heads=20)
        self.dropout15 = nn.Dropout(config.resid_pdrop)
        self.layernorm15 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear14 = nn.Linear(1280, 3072)
        self.gelu7 = nn.GELU()
        self.linear15 = nn.Linear(3072, 1280)
        self.dropout16 = nn.Dropout(config.resid_pdrop)
        self.layernorm16 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention8 = SkipformerAttention(config, layer_idx=8, embed_dim=1280,  num_heads=20)
        self.dropout17 = nn.Dropout(config.resid_pdrop)
        self.layernorm17 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear16 = nn.Linear(1280, 3072)
        self.gelu8 = nn.GELU()
        self.linear17 = nn.Linear(3072, 1280)
        self.dropout18 = nn.Dropout(config.resid_pdrop)
        self.layernorm18 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention9 = SkipformerAttention(config, layer_idx=9, embed_dim=1280,  num_heads=20)
        self.dropout19 = nn.Dropout(config.resid_pdrop)
        self.layernorm19 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear18 = nn.Linear(1280, 3072)
        self.gelu9 = nn.GELU()
        self.linear19 = nn.Linear(3072, 1280)
        self.dropout20 = nn.Dropout(config.resid_pdrop)
        self.layernorm20 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention10 = SkipformerAttention(config, layer_idx=10, embed_dim=1280,  num_heads=20)
        self.dropout21 = nn.Dropout(config.resid_pdrop)
        self.layernorm21 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear20 = nn.Linear(1280, 3072)
        self.gelu10 = nn.GELU()
        self.linear21 = nn.Linear(3072, 1280)
        self.dropout22 = nn.Dropout(config.resid_pdrop)
        self.layernorm22 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention11 = SkipformerAttention(config, layer_idx=11, embed_dim=1280,  num_heads=20)
        self.dropout23 = nn.Dropout(config.resid_pdrop)
        self.layernorm23 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)
        self.linear22 = nn.Linear(1280, 3072)
        self.gelu11 = nn.GELU()
        self.linear23 = nn.Linear(3072, 1280)
        self.dropout24 = nn.Dropout(config.resid_pdrop)
        self.layernorm24 = nn.LayerNorm(1280, eps=config.layer_norm_epsilon)


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        #encoder forward
        hidden_states = self.dropout0(hidden_states)
        skip0 = hidden_states #skip to self.dropout1

        hidden_states = self.layernorm0(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention0(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout1(hidden_states)
        hidden_states += skip0
        skip1 = hidden_states #skip to self.dropout2

        hidden_states = self.layernorm1(hidden_states)
        hidden_states = self.linear0(hidden_states)
        hidden_states = self.gelu0(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states += skip1
        skip2 = hidden_states #skip to self.dropout3

        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention1(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout3(hidden_states)
        hidden_states += skip2
        skip3 = hidden_states #skip to self.dropout4

        hidden_states = self.layernorm3(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.gelu1(hidden_states)
        hidden_states = self.linear3(hidden_states)
        hidden_states = self.dropout4(hidden_states)
        hidden_states += skip3
        skip4 = hidden_states #skip to self.dropout5

        hidden_states = self.layernorm4(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention2(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout5(hidden_states)
        hidden_states += skip4
        skip5 = hidden_states #skip to self.dropout6

        hidden_states = self.layernorm5(hidden_states)
        hidden_states = self.linear4(hidden_states)
        hidden_states = self.gelu2(hidden_states)
        hidden_states = self.linear5(hidden_states)
        hidden_states = self.dropout6(hidden_states)
        hidden_states += skip5
        skip6 = hidden_states #skip to self.dropout7

        hidden_states = self.layernorm6(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention3(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout7(hidden_states)
        hidden_states += skip6
        skip7 = hidden_states #skip to self.dropout8

        hidden_states = self.layernorm7(hidden_states)
        hidden_states = self.linear6(hidden_states)
        hidden_states = self.gelu3(hidden_states)
        hidden_states = self.linear7(hidden_states)
        hidden_states = self.dropout8(hidden_states)
        hidden_states += skip7
        skip8 = hidden_states #skip to self.dropout9

        hidden_states = self.layernorm8(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention4(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout9(hidden_states)
        hidden_states += skip8
        skip9 = hidden_states #skip to self.dropout10

        hidden_states = self.layernorm9(hidden_states)
        hidden_states = self.linear8(hidden_states)
        hidden_states = self.gelu4(hidden_states)
        hidden_states = self.linear9(hidden_states)
        hidden_states = self.dropout10(hidden_states)
        hidden_states += skip9
        skip10 = hidden_states #skip to self.dropout11

        hidden_states = self.layernorm10(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention5(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout11(hidden_states)
        hidden_states += skip10
        skip11 = hidden_states #skip to self.dropout12

        hidden_states = self.layernorm11(hidden_states)
        hidden_states = self.linear10(hidden_states)
        hidden_states = self.gelu5(hidden_states)
        hidden_states = self.linear11(hidden_states)
        hidden_states = self.dropout12(hidden_states)
        hidden_states += skip11
        skip12 = hidden_states #skip to self.dropout13

        hidden_states = self.layernorm12(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention6(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout13(hidden_states)
        hidden_states += skip12
        skip13 = hidden_states #skip to self.dropout14

        hidden_states = self.layernorm13(hidden_states)
        hidden_states = self.linear12(hidden_states)
        hidden_states = self.gelu6(hidden_states)
        hidden_states = self.linear13(hidden_states)
        hidden_states = self.dropout14(hidden_states)
        hidden_states += skip13
        skip14 = hidden_states #skip to self.dropout15

        hidden_states = self.layernorm14(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention7(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout15(hidden_states)
        hidden_states += skip14
        skip15 = hidden_states #skip to self.dropout16

        hidden_states = self.layernorm15(hidden_states)
        hidden_states = self.linear14(hidden_states)
        hidden_states = self.gelu7(hidden_states)
        hidden_states = self.linear15(hidden_states)
        hidden_states = self.dropout16(hidden_states)
        hidden_states += skip15
        skip16 = hidden_states #skip to self.dropout17

        hidden_states = self.layernorm16(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention8(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout17(hidden_states)
        hidden_states += skip16
        skip17 = hidden_states #skip to self.dropout18

        hidden_states = self.layernorm17(hidden_states)
        hidden_states = self.linear16(hidden_states)
        hidden_states = self.gelu8(hidden_states)
        hidden_states = self.linear17(hidden_states)
        hidden_states = self.dropout18(hidden_states)
        hidden_states += skip17
        skip18 = hidden_states #skip to self.dropout19

        hidden_states = self.layernorm18(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention9(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout19(hidden_states)
        hidden_states += skip18
        skip19 = hidden_states #skip to self.dropout20

        hidden_states = self.layernorm19(hidden_states)
        hidden_states = self.linear18(hidden_states)
        hidden_states = self.gelu9(hidden_states)
        hidden_states = self.linear19(hidden_states)
        hidden_states = self.dropout20(hidden_states)
        hidden_states += skip19
        skip20 = hidden_states #skip to self.dropout21

        hidden_states = self.layernorm20(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention10(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout21(hidden_states)
        hidden_states += skip20
        skip21 = hidden_states #skip to self.dropout22

        hidden_states = self.layernorm21(hidden_states)
        hidden_states = self.linear20(hidden_states)
        hidden_states = self.gelu10(hidden_states)
        hidden_states = self.linear21(hidden_states)
        hidden_states = self.dropout22(hidden_states)
        hidden_states += skip21
        skip22 = hidden_states #skip to self.dropout23

        hidden_states = self.layernorm22(hidden_states)
        attn_output = self.maskedautomaticsparsemultiheadattention11(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)
        hidden_states = attn_output[0]
        outputs = attn_output[1:]
        hidden_states = self.dropout23(hidden_states)
        hidden_states += skip22
        skip23 = hidden_states #skip to self.dropout24

        hidden_states = self.layernorm23(hidden_states)
        hidden_states = self.linear22(hidden_states)
        hidden_states = self.gelu11(hidden_states)
        hidden_states = self.linear23(hidden_states)
        hidden_states = self.dropout24(hidden_states)
        hidden_states += skip23
        hidden_states = self.layernorm24(hidden_states)


        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


class SkipformerA(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(SkipformerA, self).__init__()
        #78914
        #encoder layers
        self.dropout0 = nn.Dropout(config.resid_pdrop)
        self.layernorm0 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention0 = SkipformerAttention(config, layer_idx=0, embed_dim=768,  num_heads=12)
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.layernorm1 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear0 = nn.Linear(768, 384)
        self.relu0 = nn.ReLU()
        self.linear1 = nn.Linear(384, 768)
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.maskedautomaticsparsemultiheadattention1 = SkipformerAttention(config, layer_idx=1, embed_dim=768,  num_heads=12)
        self.dropout3 = nn.Dropout(config.resid_pdrop)
        self.layernorm2 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear2 = nn.Linear(768, 3072)
        self.gelu0 = nn.GELU()
        self.linear3 = nn.Linear(3072, 768)
        self.dropout4 = nn.Dropout(config.resid_pdrop)
        self.layernorm3 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear4 = nn.Linear(768, 768)
        self.gelu1 = nn.GELU()
        self.layernorm4 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.relu1 = nn.ReLU()
        self.maskedautomaticsparsemultiheadattention2 = SkipformerAttention(config, layer_idx=2, embed_dim=768,  num_heads=12)
        self.dropout5 = nn.Dropout(config.resid_pdrop)
        self.layernorm5 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.gelu2 = nn.GELU()
        self.layernorm6 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention3 = SkipformerAttention(config, layer_idx=3, embed_dim=768,  num_heads=12)
        self.dropout6 = nn.Dropout(config.resid_pdrop)
        self.layernorm7 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear5 = nn.Linear(768, 3456)
        self.gelu3 = nn.GELU()
        self.linear6 = nn.Linear(3456, 768)
        self.dropout7 = nn.Dropout(config.resid_pdrop)
        self.layernorm8 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear7 = nn.Linear(768, 2880)
        self.gelu4 = nn.GELU()
        self.linear8 = nn.Linear(2880, 768)
        self.dropout8 = nn.Dropout(config.resid_pdrop)
        self.relu2 = nn.ReLU()
        self.layernorm9 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention4 = SkipformerAttention(config, layer_idx=4, embed_dim=768,  num_heads=12)
        self.dropout9 = nn.Dropout(config.resid_pdrop)
        self.layernorm10 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.layernorm11 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear9 = nn.Linear(768, 3072)
        self.gelu5 = nn.GELU()
        self.linear10 = nn.Linear(3072, 768)
        self.dropout10 = nn.Dropout(config.resid_pdrop)
        self.layernorm12 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention5 = SkipformerAttention(config, layer_idx=5, embed_dim=768,  num_heads=12)
        self.dropout11 = nn.Dropout(config.resid_pdrop)
        self.layernorm13 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.layernorm14 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear11 = nn.Linear(768, 1344)
        self.gelu6 = nn.GELU()
        self.linear12 = nn.Linear(1344, 768)
        self.dropout12 = nn.Dropout(config.resid_pdrop)
        self.layernorm15 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.gelu7 = nn.GELU()
        self.relusquared0 = ReLUSquared()
        self.layernorm16 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention6 = SkipformerAttention(config, layer_idx=6, embed_dim=768,  num_heads=12)
        self.dropout13 = nn.Dropout(config.resid_pdrop)
        self.layernorm17 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.relu3 = nn.ReLU()
        self.linear13 = nn.Linear(768, 2304)
        self.gelu8 = nn.GELU()
        self.linear14 = nn.Linear(2304, 768)
        self.dropout14 = nn.Dropout(config.resid_pdrop)
        self.layernorm18 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention7 = SkipformerAttention(config, layer_idx=7, embed_dim=768,  num_heads=12)
        self.dropout15 = nn.Dropout(config.resid_pdrop)
        self.gelu9 = nn.GELU()
        self.layernorm19 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear15 = nn.Linear(768, 1344)
        self.gelu10 = nn.GELU()
        self.linear16 = nn.Linear(1344, 768)
        self.dropout16 = nn.Dropout(config.resid_pdrop)
        self.layernorm20 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention8 = SkipformerAttention(config, layer_idx=8, embed_dim=768,  num_heads=12)
        self.dropout17 = nn.Dropout(config.resid_pdrop)
        self.layernorm21 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear17 = nn.Linear(768, 3072)
        self.gelu11 = nn.GELU()
        self.linear18 = nn.Linear(3072, 768)
        self.dropout18 = nn.Dropout(config.resid_pdrop)
        self.layernorm22 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention9 = SkipformerAttention(config, layer_idx=9, embed_dim=768,  num_heads=12)
        self.dropout19 = nn.Dropout(config.resid_pdrop)
        self.layernorm23 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear19 = nn.Linear(768, 3072)
        self.gelu12 = nn.GELU()
        self.linear20 = nn.Linear(3072, 768)
        self.dropout20 = nn.Dropout(config.resid_pdrop)
        self.layernorm24 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        #encoder forward
        hidden_states = self.dropout0(hidden_states)
        skip0 = hidden_states #skip to self.dropout1

        hidden_states = self.layernorm0(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention0(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout1(hidden_states)
        hidden_states += skip0
        hidden_states = self.layernorm1(hidden_states)
        hidden_states = self.linear0(hidden_states)
        hidden_states = self.relu0(hidden_states)
        hidden_states = self.linear1(hidden_states)
        skip1 = hidden_states #skip to self.dropout11

        hidden_states = self.dropout2(hidden_states)
        skip2 = hidden_states #skip to self.dropout3

        hidden_states = self.maskedautomaticsparsemultiheadattention1(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout3(hidden_states)
        hidden_states += skip2
        skip3 = hidden_states #skip to self.dropout4

        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.gelu0(hidden_states)
        hidden_states = self.linear3(hidden_states)
        hidden_states = self.dropout4(hidden_states)
        hidden_states += skip3
        hidden_states = self.layernorm3(hidden_states)
        hidden_states = self.linear4(hidden_states)
        hidden_states = self.gelu1(hidden_states)
        skip4 = hidden_states #skip to self.dropout5

        hidden_states = self.layernorm4(hidden_states)
        hidden_states = self.relu1(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention2(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout5(hidden_states)
        hidden_states += skip4
        hidden_states = self.layernorm5(hidden_states)
        hidden_states = self.gelu2(hidden_states)
        skip5 = hidden_states #skip to self.dropout6

        skip6 = hidden_states #skip to self.dropout7

        hidden_states = self.layernorm6(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention3(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout6(hidden_states)
        hidden_states += skip5
        skip7 = hidden_states #skip to self.maskedautomaticsparsemultiheadattention5

        skip8 = hidden_states #skip to self.dropout7

        hidden_states = self.layernorm7(hidden_states)
        hidden_states = self.linear5(hidden_states)
        hidden_states = self.gelu3(hidden_states)
        hidden_states = self.linear6(hidden_states)
        hidden_states = self.dropout7(hidden_states)
        hidden_states += skip6 + skip8
        hidden_states = self.layernorm8(hidden_states)
        hidden_states = self.linear7(hidden_states)
        hidden_states = self.gelu4(hidden_states)
        hidden_states = self.linear8(hidden_states)
        hidden_states = self.dropout8(hidden_states)
        hidden_states = self.relu2(hidden_states)
        skip9 = hidden_states #skip to self.dropout9

        hidden_states = self.layernorm9(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention4(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout9(hidden_states)
        hidden_states += skip9
        hidden_states = self.layernorm10(hidden_states)
        skip10 = hidden_states #skip to self.dropout10

        hidden_states = self.layernorm11(hidden_states)
        hidden_states = self.linear9(hidden_states)
        hidden_states = self.gelu5(hidden_states)
        hidden_states = self.linear10(hidden_states)
        hidden_states = self.dropout10(hidden_states)
        hidden_states += skip10
        hidden_states = self.layernorm12(hidden_states)
        skip11 = hidden_states #skip to decoder skip18

        hidden_states = self.maskedautomaticsparsemultiheadattention5(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states += skip7
        hidden_states = self.dropout11(hidden_states)
        hidden_states += skip1
        hidden_states = self.layernorm13(hidden_states)
        skip12 = hidden_states #skip to self.dropout12

        skip13 = hidden_states #skip to decoder skip14

        hidden_states = self.layernorm14(hidden_states)
        hidden_states = self.linear11(hidden_states)
        hidden_states = self.gelu6(hidden_states)
        hidden_states = self.linear12(hidden_states)
        hidden_states = self.dropout12(hidden_states)
        hidden_states += skip12
        skip14 = hidden_states #skip to decoder skip17
        hidden_states += skip13

        hidden_states = self.layernorm15(hidden_states)
        hidden_states = self.gelu7(hidden_states)
        skip15 = hidden_states #skip to self.dropout13

        hidden_states = self.relusquared0(hidden_states)
        skip16 = hidden_states #skip to self.layernorm23

        hidden_states = self.layernorm16(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention6(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout13(hidden_states)
        hidden_states += skip15
        skip17 = hidden_states #skip to self.dropout14
        hidden_states += skip14

        hidden_states = self.layernorm17(hidden_states)
        hidden_states = self.relu3(hidden_states)
        hidden_states = self.linear13(hidden_states)
        hidden_states = self.gelu8(hidden_states)
        hidden_states = self.linear14(hidden_states)
        hidden_states = self.dropout14(hidden_states)
        hidden_states += skip17
        skip18 = hidden_states #skip to self.dropout15
        hidden_states += skip11

        hidden_states = self.layernorm18(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention7(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout15(hidden_states)
        hidden_states += skip18
        hidden_states = self.gelu9(hidden_states)
        skip19 = hidden_states #skip to self.dropout16

        hidden_states = self.layernorm19(hidden_states)
        hidden_states = self.linear15(hidden_states)
        hidden_states = self.gelu10(hidden_states)
        hidden_states = self.linear16(hidden_states)
        hidden_states = self.dropout16(hidden_states)
        hidden_states += skip19
        skip20 = hidden_states #skip to self.dropout17

        hidden_states = self.layernorm20(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention8(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout17(hidden_states)
        hidden_states += skip20
        skip21 = hidden_states #skip to self.dropout18

        hidden_states = self.layernorm21(hidden_states)
        hidden_states = self.linear17(hidden_states)
        hidden_states = self.gelu11(hidden_states)
        hidden_states = self.linear18(hidden_states)
        hidden_states = self.dropout18(hidden_states)
        hidden_states += skip21
        skip22 = hidden_states #skip to self.dropout19

        hidden_states = self.layernorm22(hidden_states)
        attn_output = self.maskedautomaticsparsemultiheadattention9(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)
        hidden_states = attn_output[0]
        outputs = attn_output[1:]
        hidden_states = self.dropout19(hidden_states)
        hidden_states += skip22
        skip23 = hidden_states #skip to self.dropout20

        hidden_states = self.layernorm23(hidden_states)
        hidden_states += skip16
        hidden_states = self.linear19(hidden_states)
        hidden_states = self.gelu12(hidden_states)
        hidden_states = self.linear20(hidden_states)
        hidden_states = self.dropout20(hidden_states)
        hidden_states += skip23
        hidden_states = self.layernorm24(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs

class SkipformerAMedium(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(SkipformerAMedium, self).__init__()

        self.skipformer_a_1 = SkipformerA(config)
        self.skipformer_a_2 =SkipformerA(config)



    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        hidden_states, _ = self.skipformer_a_1(hidden_states, 
                                     layer_past,
                                    attention_mask,
                                    head_mask,
                                    encoder_hidden_states,
                                    encoder_attention_mask,
                                    use_cache,
                                    output_attentions)
        
        

        output = self.skipformer_a_2(hidden_states, 
                                layer_past,
                            attention_mask,
                            head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            use_cache,
                            output_attentions) 
        return output


class SkipformerB(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(SkipformerB, self).__init__()
        #80210
        #encoder layers
        #in 1152
        self.dropout0 = nn.Dropout(config.resid_pdrop)
        self.layernorm0 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention0 = SkipformerAttention(config, layer_idx=0, embed_dim=768,  num_heads=12)
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.layernorm1 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear0 = nn.Linear(768, 2304)
        self.gelu0 = nn.GELU()
        self.relu0 = nn.ReLU()
        self.linear1 = nn.Linear(2304, 768)
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.relu1 = nn.ReLU()
        self.maskedautomaticsparsemultiheadattention1 = SkipformerAttention(config, layer_idx=1, embed_dim=768,  num_heads=12)
        self.dropout3 = nn.Dropout(config.resid_pdrop)
        self.linear2 = nn.Linear(768, 3072)
        self.gelu1 = nn.GELU()
        self.linear3 = nn.Linear(3072, 768)
        self.dropout4 = nn.Dropout(config.resid_pdrop)
        self.layernorm2 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear4 = nn.Linear(768, 768)
        self.layernorm3 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.relu2 = nn.ReLU()
        self.layernorm4 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.gelu2 = nn.GELU()
        self.layernorm5 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention2 = SkipformerAttention(config, layer_idx=2, embed_dim=768,  num_heads=12)
        self.dropout5 = nn.Dropout(config.resid_pdrop)
        self.layernorm6 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.gelu3 = nn.GELU()
        self.linear5 = nn.Linear(768, 768)
        self.dropout6 = nn.Dropout(config.resid_pdrop)
        self.layernorm7 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear6 = nn.Linear(768, 1344)
        self.gelu4 = nn.GELU()
        self.linear7 = nn.Linear(1344, 768)
        self.dropout7 = nn.Dropout(config.resid_pdrop)
        self.layernorm8 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention3 = SkipformerAttention(config, layer_idx=3, embed_dim=768,  num_heads=12)
        self.dropout8 = nn.Dropout(config.resid_pdrop)
        self.layernorm9 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.gelu5 = nn.GELU()
        self.linear8 = nn.Linear(768, 768)
        self.dropout9 = nn.Dropout(config.resid_pdrop)
        self.layernorm10 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention4 = SkipformerAttention(config, layer_idx=4, embed_dim=768,  num_heads=12)
        self.dropout10 = nn.Dropout(config.resid_pdrop)
        self.layernorm11 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.relu3 = nn.ReLU()
        self.layernorm12 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear9 = nn.Linear(768, 960)
        self.gelu6 = nn.GELU()
        self.linear10 = nn.Linear(960, 768)
        self.dropout11 = nn.Dropout(config.resid_pdrop)
        self.layernorm13 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.layernorm14 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.relusquared0 = ReLUSquared()
        self.layernorm15 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention5 = SkipformerAttention(config, layer_idx=5, embed_dim=768,  num_heads=12)
        self.dropout12 = nn.Dropout(config.resid_pdrop)
        self.layernorm16 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear11 = nn.Linear(768, 2304)
        self.gelu7 = nn.GELU()
        self.linear12 = nn.Linear(2304, 768)
        self.dropout13 = nn.Dropout(config.resid_pdrop)
        self.layernorm17 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention6 = SkipformerAttention(config, layer_idx=6, embed_dim=768,  num_heads=12)
        self.dropout14 = nn.Dropout(config.resid_pdrop)
        self.gelu8 = nn.GELU()
        self.relusquared1 = ReLUSquared()
        self.layernorm18 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear13 = nn.Linear(768, 1344)
        self.gelu9 = nn.GELU()
        self.linear14 = nn.Linear(1344, 768)
        self.dropout15 = nn.Dropout(config.resid_pdrop)
        self.layernorm19 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention7 = SkipformerAttention(config, layer_idx=7, embed_dim=768,  num_heads=12)
        self.dropout16 = nn.Dropout(config.resid_pdrop)
        self.layernorm20 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear15 = nn.Linear(768, 3072)
        self.gelu10 = nn.GELU()
        self.linear16 = nn.Linear(3072, 768)
        self.dropout17 = nn.Dropout(config.resid_pdrop)
        self.layernorm21 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.maskedautomaticsparsemultiheadattention8 = SkipformerAttention(config, layer_idx=8, embed_dim=768,  num_heads=12)
        self.dropout18 = nn.Dropout(config.resid_pdrop)
        self.layernorm22 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)
        self.linear17 = nn.Linear(768, 3072)
        self.layernorm23 = nn.LayerNorm(3072, eps=config.layer_norm_epsilon)
        self.gelu11 = nn.GELU()
        self.linear18 = nn.Linear(3072, 768)
        self.dropout19 = nn.Dropout(config.resid_pdrop)
        self.layernorm24 = nn.LayerNorm(768, eps=config.layer_norm_epsilon)


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        #encoder forward
        hidden_states = self.dropout0(hidden_states)
        skip0 = hidden_states #skip to self.dropout1

        hidden_states = self.layernorm0(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention0(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout1(hidden_states)
        hidden_states += skip0
        hidden_states = self.layernorm1(hidden_states)
        hidden_states = self.linear0(hidden_states)
        hidden_states = self.gelu0(hidden_states)
        hidden_states = self.relu0(hidden_states)
        hidden_states = self.linear1(hidden_states)
        skip1 = hidden_states #skip to self.dropout10

        hidden_states = self.dropout2(hidden_states)
        hidden_states = self.relu1(hidden_states)
        skip2 = hidden_states #skip to self.dropout3

        hidden_states = self.maskedautomaticsparsemultiheadattention1(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        skip3 = hidden_states #skip to self.maskedautomaticsparsemultiheadattention3

        hidden_states = self.dropout3(hidden_states)
        hidden_states += skip2
        skip4 = hidden_states #skip to self.dropout4

        hidden_states = self.linear2(hidden_states)
        hidden_states = self.gelu1(hidden_states)
        hidden_states = self.linear3(hidden_states)
        hidden_states = self.dropout4(hidden_states)
        hidden_states += skip4
        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.linear4(hidden_states)
        hidden_states = self.layernorm3(hidden_states)
        hidden_states = self.relu2(hidden_states)
        hidden_states = self.layernorm4(hidden_states)
        hidden_states = self.gelu2(hidden_states)
        skip5 = hidden_states #skip to self.dropout5

        skip6 = hidden_states #skip to self.dropout6

        hidden_states = self.layernorm5(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention2(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout5(hidden_states)
        hidden_states += skip5
        hidden_states = self.layernorm6(hidden_states)
        hidden_states = self.gelu3(hidden_states)
        hidden_states = self.linear5(hidden_states)
        hidden_states = self.dropout6(hidden_states)
        hidden_states += skip6
        hidden_states = self.layernorm7(hidden_states)
        hidden_states = self.linear6(hidden_states)
        hidden_states = self.gelu4(hidden_states)
        hidden_states = self.linear7(hidden_states)
        hidden_states = self.dropout7(hidden_states)
        skip7 = hidden_states #skip to self.dropout8

        hidden_states = self.layernorm8(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention3(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states += skip3
        hidden_states = self.dropout8(hidden_states)
        hidden_states += skip7
        skip8 = hidden_states #skip to self.dropout9

        hidden_states = self.layernorm9(hidden_states)
        hidden_states = self.gelu5(hidden_states)
        hidden_states = self.linear8(hidden_states)
        hidden_states = self.dropout9(hidden_states)
        hidden_states += skip8
        hidden_states = self.layernorm10(hidden_states)
        skip9 = hidden_states #skip to self.layernorm19

        skip10 = hidden_states #skip to decoder skip16

        hidden_states = self.maskedautomaticsparsemultiheadattention4(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout10(hidden_states)
        hidden_states += skip1
        hidden_states = self.layernorm11(hidden_states)
        skip11 = hidden_states #skip to self.dropout11

        hidden_states = self.relu3(hidden_states)
        hidden_states = self.layernorm12(hidden_states)
        hidden_states = self.linear9(hidden_states)
        hidden_states = self.gelu6(hidden_states)
        hidden_states = self.linear10(hidden_states)
        hidden_states = self.dropout11(hidden_states)
        hidden_states += skip11
        skip12 = hidden_states #skip to decoder skip15

        hidden_states = self.layernorm13(hidden_states)
        skip13 = hidden_states #skip to self.dropout12

        # hidden_states = self.layernorm14(hidden_states)
        hidden_states = self.relusquared0(hidden_states)
        skip14 = hidden_states #skip to self.layernorm22

        hidden_states = self.layernorm15(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention5(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout12(hidden_states)
        hidden_states += skip13
        skip15 = hidden_states #skip to self.dropout13

        hidden_states = self.layernorm16(hidden_states)
        hidden_states = self.linear11(hidden_states)
        hidden_states = self.gelu7(hidden_states)
        hidden_states = self.linear12(hidden_states)
        hidden_states = self.dropout13(hidden_states)
        hidden_states += skip15
        hidden_states += skip12
        skip16 = hidden_states #skip to self.dropout14

        hidden_states = self.layernorm17(hidden_states)
        hidden_states = self.maskedautomaticsparsemultiheadattention6(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout14(hidden_states)
        hidden_states += skip16
        hidden_states += skip10
        hidden_states = self.gelu8(hidden_states)
        skip17 = hidden_states #skip to self.dropout15

        hidden_states = self.relusquared1(hidden_states)
        hidden_states = self.layernorm18(hidden_states)
        hidden_states = self.linear13(hidden_states)
        hidden_states = self.gelu9(hidden_states)
        hidden_states = self.linear14(hidden_states)
        hidden_states = self.dropout15(hidden_states)
        hidden_states += skip17
        skip18 = hidden_states #skip to self.dropout16

        hidden_states = self.layernorm19(hidden_states)
        hidden_states += skip9
        skip19 = hidden_states #skip to self.dropout16

        hidden_states = self.maskedautomaticsparsemultiheadattention7(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)[0]
        hidden_states = self.dropout16(hidden_states)
        hidden_states += skip18 + skip19
        skip20 = hidden_states #skip to self.dropout17

        hidden_states = self.layernorm20(hidden_states)
        hidden_states = self.linear15(hidden_states)
        hidden_states = self.gelu10(hidden_states)
        hidden_states = self.linear16(hidden_states)
        hidden_states = self.dropout17(hidden_states)
        hidden_states += skip20
        skip21 = hidden_states #skip to self.dropout18

        hidden_states = self.layernorm21(hidden_states)
        skip22 = hidden_states #skip to self.dropout18

        attn_output = self.maskedautomaticsparsemultiheadattention8(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions,)
        hidden_states = attn_output[0]
        outputs = attn_output[1:]
        hidden_states = self.dropout18(hidden_states)
        hidden_states += skip21 + skip22
        skip23 = hidden_states #skip to self.dropout19

        hidden_states = self.layernorm22(hidden_states)
        hidden_states += skip14
        hidden_states = self.linear17(hidden_states)
        hidden_states = self.layernorm23(hidden_states)
        hidden_states = self.gelu11(hidden_states)
        hidden_states = self.linear18(hidden_states)
        hidden_states = self.dropout19(hidden_states)
        hidden_states += skip23
        hidden_states = self.layernorm24(hidden_states)


        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
