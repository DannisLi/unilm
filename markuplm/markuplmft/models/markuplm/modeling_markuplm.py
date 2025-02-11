# coding=utf-8
# Copyright 2018 The Microsoft Research Asia MarkupLM Team Authors and the HuggingFace Inc. team.
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
""" PyTorch MarkupLM model. """

import math
import os

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
import torch.nn.init as init

from transformers.activations import ACT2FN
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, \
    replace_return_docstrings
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from .configuration_markuplm import MarkupLMConfig

from typing import Optional, Union

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MarkupLMConfig"
_TOKENIZER_FOR_DOC = "MarkupLMTokenizer"

MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/markuplm-base",
    "microsoft/markuplm-large",
]

MarkupLMLayerNorm = torch.nn.LayerNorm


class XPathEmbeddings(nn.Module):
    """Construct the embddings from xpath -- tag and subscript"""

    # we drop tree-id in this version, as its info can be covered by xpath

    def __init__(self, config):
        super(XPathEmbeddings, self).__init__()
        self.max_depth = config.max_depth

        self.xpath_unitseq2_embeddings = nn.Linear(
            config.xpath_unit_hidden_size * self.max_depth, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.activation = nn.ReLU()
        self.xpath_unitseq2_inner = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, 4 * config.hidden_size)
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)

        self.xpath_tag_sub_embeddings = nn.ModuleList(
            [nn.Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size) for _ in
             range(self.max_depth)])

        self.xpath_subs_sub_embeddings = nn.ModuleList(
            [nn.Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size) for _ in
             range(self.max_depth)])

    def forward(self,
                xpath_tags_seq=None,
                xpath_subs_seq=None):
        xpath_tags_embeddings = []
        xpath_subs_embeddings = []

        for i in range(self.max_depth):
            xpath_tags_embeddings.append(self.xpath_tag_sub_embeddings[i](xpath_tags_seq[:, :, i]))
            xpath_subs_embeddings.append(self.xpath_subs_sub_embeddings[i](xpath_subs_seq[:, :, i]))

        xpath_tags_embeddings = torch.cat(xpath_tags_embeddings, dim=-1)
        xpath_subs_embeddings = torch.cat(xpath_subs_embeddings, dim=-1)

        xpath_embeddings = xpath_tags_embeddings + xpath_subs_embeddings

        xpath_embeddings = self.inner2emb(
            self.dropout(self.activation(self.xpath_unitseq2_inner(xpath_embeddings))))

        return xpath_embeddings


class MarkupLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(MarkupLMEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.max_depth = config.max_depth

        self.xpath_embeddings = XPathEmbeddings(config)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = MarkupLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
            self,
            input_ids=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx,
                                                                  past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # xpath seq prepare

        if xpath_tags_seq is None:
            xpath_tags_seq = 216 * torch.ones(tuple(list(input_shape) + [self.max_depth]), dtype=torch.long,
                                              device=device)

        if xpath_subs_seq is None:
            xpath_subs_seq = 1001 * torch.ones(tuple(list(input_shape) + [self.max_depth]), dtype=torch.long,
                                               device=device)
        # xpath seq prepare

        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        xpath_embeddings = self.xpath_embeddings(xpath_tags_seq,
                                                 xpath_subs_seq)
        embeddings = (
                words_embeddings
                + position_embeddings
                + token_type_embeddings
                + xpath_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->MarkupLM
class MarkupLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class MarkupLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->MarkupLM
class MarkupLMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler
class MarkupLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->MarkupLM
class MarkupLMPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->MarkupLM
class MarkupLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MarkupLMPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->MarkupLM
class MarkupLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MarkupLMLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MarkupLMSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,

    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MarkupLMModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class MarkupLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MarkupLMSelfAttention(config)
        self.output = MarkupLMSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MarkupLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MarkupLMAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = MarkupLMAttention(config)
        self.intermediate = MarkupLMIntermediate(config)
        self.output = MarkupLMOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MarkupLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MarkupLMLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class MarkupLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MarkupLMConfig
    pretrained_model_archive_map = MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "markuplm"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, MarkupLMLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        return super(MarkupLMPreTrainedModel, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


MARKUPLM_START_DOCSTRING = r"""
    The MarkupLM model was proposed in 
     ----- NOTHING!!!!!! -----

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.MarkupLMConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

MARKUPLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.MarkupLMTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        
                
        xpath_tags_seq (:obj:`torch.LongTensor` of shape :obj:`({0}, 50)`, `optional`):
            None
        
        xpath_subs_seq (:obj:`torch.LongTensor` of shape :obj:`({0}, 50)`, `optional`):
            None
        
        tree_index_seq (:obj:`torch.LongTensor` of shape :obj:`({0}, 50)`, `optional`):
            None

        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``: ``0`` corresponds to a `sentence A` token, ``1`` corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``: :obj:`1`
            indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under
            returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned
            tensors for more detail.
        return_dict (:obj:`bool`, `optional`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


@add_start_docstrings(
    "The bare MarkupLM Model transformer outputting raw hidden-states without any specific head on top.",
    MARKUPLM_START_DOCSTRING,
)
class MarkupLMModel(MarkupLMPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super(MarkupLMModel, self).__init__(config)
        self.config = config

        self.embeddings = MarkupLMEmbeddings(config)
        self.encoder = MarkupLMEncoder(config)
        self.pooler = MarkupLMPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
    ):
        r"""
        Returns:

        Examples::

            No examples now !
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    MarkupLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MARKUPLM_START_DOCSTRING,
)
class MarkupLMForQuestionAnswering(MarkupLMPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.

        Returns:

        Examples:
            No example now !

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MarkupLMOnlyTokenClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.node_type_size)

    def forward(self, sequence_output):
        # sequence_output : (bs,seq_len,dim)
        sequence_output_x = self.dense(sequence_output)
        sequence_output_x = self.transform_act_fn(sequence_output_x)
        sequence_output_x = self.LayerNorm(sequence_output_x)
        output_res = self.decoder(sequence_output_x)
        # (bs,seq_len,node_type_size) here node_type_size is real+none
        return output_res


@add_start_docstrings("""MarkupLM Model with a `token_classification` head on top. """, MARKUPLM_START_DOCSTRING)
class MarkupLMForTokenClassification(MarkupLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        self.token_cls = MarkupLMOnlyTokenClassificationHead(config)
        self.init_weights()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

            xpath_tags_seq=None,
            xpath_subs_seq=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[-100, 0, ...,
            config.node_type_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.node_type_size]``
        Returns:

        Examples:
            No example now !

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.token_cls(sequence_output)  # (bs,seq,node_type_size)
        # pred_node_types = torch.argmax(prediction_scores,dim=2) # (bs,seq)

        token_cls_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            token_cls_loss = loss_fct(
                prediction_scores.view(-1, self.config.node_type_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((token_cls_loss,) + output) if token_cls_loss is not None else output

        return TokenClassifierOutput(
            loss=token_cls_loss,
            logits=prediction_scores,  # (bs,seq,node_type_size)
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def focal_loss(inputs, targets, alpha, gamma:float=0., ignore_index:int=-100):
    # inputs: (N, C)
    # targets: (N, )
    if gamma == 0.:
        return F.cross_entropy(inputs, targets, weight=alpha, ignore_index=ignore_index)
    else:
        # print ("before:", inputs.shape, targets.shape)
        case_ids = (targets != ignore_index).nonzero(as_tuple=True)[0]
        inputs = inputs[case_ids]
        targets = targets[case_ids]
        ce = F.cross_entropy(inputs, targets, weight=alpha, reduce=False)
        prob = torch.gather(F.softmax(inputs, dim=1), 1, targets.view(-1,1)).squeeze(dim=1)
        return torch.mean(ce * (1. - prob)**gamma)

#############################################################################################
# The following codes are written by Li Zimeng for applying MarkupLM to conduct HTML node classification.
# Node embeddings are generated by taking avergae of its token (output) embeddings.

class MarkupLMForNodeClassification(MarkupLMPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels

        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        self.classifiers = nn.ModuleList([self.build_classifier(config) for _ in range(config.num_hidden_layers+1)])
        self.register_buffer("class_weight", torch.tensor([1., 30.]))
        self.loss_fct = CrossEntropyLoss(weight=self.class_weight, ignore_index=-100)

        self.init_weights()
    

    def build_classifier(self, config):
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.num_labels),
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            num_nodes=None,
            node_spans=None,
            node_labels=None,
            query_span=None,
            output_attentions=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            max_num_nodes=128,
        ):
        '''
        num_nodes[bs]: the number of DOM nodes
        node_labels[bs*max_num_nodes]: the labels of DOM nodes, i.e. whether contain answers (exactly)
        node_spans[bs*max_num_nodes*2]: the boundaries of DOM nodes
        query_span[bs*2]: 
        '''

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,    # 返回每层的hidden states，以便让每层的表征过分类器
            return_dict=return_dict,
        )

        # [batch_size * num_layer * seq_len * dim]
        hidden_states = torch.stack(outputs.hidden_states, dim=0).transpose(0, 1)
        # hidden_states = outputs.hidden_states
        # max_num_nodes = node_labels.shape[1]
        batch_size = node_labels.shape[0]

        logits, loss = [], []
        query_rep = []
        node_reps = []
        for b in range(batch_size):
            # 当前case的query embeddings: [num_layers * dim]
            query_rep_case = hidden_states[b, :, query_span[b,0]:query_span[b,1]].mean(dim=1)
            query_rep.append(query_rep_case)
            # 当前case的节点数和spans
            num_nodes_case = num_nodes[b].item()
            node_spans_case = node_spans[b, :num_nodes_case]    # [num_nodes * 2]
            # 当前case的所有nodes的embedings: [num_layers * num_nodes * dim]
            node_reps_case = torch.stack([hidden_states[b, :, sp[0]:sp[1]].mean(dim=1) for sp in node_spans_case], dim=1)
            # pad to [num_layers * max_num_nodes * dim]
            # print (max_num_nodes, num_nodes_case, node_reps_case.shape)
            node_reps_case = F.pad(node_reps_case, (0, 0, 0, max_num_nodes-num_nodes_case, 0, 0), "constant", 0)
            # print (max_num_nodes, node_reps_case.shape)
            node_reps.append(node_reps_case)
        
        # query_rep: [batch_size * num_layers * dim] & node_reps: [batch_size * num_layers * max_num_nodes * dim]
        query_rep = torch.stack(query_rep, dim=0)
        node_reps = torch.stack(node_reps, dim=0)

        # cat query and node reps
        cls_inputs = torch.cat((query_rep.unsqueeze(2).repeat(1,1,max_num_nodes,1), node_reps), dim=3)

        logits, loss = [], []
        # 计算logits和loss
        for i in range(self.config.num_hidden_layers+1):
            # logits for layer: [batch_size, max_num_nodes, num_labels]
            preds = self.classifiers[i](cls_inputs[:,i])
            logits.append(preds)
            # loss for layer
            loss.append(self.loss_fct(preds.view(-1, self.num_labels), node_labels.view(-1)))
        
        logits = torch.stack(logits, dim=1)                # [bs*layers*max_num_nodes*num_labels]
        loss = torch.stack(loss, dim=0).unsqueeze(0)        # [1 * layers]

        return logits, loss


class MarkupLMForQuestionAnswering_true_removal(MarkupLMPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # qa output
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()


    def extend_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


    def forward(
            self,
            input_ids=None,
            attention_mask_1=None,
            attention_mask_2=None,    # after node removal
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask_1 is None:
            attention_mask_1 = torch.ones(input_shape, device=device)
        if attention_mask_2 is None:
            attention_mask_2 = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # set head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Build the forward prop
        # embedding layer
        hidden_states = self.markuplm.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # first six self-attention layers
        extend_attention_mask_1 = self.extend_attention_mask(attention_mask_1)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[:self.config.num_hidden_layers//2]):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask_1,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        # last six self-attention layers
        extend_attention_mask_2 = self.extend_attention_mask(attention_mask_2)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[self.config.num_hidden_layers//2:]):
            i += self.config.num_hidden_layers//2
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask_2,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # QA loss and train
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 去掉padding和remove的节点
        tmp = attention_mask_2.bool()
        start_logits[~tmp] = -1e6
        end_logits[~tmp] = -1e6
        del tmp

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 排除padding和removal位置的logits
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
            return  total_loss, start_logits, end_logits
        else:
            return start_logits, end_logits



'''
class MarkupLMForQuestionAnswering_node_removal(MarkupLMPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # node removal layer: this component will give a score for each node to measure how informative it is for answering the question.
        self.node_removal_layer = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),    # 0
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),      # 2
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),                       # 4
        )
        # qa output
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()


    def extend_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            query_span=None,
            node_spans=None,
            num_nodes=None,
            constraints=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # set head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Build the forward prop
        # embedding layer
        hidden_states = self.markuplm.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # first six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[:self.config.num_hidden_layers//2]):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        # node removal
        batch_size = node_spans.size(0)
        max_num_nodes = node_spans.size(1)
        query_rep = []
        node_reps = []
        for b in range(batch_size):
            # 当前case的query embeddings: [dim]
            query_rep.append(hidden_states[b, query_span[b,0]:query_span[b,1]].mean(dim=0))
            # 当前case的所有nodes的embedings: [max_num_nodes * dim]
            node_reps.append(
                torch.stack(
                    [hidden_states[b, sp[0]:sp[1]].mean(dim=0) for sp in node_spans[b,:num_nodes[b]]] + \
                    [torch.zeros(self.config.hidden_size, device=device) for _ in range(max_num_nodes - num_nodes[b])], dim=0))
        # query_rep: [batch_size * dim] & node_reps: [batch_size * max_num_nodes * dim]
        query_rep = torch.stack(query_rep, dim=0)
        node_reps = torch.stack(node_reps, dim=0)
        # query_node_reps: [batch_size * num_nodes * 2dim]
        query_node_reps = torch.cat(
            (query_rep.unsqueeze(1).repeat(1, max_num_nodes, 1), node_reps), dim=-1)
        # node removal layer [batch_size * max_num_nodes]
        node_scores = self.node_removal_layer(query_node_reps).squeeze(-1)
        removed_nodes = node_scores < 0
        # make mask
        removal_mask = torch.ones(input_shape, device=device)
        for i in range(batch_size):
            for j in range(num_nodes[i]):
                if removed_nodes[i,j]:
                    removal_mask[i, node_spans[i,j,0]:node_spans[i,j,1]] = 0
        attention_mask_after_removal = removal_mask * attention_mask
        # last six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask_after_removal)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[self.config.num_hidden_layers//2:]):
            i += self.config.num_hidden_layers//2
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # QA loss and train
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 去掉padding和remove的节点
        tmp = attention_mask_after_removal.bool()
        start_logits[~tmp] = -1e3
        end_logits[~tmp] = -1e3
        del tmp

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 排除padding和removal位置的logits
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            QA_loss = (start_loss + end_loss) / 2
            # total_loss = (start_loss + end_loss) / 2
            # node importance score regularization (tree)
            R_tree = 0.
            for b in range(batch_size):
                tmp = 0.
                for i in range(num_nodes[b]):
                    if constraints[b,i] >= 0:
                        tmp += F.relu(node_scores[b, i] - node_scores[b, constraints[b,i]])
                R_tree += tmp / num_nodes[b]
            R_tree /= batch_size
            # 添加length regularization: 1 - 删除的比例 = 保留的比例
            R_len = torch.mean(1. - torch.sum(1.-removal_mask, dim=1) / torch.sum(attention_mask, dim=1))

            total_loss = QA_loss + 1e-3 * R_len + 1e-1 * R_tree
        
            return  total_loss, QA_loss, R_len, R_tree
        else:
            return start_logits, end_logits
'''



class MarkupLMForQuestionAnswering_node_removal(MarkupLMPreTrainedModel):
    '''
    此版本用于derivative-free optimization
    '''

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # node removal layer: this component will give a score for each node to measure how informative it is for answering the question.
        self.node_removal_layer = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),       # 0
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),      # 3
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),                # 6
        )


        # qa output
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()


    def extend_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    

    def make_removal_mask(self, A, B, seq_len):
        '''
        A is a tensor with shape [batch size, num nodes], which entries are either 0 or 1.
        B is a nonnegative integer tensor with shape [batch size, num nodes, 2] such that 0 <= B[i,j,0] < B[i,j,1] < seq len.
        C which shape [batch size, seq len] such that C[i, B[i,j,0]:B[i,j,1]] = 0 if A[i,j]=1 and other entries of C is equal to 1. 
        '''
        # Expand dimensions to match the shape of C
        A_expanded = A.unsqueeze(2)  # shape: [batch_size, num_nodes, 1]
        B_start = B[:, :, 0].unsqueeze(2)  # shape: [batch_size, num_nodes, 1]
        B_end = B[:, :, 1].unsqueeze(2)  # shape: [batch_size, num_nodes, 1]

        # Create a mask to identify the ranges to be set to 0
        mask = (torch.arange(seq_len, device=A.device).unsqueeze(0).unsqueeze(1) >= B_start) & \
            (torch.arange(seq_len, device=A.device).unsqueeze(0).unsqueeze(1) < B_end)

        # Set the ranges to 0 where A equals 1
        C = (1 - A_expanded * mask.float()).prod(dim=1)

        return C


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            query_span=None,
            node_spans=None,
            num_nodes=None,
            constraints=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # set head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Build the forward prop
        # embedding layer
        hidden_states = self.markuplm.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # first six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[:self.config.num_hidden_layers//2]):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # node removal
        batch_size = node_spans.size(0)

        # query representation
        _idx = torch.arange(input_shape[1], device=device).unsqueeze(0)
        _mask = (_idx >= query_span[:, 0].unsqueeze(1)) & (_idx < query_span[:, 1].unsqueeze(1))
        _mask = _mask.float().unsqueeze(-1)
        query_rep = (_mask * hidden_states).sum(dim=1) / _mask.sum(dim=1)
        
        # node repesentation
        _start = node_spans[:, :, 0].unsqueeze(-1)  # shape: (batch_size, num_nodes, 1)
        _end = node_spans[:, :, 1].unsqueeze(-1)  # shape: (batch_size, num_nodes, 1)
        _index = torch.arange(input_shape[1], device=query_span.device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len)
        _mask = (_index >= _start) & (_index < _end)  # shape: (batch_size, num_nodes, seq_len)
        node_reps = (hidden_states.unsqueeze(1) * _mask.unsqueeze(-1)).sum(dim=2) / (_end - _start)
        
        # concat query rep and node rep
        query_node_reps = torch.cat(
            (query_rep.unsqueeze(1).repeat(1, node_reps.size(1), 1), node_reps), dim=-1)
        # node_scores: [batch_size * max_num_nodes]
        node_scores = self.node_removal_layer(query_node_reps).squeeze(-1)
        removed_nodes = (node_scores <= 0)
        
        # make removal mask
        removal_mask = self.make_removal_mask(removed_nodes, node_spans, input_shape[1])
        attention_mask_after_removal = removal_mask * attention_mask
        
        # last six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask_after_removal)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[self.config.num_hidden_layers//2:]):
            i += self.config.num_hidden_layers//2
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # QA loss and train
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 去掉padding和remove的节点
        tmp = attention_mask_after_removal.bool()
        start_logits[~tmp] = -20.
        end_logits[~tmp] = -20.
        del tmp

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 排除padding和removal位置的logits
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            QA_loss = (start_loss + end_loss) / 2

            # node importance score regularization (tree)
            R_tree = 0.
            for b in range(batch_size):
                tmp = 0.
                for i in range(num_nodes[b]):
                    if constraints[b,i] >= 0:
                        tmp += F.relu(node_scores[b, i] - node_scores[b, constraints[b,i]])
                R_tree += tmp / num_nodes[b]
            R_tree /= batch_size

            # 添加length regularization: 1 - 删除的比例 = 保留的比例
            R_len = torch.mean(1. - torch.sum(1.-removal_mask, dim=1) / torch.sum(attention_mask, dim=1))

            total_loss = QA_loss + 5e-2 * R_len + 1e-1 * R_tree
            return  total_loss, QA_loss, R_len, R_tree
        else:
            return start_logits, end_logits


class MarkupLMForQuestionAnswering_node_removal_v2(MarkupLMPreTrainedModel):
    '''
    intersect with answer node + nodes without intersection removal
    '''

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # auxiliuary classifier
        self.intersect_with_answer_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
        )
        # node removal module
        self.node_removal_layer = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )
        self.register_buffer("intersect_with_answer_weight", torch.tensor([1., 30.]))
        self.intersect_with_answer_loss_fct = CrossEntropyLoss(weight=self.intersect_with_answer_weight, ignore_index=-100)
        # qa output
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        # initialize
        self.init_weights()
        # 


    def extend_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            query_span=None,
            node_spans=None,
            intersect_with_answer_labels=None,
    ):
        '''
        query_span: [batch_size]
        node_spans: [batch_size, num_nodes, 2]
        intersect_with_answer_labels: [batch_size, num_nodes]
        '''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # set head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Build the forward prop
        # embedding layer
        hidden_states = self.markuplm.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # first six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[:self.config.num_hidden_layers//2]):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        # node removal
        batch_size = node_spans.size(0)
        num_nodes = node_spans.size(1)
        query_rep = []
        node_reps = []
        for b in range(batch_size):
            # 当前case的query embeddings: [dim]
            query_rep_case = hidden_states[b, query_span[b,0]:query_span[b,1]].mean(dim=0)
            query_rep.append(query_rep_case)
            # 当前case的节点数和spans
            # num_nodes_case = num_nodes[b].item()
            node_spans_case = node_spans[b]    # [num_nodes * 2]
            # 当前case的所有nodes的embedings: [num_nodes * dim]
            node_reps_case = torch.stack([hidden_states[b, sp[0]:sp[1]].mean(dim=0) for sp in node_spans[b]], dim=0)
            node_reps.append(node_reps_case)
        # query_rep: [batch_size * dim] & node_reps: [batch_size * max_num_nodes * dim]
        query_rep = torch.stack(query_rep, dim=0)
        node_reps = torch.stack(node_reps, dim=0)
        # query_node_reps: [batch_size, num_nodes, 2*dim]
        query_node_reps = torch.cat((query_rep.unsqueeze(1).repeat(1,num_nodes,1), node_reps), dim=-1)
        # intersect_with_answer_logits: [batch_size, num_nodes, 2]
        intersect_with_answer_logits = self.intersect_with_answer_classifier(query_node_reps)
        if intersect_with_answer_labels is not None:
            intersect_with_answer_loss = self.intersect_with_answer_loss_fct(
                intersect_with_answer_logits.view(-1, 2), intersect_with_answer_labels.view(-1))
        intersect_with_answer_nodes = intersect_with_answer_logits[:,:,1] > intersect_with_answer_logits[:,:,0]
        # node removal layer: bool, [batch_size, num_nodes]
        removed_nodes = self.node_removal_layer(query_node_reps).squeeze(-1) < 0
        # make mask
        removal_mask = torch.ones(input_shape, device=device)
        # removal_mask = (intersect_with_node_logits[:,:,1] > intersect_with_node_logits[:,:,0]).int()
        for i in range(batch_size):
            for j in range(num_nodes):
                if not intersect_with_answer_nodes[i,j] and removed_nodes[i,j]:
                    removal_mask[i, node_spans[i,j,0]:node_spans[i,j,1]] = 0
        attention_mask_after_removal = removal_mask * attention_mask
        # last six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask_after_removal)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[self.config.num_hidden_layers//2:]):
            i += self.config.num_hidden_layers//2
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # QA loss and train
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 去掉padding和remove的节点
        tmp = attention_mask_after_removal.bool()
        start_logits[~tmp] = -1e6
        end_logits[~tmp] = -1e6
        del tmp

        total_loss = None
        if start_positions is not None and end_positions is not None and intersect_with_answer_labels is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 排除padding和removal位置的logits
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2 + 0.4 * intersect_with_answer_loss
        
            return  total_loss, start_logits, end_logits
        else:
            return start_logits, end_logits




class MarkupLMForQuestionAnswering_node_removal_v3(MarkupLMPreTrainedModel):
    '''
    Webpage segmentation + Gumbel-softmax trick
    1. 将一个HTML document分割成若干不相交的部分
    2. 判断每个部分是否需要冗余
    3. 用Gumbel-softmax trick进行删除操作
    '''

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # node removal layer: this component will give a score for each node to measure how informative it is for answering the question.
        self.node_removal_layer = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )

        # qa output
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()
        self.init_node_removal_layer()


    def init_node_removal_layer(self):
        for layer in self.node_removal_layer:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)


    def extend_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    

    def make_removal_mask(self, A, B, seq_len):
        '''
        A is a tensor with shape [batch size, num nodes], which entries are either 0 or 1.
        B is a nonnegative integer tensor with shape [batch size, num nodes, 2] such that 0 <= B[i,j,0] < B[i,j,1] < seq len.
        C which shape [batch size, seq len] such that C[i, B[i,j,0]:B[i,j,1]] = 0 if A[i,j]=1 and other entries of C is equal to 1. 
        '''
        # Expand dimensions to match the shape of C
        A_expanded = A.unsqueeze(2)  # shape: [batch_size, num_nodes, 1]
        B_start = B[:, :, 0].unsqueeze(2)  # shape: [batch_size, num_nodes, 1]
        B_end = B[:, :, 1].unsqueeze(2)  # shape: [batch_size, num_nodes, 1]

        # Create a mask to identify the ranges to be set to 0
        mask = (torch.arange(seq_len, device=A.device).unsqueeze(0).unsqueeze(1) >= B_start) & \
            (torch.arange(seq_len, device=A.device).unsqueeze(0).unsqueeze(1) < B_end)

        # Set the ranges to 0 where A equals 1
        C = (1 - A_expanded * mask.float()).prod(dim=1)

        return C


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            query_span=None,
            node_spans=None,
            num_nodes=None,
            constraints=None,
            temperature=1.,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # set head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Build the forward prop
        # embedding layer
        hidden_states = self.markuplm.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # first six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[:self.config.num_hidden_layers//2]):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # node removal
        batch_size = node_spans.size(0)

        # query representation
        _idx = torch.arange(input_shape[1], device=device).unsqueeze(0)
        _mask = (_idx >= query_span[:, 0].unsqueeze(1)) & (_idx < query_span[:, 1].unsqueeze(1))
        _mask = _mask.float().unsqueeze(-1)
        query_rep = (_mask * hidden_states).sum(dim=1) / _mask.sum(dim=1)
        
        # node repesentation
        _start = node_spans[:, :, 0].unsqueeze(-1)  # shape: (batch_size, num_nodes, 1)
        _end = node_spans[:, :, 1].unsqueeze(-1)  # shape: (batch_size, num_nodes, 1)
        _index = torch.arange(input_shape[1], device=query_span.device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len)
        _mask = (_index >= _start) & (_index < _end)  # shape: (batch_size, num_nodes, seq_len)
        node_reps = (hidden_states.unsqueeze(1) * _mask.unsqueeze(-1)).sum(dim=2) / (_end - _start)
        
        # concat query rep and node rep
        query_node_reps = torch.cat(
            (query_rep.unsqueeze(1).repeat(1, node_reps.size(1), 1), node_reps), dim=-1)
        # node_scores: [batch_size, max_num_nodes]
        node_scores = self.node_removal_layer(query_node_reps)
        removed_nodes = 1. - F.gumbel_softmax(
            torch.cat((torch.zeros_like(node_scores), node_scores), dim=-1), tau=temperature, hard=True)[:,:,1]
        node_scores = node_scores.squeeze(-1)
        
        # make removal mask
        removal_mask = self.make_removal_mask(removed_nodes, node_spans, input_shape[1])
        attention_mask_after_removal = removal_mask * attention_mask
        
        # last six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask_after_removal)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[self.config.num_hidden_layers//2:]):
            i += self.config.num_hidden_layers//2
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # QA loss and train
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 去掉padding和remove的节点
        tmp = attention_mask_after_removal.bool()
        start_logits[~tmp] = -100.
        end_logits[~tmp] = -100.
        del tmp

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 排除padding和removal位置的logits
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            QA_loss = (start_loss + end_loss) / 2

            # node importance score regularization (tree)
            R_tree = 0.
            for b in range(batch_size):
                tmp = 0.
                for i in range(num_nodes[b]):
                    if constraints[b,i] >= 0:
                        tmp += F.relu(node_scores[b, i] - node_scores[b, constraints[b,i]])
                R_tree += tmp / num_nodes[b]
            R_tree /= batch_size

            # 添加length regularization: 1 - 删除的比例 = 保留的比例
            R_len = torch.mean(1. - torch.sum(1.-removal_mask, dim=1) / torch.sum(attention_mask, dim=1))

            total_loss = QA_loss + 1e-3 * R_len + 1e-2 * R_tree
            return  total_loss, QA_loss, R_len, R_tree
        else:
            return start_logits, end_logits


class MarkupLMForQuestionAnswering_node_removal_v4(MarkupLMPreTrainedModel):
    '''
    SIF embedding replaces mean embedding
    CRF layer or Basyesnian network layer
    '''

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # node removal layer: this component will give a score for each node to measure how informative it is for answering the question.
        self.node_removal_layer = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),    # 0
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),      # 2
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),                       # 4
        )
        # qa output
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()


    def extend_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            return_dict=None,
            xpath_tags_seq=None,
            xpath_subs_seq=None,
            query_span=None,
            node_spans=None,
            num_nodes=None,
            constraints=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # set head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Build the forward prop
        # embedding layer
        hidden_states = self.markuplm.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # first six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[:self.config.num_hidden_layers//2]):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        # node removal
        batch_size = node_spans.size(0)
        max_num_nodes = node_spans.size(1)

        # query representation
        _idx = torch.arange(input_shape[1], device=query_span.device).unsqueeze(0)
        _mask = (_idx >= query_span[:, 0].unsqueeze(1)) & (_idx < query_span[:, 1].unsqueeze(1))
        _mask = _mask.float().unsqueeze(-1)
        query_rep = (_mask * hidden_states).sum(dim=1) / _mask.sum(dim=1)
        
        # node repesentation
        _start = node_spans[:, :, 0].unsqueeze(-1)  # shape: (batch_size, num_nodes, 1)
        _end = node_spans[:, :, 1].unsqueeze(-1)  # shape: (batch_size, num_nodes, 1)
        _index = torch.arange(input_shape[1], device=query_span.device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len)
        _mask = (_index >= _start) & (_index < _end)  # shape: (batch_size, num_nodes, seq_len)
        node_reps = (hidden_states.unsqueeze(1) * _mask.unsqueeze(-1)).sum(dim=2) / (_end - _start)
        
        # concat query rep and node rep
        query_node_reps = torch.cat(
            (query_rep.unsqueeze(1).repeat(1, max_num_nodes, 1), node_reps), dim=-1)
        
        # node removal layer [batch_size * max_num_nodes]
        node_scores = self.node_removal_layer(query_node_reps).squeeze(-1)
        # removed_nodes = node_scores <= 0
        reserved_nodes = F.gumbel_softmax(node_scores, tau=1, hard=True)
        
        # make mask
        removal_mask = torch.ones(input_shape, device=device)
        for i in range(batch_size):
            for j in range(num_nodes[i]):
                if not reserved_nodes[i,j]:
                    removal_mask[i, node_spans[i,j,0]:node_spans[i,j,1]] = 0
        attention_mask_after_removal = removal_mask * attention_mask

        # make hidden_states[i,j] = 0 if its affiliated node is removed
        hidden_states
        
        # last six self-attention layers
        extend_attention_mask = self.extend_attention_mask(attention_mask_after_removal)
        for i, layer_module in enumerate(self.markuplm.encoder.layer[self.config.num_hidden_layers//2:]):
            i += self.config.num_hidden_layers//2
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extend_attention_mask,
                head_mask=layer_head_mask,
            )
            hidden_states = layer_outputs[0]
        
        # QA loss and train
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 去掉padding和remove的节点
        tmp = attention_mask_after_removal.bool()
        start_logits[~tmp] = -1e2
        end_logits[~tmp] = -1e2
        del tmp

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 排除padding和removal位置的logits
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            QA_loss = (start_loss + end_loss) / 2
            # node importance score regularization (tree)
            R_tree = 0.
            for b in range(batch_size):
                tmp = 0.
                for i in range(num_nodes[b]):
                    if constraints[b,i] >= 0:
                        tmp += F.relu(node_scores[b, i] - node_scores[b, constraints[b,i]])
                R_tree += tmp / num_nodes[b]
            R_tree /= batch_size
            # 添加length regularization: 1 - 删除的比例 = 保留的比例
            R_len = torch.mean(1. - torch.sum(1.-removal_mask, dim=1) / torch.sum(attention_mask, dim=1))

            total_loss = QA_loss + 1e-2 * R_len + 1e-1 * R_tree
        
            return  total_loss, QA_loss, R_len, R_tree
        else:
            return start_logits, end_logits