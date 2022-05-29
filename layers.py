import math
from typing import Optional

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# Attention implementation adapted from:
# https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py


def attend(q, k, v, dropout_prob=0.0, attention_mask=None):
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim

    attention = jnp.einsum("bthd,bThd->bhtT", q, k)

    scale = 1.0 / math.sqrt(q_head_dim)
    attention *= scale

    if attention_mask is not None:
        # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
        # left-padded sampling.
        large_k = jnp.array(
            1e4 if attention.dtype == jnp.float16 else 1e30, dtype=attention.dtype
        )

        attention = jnp.where(attention_mask[:, None, :, :], attention, -large_k)

    normalized = jax.nn.softmax(attention)
    if dropout_prob > 0:
        normalized = hk.dropout(hk.next_rng_key(), dropout_prob, normalized)
    summed = jnp.einsum("bhtT,bThd->bthd", normalized, v)
    summed = jnp.reshape(summed, [batch, q_indices, hiddens])

    if attention_mask is not None:
        # If all attended tokens are masked, or for masked tokens
        # some rows of logits gets completely masked, in which case the softmax
        # gives a uniform row and we obtain non-zero outputs where it should be
        # zero. We force zeros.
        wipe_attn = jnp.all(
            attention_mask == 0, axis=2, keepdims=True
        )  # shape (B, T, 1)
        summed = jnp.where(wipe_attn, jnp.zeros_like(summed), summed)
    return summed


def conv_1d(output_channels, init_scale=1.0, with_bias=True, name=None):
    """A 1D convolution."""
    return hk.Linear(
        output_size=output_channels,
        with_bias=with_bias,
        w_init=hk.initializers.VarianceScaling(init_scale),
        name=name,
    )


def layer_norm(x, name=None):
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


def make_cross_attention_mask(query_mask, kv_mask):
    batch_size, query_len = query_mask.shape
    _, key_len = kv_mask.shape
    mask = jax.vmap(jnp.outer)(query_mask, kv_mask)
    assert mask.shape == (batch_size, query_len, key_len)
    return mask


class Attention(hk.Module):
    def __init__(
        self,
        num_heads=8,
        init_scale=1.0,
        with_final_bias=True,
        final_init_scale_multiplier=1.0,
        dropout_prob=0.0,
        qk_channels=None,
        v_channels=None,
        output_channels=None,
        name=None,
    ):
        super(Attention, self).__init__(name=name)
        self._num_heads = num_heads
        self._init_scale = init_scale
        self._with_final_bias = with_final_bias
        self._final_init_scale = final_init_scale_multiplier * init_scale
        self._dropout_prob = dropout_prob

        # If none of these are passed, the Q input determines the output shape:
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._output_channels = output_channels

    def __call__(self, inputs_q, inputs_kv, attention_mask=None):
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if self._qk_channels is None:
            self._qk_channels = inputs_q.shape[-1]
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if self._v_channels is None:
            self._v_channels = self._qk_channels
        # Project the output of QKV attention to a desired number of channels.
        # Default to the same number as the output of the QKV attention operation.
        if self._output_channels is None:
            self._output_channels = self._v_channels

        if self._qk_channels % self._num_heads != 0:
            raise ValueError(
                f"qk_channels ({self._qk_channels}) must be divisible by"
                f" num_heads ({self._num_heads})."
            )
        if self._v_channels % self._num_heads != 0:
            raise ValueError(
                f"v_channels ({self._v_channels}) must be divisible by"
                f" num_heads ({self._num_heads})."
            )
        qk_channels_per_head = self._qk_channels // self._num_heads
        v_channels_per_head = self._v_channels // self._num_heads

        # Project QKV to a common feature dimension.
        q = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_q)
        k = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_kv)
        v = conv_1d(self._v_channels, init_scale=self._init_scale)(inputs_kv)

        # Reshape channels for multi-head attention.
        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape
        q = jnp.reshape(q, [batch, q_time, self._num_heads, qk_channels_per_head])
        k = jnp.reshape(k, [batch, kv_time, self._num_heads, qk_channels_per_head])
        v = jnp.reshape(v, [batch, kv_time, self._num_heads, v_channels_per_head])

        result = attend(
            q, k, v, dropout_prob=self._dropout_prob, attention_mask=attention_mask
        )
        return conv_1d(
            self._output_channels,
            with_bias=self._with_final_bias,
            init_scale=self._final_init_scale,
        )(result)


class MLP(hk.Module):
    def __init__(self, widening_factor=4, dropout_prob=0.0, init_scale=1.0, name=None):
        super(MLP, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._init_scale = init_scale

    def __call__(self, x, *, is_training):
        dropout_prob = self._dropout_prob if is_training else 0.0
        output_channels = x.shape[-1]
        x = conv_1d(
            output_channels=self._widening_factor * output_channels,
            init_scale=self._init_scale,
        )(x)
        x = jax.nn.gelu(x)
        x = conv_1d(output_channels=output_channels, init_scale=self._init_scale)(x)
        return hk.dropout(hk.next_rng_key(), dropout_prob, x)


class CausalSelfAttention(hk.Module):
    def __init__(
        self,
        include_diagonal: bool = True,
        widening_factor: int = 4,
        dropout_prob: float = 0.0,
        dropout_attn_prob: float = 0.0,
        num_heads: int = 8,
        att_init_scale: float = 1.0,
        dense_init_scale: float = 1.0,
        name: Optional[str] = None,
    ):
        super(CausalSelfAttention, self).__init__(name=name)
        self._include_diagonal = include_diagonal
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale

    def __call__(self, inputs, *, is_training, attention_mask=None):
        dropout_prob = self._dropout_prob if is_training else 0.0
        dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

        n = inputs.shape[1]
        if self._include_diagonal:
            causal_attention_mask = jnp.tril(jnp.ones((n, n)))
        else:
            causal_attention_mask = 1 - jnp.triu(jnp.ones((n, n)))
        causal_attention_mask = jnp.expand_dims(causal_attention_mask, 0)

        attention_mask = (
            attention_mask if attention_mask is not None else causal_attention_mask
        )

        x = inputs
        qkv_inputs = layer_norm(inputs)
        attention = Attention(
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            dropout_prob=dropout_attn_prob,
        )(qkv_inputs, qkv_inputs, attention_mask=attention_mask)
        attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)
        x += attention

        x += MLP(
            widening_factor=self._widening_factor,
            dropout_prob=dropout_prob,
            init_scale=self._dense_init_scale,
        )(layer_norm(x), is_training=is_training)
        return x


class OneDimensionalGMM(hk.Module):
    def __init__(self, num_components: int = 10, name: Optional[str] = None):
        super().__init__(name=name)
        self._num_components = num_components

    def __call__(self, inputs):
        params = hk.Linear(3 * self._num_components)(inputs)
        params = jnp.reshape(params, [inputs.shape[0], -1, 3 * self._num_components])

        params = params.astype(jnp.float32)
        logits = params[..., : self._num_components]
        means = params[..., self._num_components : -self._num_components]
        scales = jax.nn.softplus(params[..., -self._num_components :]) + 1e-5

        components_dist = distrax.Normal(means, scales)
        mixture_dist = distrax.Categorical(logits)
        return distrax.MixtureSameFamily(mixture_dist, components_dist)


class Categorical(hk.Module):
    def __init__(self, discrete_classes, name=None):
        super().__init__(name=name)
        self._discrete_classes = discrete_classes

    def __call__(self, inputs):
        max_classes = np.max(self._discrete_classes)
        logits = hk.Linear(max_classes)(inputs)
        mask = (
            jnp.arange(max_classes)[None, None] < self._discrete_classes[None, :, None]
        )
        logits = jnp.where(mask, logits, -1e12)
        return distrax.Categorical(logits)
