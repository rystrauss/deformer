from typing import Sequence, Optional, Tuple

import chex
import distrax
import einops
import haiku as hk
import jax.numpy as jnp
import numpy as np
from chex import Array
from tensorflow_probability.python.internal.backend import jax as tf2jax

from layers import CausalSelfAttention, OneDimensionalGMM, Categorical


class DEformer(hk.Module):
    """The DEformer model for continuous features.

    This is an implementation of the DEformer model described in:
    https://arxiv.org/abs/2106.06989

    Args:
        num_features: The number of data features.
        num_layers: The number of Transformer layers.
        mlp_hidden_units: A sequence of integers, where the ith integer is the number
            of hidden units in the ith layer of the pre-attention MLPs. Note that the
            final number in the sequence (i.e. the output dimensionality), will define
            the width of the rest of the network.
        index_embedding_dim: Dimension of the index embeddings.
        mixture_components: The number of components in the GMM output distributions.
        num_heads: The number of heads used in the multi-head attention.
        dropout: The dropout rate.
        widening_factor: The widening factor in the Transformer layer MLPs.
        name: Optional. The name of the module.
    """

    def __init__(
        self,
        num_features: int,
        num_layers: int,
        mlp_hidden_units: Sequence[int],
        index_embedding_dim: int = 32,
        mixture_components: int = 100,
        num_heads: int = 4,
        dropout: float = 0.0,
        widening_factor: int = 4,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._index_embedding = hk.Embed(
            num_features, index_embedding_dim, name="index_embeddings"
        )
        self._index_mlp = hk.nets.MLP(mlp_hidden_units)
        self._value_mlp = hk.nets.MLP(mlp_hidden_units)

        self._attention_layers = [
            CausalSelfAttention(
                widening_factor=widening_factor,
                dropout_prob=dropout,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ]

        self._out_dist = OneDimensionalGMM(mixture_components)

    def _prepare_inputs(self, x: Array, order: Array) -> Array:
        x = tf2jax.gather(x, order, batch_dims=1, axis=1)

        index_embeddings = self._index_embedding(order)
        index_features = self._index_mlp(index_embeddings)
        index_features *= index_features.shape[-1] ** 0.5

        values = jnp.expand_dims(x, axis=-1)
        index_values = jnp.concatenate([index_embeddings, values], axis=-1)
        value_features = self._value_mlp(index_values)
        value_features *= value_features.shape[-1] ** 0.5

        combined = jnp.stack([index_features, value_features], axis=2)
        combined = einops.rearrange(combined, "b t s h -> b (t s) h")

        return combined

    def __call__(
        self, x: Array, order: Array, is_training: bool = False
    ) -> distrax.Distribution:
        """Performs a forward pass.

        Args:
            x: The feature values, with shape [batch_size, num_features].
            order: The orders in which to perform the autoregressive factorizations.
                This should be a matrix of shape [batch_size, num_features] and have
                integer values. Each row in this matrix should be a permutation of
                [0, 1, ..., d - 1], where d is the number of features. As an example,
                if the provided order for a given input is [1, 0, 2], then this means
                that the outputted distribution at index 0 will be p(x_0 | x_1), the
                outputted distribution at index 1 will be p(x_1), and the outputted
                distribution at index 2 will be p(x_2 | x_1, x_0).
            is_training: Whether or not to run the model in training mode.

        Returns:
            A `tfd.Distribution` containing all of the 1D conditionals.
        """
        h = self._prepare_inputs(x, order)

        for layer in self._attention_layers:
            h = layer(h, is_training=is_training)

        h = h[:, ::2]
        inverse_order = jnp.argsort(order, axis=-1)
        h = tf2jax.gather(h, inverse_order, batch_dims=1, axis=1)

        return self._out_dist(h)

    def get_conditional_distributions(
        self, x: Array, b: Array, order: Array
    ) -> distrax.Distribution:
        """Gets the distributions p(x_i | x_o) for unobserved i.

        Args:
            x: The observed feature values, with shape [batch_size, num_features].
            b: A binary mask indicating which features are observed, with shape
                [batch_size, num_features].
            order: The orders in which to perform the autoregressive factorizations.
                Note that these orderings should satisfy the condition that all observed
                features come before all unobserved features in the order.

        Returns:
            The distributions p(x_i | x_o).
        """
        h = self._prepare_inputs(x, order)

        n = x.shape[-1]
        attention_mask = jnp.tril(jnp.ones((n * 2, n * 2)))
        c = jnp.count_nonzero(b, axis=-1) * 2
        attention_mask = jnp.expand_dims(attention_mask, 0) * jnp.expand_dims(
            jnp.less(jnp.expand_dims(jnp.arange(n * 2), 0), jnp.expand_dims(c, axis=1)),
            axis=1,
        )
        attention_mask = jnp.maximum(attention_mask, jnp.expand_dims(jnp.eye(n * 2), 0))

        for layer in self._attention_layers:
            h = layer(h, is_training=False, attention_mask=attention_mask)

        h = h[:, ::2]
        inverse_order = jnp.argsort(order, axis=-1)
        h = tf2jax.gather(h, inverse_order, batch_dims=1, axis=1)
        dist = self._out_dist(h)

        return dist

    def impute(self, x: Array, b: Array, order: Array) -> Array:
        """Imputes missing values.

        Args:
            x: The observed feature values, with shape [batch_size, num_features].
            b: A binary mask indicating which features are observed, with shape
                [batch_size, num_features].
            order: The orders in which to perform the autoregressive factorizations.
                Note that these orderings should satisfy the condition that all observed
                features come before all unobserved features in the order.

        Returns:
            The original input `x`, but with unobserved indices imputed with the mean of
            the predicted distributions.
        """
        dist = self.get_conditional_distributions(x, b, order)
        imputed = jnp.where(b == 1, x, dist.mean())
        return imputed


class ContinuousDiscreteDEformer(hk.Module):
    """An extension of the DEformer model for mixed continuous and discrete features.

    This version of DEformer was not described in the original paper.

    Args:
        classes_per_feature: A vector with as many elements as features, where the ith
           element is an integer specifying the number of classes that the ith feature
           can take on. If the ith element is 0 or 1, then that feature is treated
           as continuous.
        num_layers: The number of Transformer layers.
        mlp_hidden_units: A sequence of integers, where the ith integer is the number
            of hidden units in the ith layer of the pre-attention MLPs. Note that the
            final number in the sequence (i.e. the output dimensionality), will define
            the width of the rest of the network.
        index_embedding_dim: Dimension of the index embeddings.
        class_embedding_dim: Dimension of the class embeddings.
        mixture_components: The number of components in the GMM output distributions.
        num_heads: The number of heads used in the multi-head attention.
        dropout: The dropout rate.
        widening_factor: The widening factor in the Transformer layer MLPs.
        name: Optional. The name of the module.
    """

    def __init__(
        self,
        classes_per_feature: Sequence[int],
        num_layers: int,
        mlp_hidden_units: Sequence[int],
        index_embedding_dim: int = 32,
        class_embedding_dim: int = 32,
        mixture_components: int = 100,
        num_heads: int = 4,
        dropout: float = 0.0,
        widening_factor: int = 4,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        classes_per_feature = np.asarray(classes_per_feature, dtype=jnp.int32)
        chex.assert_rank(classes_per_feature, 1)

        self._num_features = len(classes_per_feature)
        self._index_embedding = hk.Embed(
            self._num_features, index_embedding_dim, name="index_embeddings"
        )

        self._index_mlp = hk.nets.MLP(mlp_hidden_units)
        self._continuous_value_mlp = hk.nets.MLP(mlp_hidden_units)
        self._discrete_value_mlp = hk.nets.MLP(mlp_hidden_units)

        self._discrete_indices = np.where(classes_per_feature > 1)[0]
        self._continuous_indices = np.where(classes_per_feature <= 1)[0]

        discrete_classes = classes_per_feature[classes_per_feature > 1]
        self._cumulative_classes = np.cumsum(discrete_classes)
        self._class_embeddings = hk.Embed(
            self._cumulative_classes[-1], class_embedding_dim, name="class_embeddings"
        )

        self._attention_layers = [
            CausalSelfAttention(
                widening_factor=widening_factor,
                dropout_prob=dropout,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ]

        self._continuous_dist = OneDimensionalGMM(mixture_components)
        self._discrete_dist = Categorical(discrete_classes)

    def _merge_continuous_discrete(
        self,
        continuous_tensor: Array,
        discrete_tensor: Array,
        continuous_inds: Array,
        discrete_inds: Array,
    ) -> Array:
        batch_size = continuous_tensor.shape[0]

        all_shape = (batch_size, self._num_features) + continuous_tensor.shape[2:]
        all_value_features = jnp.zeros(all_shape)

        continuous_batch_inds = jnp.broadcast_to(
            jnp.expand_dims(jnp.arange(batch_size), 1), continuous_inds.shape
        )
        continuous_scatter_indices = einops.rearrange(
            jnp.stack([continuous_batch_inds, continuous_inds], axis=1),
            "b x y -> b y x",
        )
        discrete_batch_inds = jnp.broadcast_to(
            jnp.expand_dims(jnp.arange(batch_size), 1), discrete_inds.shape
        )
        discrete_scatter_indices = einops.rearrange(
            jnp.stack([discrete_batch_inds, discrete_inds], axis=1), "b x y -> b y x"
        )
        all_value_features = all_value_features.at[
            tuple(jnp.moveaxis(continuous_scatter_indices, -1, 0))
        ].set(continuous_tensor)
        all_value_features = all_value_features.at[
            tuple(jnp.moveaxis(discrete_scatter_indices, -1, 0))
        ].set(discrete_tensor.astype(all_value_features.dtype))

        return all_value_features

    def _prepare_inputs(
        self, x: Array, order: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        inverse_order = jnp.argsort(order, axis=-1)
        discrete_inds = tf2jax.gather(inverse_order, self._discrete_indices, axis=1)
        continuous_inds = tf2jax.gather(inverse_order, self._continuous_indices, axis=1)

        index_embeddings = self._index_embedding(order)
        index_features = self._index_mlp(index_embeddings)
        index_features *= index_features.shape[-1] ** 0.5

        x = tf2jax.gather(x, order, batch_dims=1, axis=1)

        discrete_values = tf2jax.gather(x, discrete_inds, batch_dims=1, axis=1).astype(
            jnp.int32
        )
        mapped_discrete_values = discrete_values + jnp.pad(
            self._cumulative_classes[:-1], (1, 0)
        )
        discrete_class_embeddings = self._class_embeddings(mapped_discrete_values)

        continuous_values = tf2jax.gather(x, continuous_inds, batch_dims=1, axis=1)

        continuous_index_embeddings = tf2jax.gather(
            index_embeddings, continuous_inds, batch_dims=1, axis=1
        )
        discrete_index_embeddings = tf2jax.gather(
            index_embeddings, discrete_inds, batch_dims=1, axis=1
        )

        continuous_index_values = jnp.concatenate(
            [continuous_index_embeddings, continuous_values[..., None]], axis=-1
        )
        discrete_index_values = jnp.concatenate(
            [discrete_index_embeddings, discrete_class_embeddings], axis=-1
        )

        continuous_value_features = self._continuous_value_mlp(continuous_index_values)
        discrete_value_features = self._discrete_value_mlp(discrete_index_values)
        continuous_value_features *= continuous_value_features.shape[-1] ** 0.5
        discrete_value_features *= discrete_value_features.shape[-1] ** 0.5

        all_value_features = self._merge_continuous_discrete(
            continuous_value_features,
            discrete_value_features,
            continuous_inds,
            discrete_inds,
        )

        combined = jnp.stack([index_features, all_value_features], axis=2)
        combined = einops.rearrange(combined, "b t s h -> b (t s) h")

        return (
            inverse_order,
            combined,
            continuous_inds,
            discrete_inds,
            continuous_values,
            discrete_values,
        )

    def __call__(self, x: Array, order: Array, is_training: bool = False) -> Array:
        """Performs a forward pass.

        Args:
            x: The feature values, with shape [batch_size, num_features].
            order: The orders in which to perform the autoregressive factorizations.
                This should be a matrix of shape [batch_size, num_features] and have
                integer values. Each row in this matrix should be a permutation of
                [0, 1, ..., d - 1], where d is the number of features. As an example,
                if the provided order for a given input is [1, 0, 2], then this means
                that the outputted distribution at index 0 will be p(x_0 | x_1), the
                outputted distribution at index 1 will be p(x_1), and the outputted
                distribution at index 2 will be p(x_2 | x_1, x_0).
            is_training: Whether or not to run the model in training mode.

        Returns:
            The model log-likelihoods of the inputs, with shape
            [batch_size, num_features].
        """
        (
            inverse_order,
            h,
            continuous_inds,
            discrete_inds,
            continuous_values,
            discrete_values,
        ) = self._prepare_inputs(x, order)

        for layer in self._attention_layers:
            h = layer(h, is_training=is_training)

        h = h[:, ::2]
        continuous_h = tf2jax.gather(h, continuous_inds, batch_dims=1, axis=1)
        discrete_h = tf2jax.gather(h, discrete_inds, batch_dims=1, axis=1)

        continuous_dist = self._continuous_dist(continuous_h)
        discrete_dist = self._discrete_dist(discrete_h)

        continuous_ll = continuous_dist.log_prob(continuous_values)
        discrete_ll = discrete_dist.log_prob(discrete_values)

        all_ll = self._merge_continuous_discrete(
            continuous_ll, discrete_ll, continuous_inds, discrete_inds
        )

        all_ll = tf2jax.gather(all_ll, inverse_order, batch_dims=1, axis=1)

        return all_ll

    def impute(self, x: Array, b: Array, order: Array) -> Array:
        """Imputes missing values.

        Args:
            x: The observed feature values, with shape [batch_size, num_features].
            b: A binary mask indicating which features are observed, with shape
                [batch_size, num_features].
            order: The orders in which to perform the autoregressive factorizations.
                Note that these orderings should satisfy the condition that all observed
                features come before all unobserved features in the order.

        Returns:
            The original input `x`, but with unobserved indices imputed with the mean of
            the predicted distributions for continuous features and the mode for
            discrete features.
        """
        (
            inverse_order,
            h,
            continuous_inds,
            discrete_inds,
            continuous_values,
            discrete_values,
        ) = self._prepare_inputs(x, order)

        n = x.shape[-1]
        attention_mask = jnp.tril(jnp.ones((n * 2, n * 2)))
        c = jnp.count_nonzero(b, axis=-1) * 2
        attention_mask = jnp.expand_dims(attention_mask, 0) * jnp.expand_dims(
            jnp.less(jnp.expand_dims(jnp.arange(n * 2), 0), jnp.expand_dims(c, axis=1)),
            axis=1,
        )
        attention_mask = jnp.maximum(attention_mask, jnp.expand_dims(jnp.eye(n * 2), 0))

        for layer in self._attention_layers:
            h = layer(h, is_training=False, attention_mask=attention_mask)

        h = h[:, ::2]
        continuous_h = tf2jax.gather(h, continuous_inds, batch_dims=1, axis=1)
        discrete_h = tf2jax.gather(h, discrete_inds, batch_dims=1, axis=1)

        continuous_dist = self._continuous_dist(continuous_h)
        discrete_dist = self._discrete_dist(discrete_h)

        continuous_imputations = continuous_dist.mean()
        discrete_imputations = discrete_dist.mode()

        all_imputations = self._merge_continuous_discrete(
            continuous_imputations, discrete_imputations, continuous_inds, discrete_inds
        )

        all_imputations = tf2jax.gather(
            all_imputations, inverse_order, batch_dims=1, axis=1
        )

        return all_imputations
