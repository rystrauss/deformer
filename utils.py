from typing import Optional, List, Callable, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax


def _weight_decay_exclude(
    exclude_names: Optional[List[str]] = None,
) -> Callable[[str, str, jnp.ndarray], bool]:
    """Logic for deciding which parameters to include for weight decay.

    Args:
      exclude_names: an optional list of names to include for weight_decay. ['w']
        by default.

    Returns:
      A predicate that returns True for params that need to be excluded from
      weight_decay.
    """
    # By default weight_decay the weights but not the biases.
    if not exclude_names:
        exclude_names = ["b"]

    def exclude(module_name: str, name: str, value: jnp.array):
        del value
        # Do not weight decay the parameters of normalization blocks.
        if any([norm_name in module_name for norm_name in ["layer_norm", "batchnorm"]]):
            return True
        else:
            return name in exclude_names

    return exclude


class AddWeightDecayState(NamedTuple):
    """Stateless transformation."""


def add_weight_decay(
    weight_decay: float, exclude_names: Optional[List[str]] = None
) -> optax.GradientTransformation:
    """Add parameter scaled by `weight_decay` to the `updates`.

    Same as optax.add_decayed_weights but can exclude parameters by name.

    Args:
      weight_decay: weight_decay coefficient.
      exclude_names: an optional list of names to exclude for weight_decay. ['b']
        by default.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return AddWeightDecayState()

    def update_fn(updates, state, params):
        exclude = _weight_decay_exclude(exclude_names=exclude_names)

        u_ex, u_in = hk.data_structures.partition(exclude, updates)
        _, p_in = hk.data_structures.partition(exclude, params)
        u_in = jax.tree_map(lambda g, p: g + weight_decay * p, u_in, p_in)
        updates = hk.data_structures.merge(u_ex, u_in)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
