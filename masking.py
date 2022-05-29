from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence, Union

import numpy as np
import tensorflow as tf


class MaskGenerator(ABC):
    def __init__(
        self, seed: Optional[int] = None, dtype: Union[str, object] = np.float32
    ):
        self._rng = np.random.RandomState(seed=seed)
        self._dtype = dtype

    def __call__(self, shape: Sequence[int]):
        return self.call(np.asarray(shape)).astype(self._dtype)

    @abstractmethod
    def call(self, shape: Sequence[int]) -> np.ndarray:
        pass


class MixtureMaskGenerator(MaskGenerator):
    def __init__(self, generators, weights=None, batch_level=False, **kwargs):
        super().__init__(**kwargs)
        self._generators = generators

        if weights is None:
            weights = [1] * len(generators)
        weights = np.asarray(weights)

        assert len(generators) == len(weights)

        self._weights = weights / np.sum(weights)

        self._batch_level = batch_level

    def call(self, shape, **kwargs):
        if self._batch_level:
            ind = self._rng.choice(len(self._generators), 1, p=self._weights)[0]
            return self._generators[ind](shape)

        inds = self._rng.choice(len(self._generators), shape[0], p=self._weights)
        return np.concatenate(
            [self._generators[i]((1, *shape[1:])) for i in inds], axis=0
        )


class UniformMaskGenerator(MaskGenerator):
    def __init__(self, bounds: Optional[Tuple[float, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self._bounds = bounds

    def call(self, shape: Sequence[int]) -> np.ndarray:
        orig_shape = None
        if len(shape) != 2:
            orig_shape = shape
            shape = (shape[0], np.prod(shape[1:]))

        b, d = shape

        result = []
        for _ in range(b):
            if self._bounds is None:
                q = self._rng.choice(d)
            else:
                l = int(d * self._bounds[0])
                h = int(d * self._bounds[1])
                q = l + self._rng.choice(h)
            inds = self._rng.choice(d, q, replace=False)
            mask = np.zeros(d)
            mask[inds] = 1
            result.append(mask)

        result = np.vstack(result)

        if orig_shape is not None:
            result = np.reshape(result, orig_shape)

        return result


class BernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, shape: Sequence[int]) -> np.ndarray:
        return self._rng.binomial(1, self.p, size=shape)


def get_mask_generator(mask_generator_name, **kwargs):
    return {
        "BernoulliMaskGenerator": BernoulliMaskGenerator,
        "UniformMaskGenerator": UniformMaskGenerator,
    }[mask_generator_name](**kwargs)


def get_add_mask_fn(mask_generator):
    def fn(d):
        key = "image" if "image" in d else "features"
        x = d[key]
        [mask] = tf.py_function(mask_generator, [tf.shape(x)], [x.dtype])
        if key == "image":
            mask = tf.reshape(mask, [*x.shape[:-1], 1])
        else:
            mask = tf.reshape(mask, x.shape)
        d["mask"] = mask
        return d

    return fn
