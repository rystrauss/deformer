import json
import os
import pickle

import click
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_probability.python.internal.backend import jax as tf2jax
from tensorflow_probability.substrates.jax.math import reduce_logmeanexp
from tqdm import tqdm

import wandb
from masking import BernoulliMaskGenerator, get_add_mask_fn
from models import DEformer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
tf.config.set_visible_devices([], "GPU")


@click.command()
@click.option("--dataset", type=click.STRING, required=True)
@click.option("--model_artifact", type=click.STRING, required=True)
@click.option("--batch_size", type=click.INT, default=32)
@click.option("--num_permutations", type=click.INT, default=10)
@click.option("--num_masks", type=click.INT, default=5)
@click.option("--num_instances", type=click.INT)
def main(
    dataset,
    model_artifact,
    batch_size,
    num_permutations,
    num_masks,
    num_instances,
):
    config = locals()

    run = wandb.init(
        project="deformer",
        job_type="eval_deformer",
        config=config,
    )

    ds = tfds.load(dataset, split="test")

    if num_instances is not None:
        ds = ds.take(num_instances)

    ds = ds.batch(batch_size, drop_remainder=True)

    add_mask_fn = get_add_mask_fn(BernoulliMaskGenerator())
    ds = ds.map(add_mask_fn)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    data_std = np.std(
        np.vstack([x["features"] for x in ds.as_numpy_iterator()]),
        axis=0,
        keepdims=True,
    )

    model_artifact = run.use_artifact(model_artifact)
    model_dir = model_artifact.download()

    with open(os.path.join(model_dir, "model_config.json"), "r") as fp:
        model_config = json.load(fp)

    with open(os.path.join(model_dir, "state.pkl"), "rb") as fp:
        model_state = pickle.load(fp)

    def eval_fn(batch):
        model = DEformer(**model_config)

        def eval_single_order(key):
            x = batch["features"]
            b = batch["mask"]

            noise = jax.random.uniform(key, x.shape) - b
            order = jnp.argsort(noise, axis=-1)

            dist = model(x, order, is_training=False)
            chain_lls = dist.log_prob(x)
            joint_ll = jnp.sum(chain_lls, axis=-1)

            ac_ll = jnp.sum(chain_lls * (1 - b), axis=-1)

            imputations = model.impute(x, b, order)
            inverse_order = jnp.argsort(order, axis=-1)
            imputations = tf2jax.gather(
                imputations, inverse_order, batch_dims=1, axis=1
            )
            x = tf2jax.gather(x, inverse_order, batch_dims=1, axis=1)

            error = (imputations - x) ** 2 * (1 - b)

            return joint_ll, ac_ll, error

        keys = jnp.asarray(hk.next_rng_keys(num_permutations))
        joint_lls, ac_lls, errors = jax.vmap(eval_single_order)(keys)
        joint_lls = reduce_logmeanexp(joint_lls, axis=0)
        ac_lls = reduce_logmeanexp(ac_lls, axis=0)
        errors = jnp.mean(errors, axis=0)
        return joint_lls, ac_lls, errors

    eval_fn = jax.jit(hk.transform_with_state(eval_fn).apply)

    prng = hk.PRNGSequence(91)

    joint_lls, ac_lls, errors, bs = [], [], [], []

    for i in range(num_masks):
        joint_lls.append([])
        ac_lls.append([])
        errors.append([])
        bs.append([])

        for batch in tqdm(
            ds.as_numpy_iterator(),
            desc=f"Trial {i + 1}/{num_masks}",
            total=ds.cardinality().numpy().item(),
        ):
            (joint, ac, error), _ = eval_fn(
                model_state.params, model_state.state, prng.next(), batch
            )

            joint_lls[-1].append(jax.device_get(joint))
            ac_lls[-1].append(jax.device_get(ac))
            errors[-1].append(jax.device_get(error))
            bs[-1].append(batch["mask"])

        joint_lls[-1] = np.concatenate(joint_lls[-1], axis=-1)
        ac_lls[-1] = np.concatenate(ac_lls[-1], axis=-1)
        errors[-1] = np.concatenate(errors[-1], axis=0)
        bs[-1] = np.concatenate(bs[-1], axis=0)

    joint_lls = np.asarray(joint_lls)
    ac_lls = np.asarray(ac_lls)
    errors = np.asarray(errors)
    bs = np.asarray(bs)

    mse = np.sum(errors, axis=-2) / np.count_nonzero(1 - bs, axis=-2)
    nrmse = np.sqrt(mse) / data_std

    results = {
        "joint_ll": np.mean(joint_lls).item(),
        "ac_ll": np.mean(ac_lls).item(),
        "ac_ll_std": np.std(np.mean(ac_lls, axis=1)).item(),
        "nrmse": np.mean(nrmse).item(),
        "nrmse_std": np.std(np.mean(nrmse, axis=1)).item(),
    }

    run.log(results)


if __name__ == "__main__":
    main()