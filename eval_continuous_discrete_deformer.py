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
from scipy.stats import mode
from tensorflow_probability.substrates.jax.math import reduce_logmeanexp
from tqdm import tqdm

import wandb
from masking import BernoulliMaskGenerator, get_add_mask_fn
from models import ContinuousDiscreteDEformer

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
    dataset, model_artifact, batch_size, num_permutations, num_masks, num_instances
):
    config = locals()

    run = wandb.init(
        project="deformer",
        job_type="eval_continuous_discrete_deformer",
        config=config,
    )

    ds = tfds.load(dataset, split="test")

    if num_instances is not None:
        ds = ds.take(num_instances)

    ds = ds.batch(batch_size, drop_remainder=True)

    add_mask_fn = get_add_mask_fn(BernoulliMaskGenerator())
    ds = ds.map(add_mask_fn)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    model_artifact = run.use_artifact(model_artifact)
    model_dir = model_artifact.download()

    with open(os.path.join(model_dir, "model_config.json"), "r") as fp:
        model_config = json.load(fp)

    with open(os.path.join(model_dir, "state.pkl"), "rb") as fp:
        model_state = pickle.load(fp)

    def eval_fn(batch):
        model = ContinuousDiscreteDEformer(**model_config)

        x = batch["features"]
        b = batch["mask"]

        def eval_single_order(key):
            noise = jax.random.uniform(key, x.shape) - b
            order = jnp.argsort(noise, axis=-1)

            lls = model(x, order, is_training=False)
            joint_ll = jnp.sum(lls, axis=-1)
            ac_ll = jnp.sum(lls * (1 - b), axis=-1)

            imputations = model.impute(x, b, order)

            return joint_ll, ac_ll, imputations

        keys = jnp.asarray(hk.next_rng_keys(num_permutations))
        joint_ll, ac_ll, imputations = jax.vmap(eval_single_order)(keys)
        joint_ll = reduce_logmeanexp(joint_ll, axis=0)
        ac_ll = reduce_logmeanexp(ac_ll, axis=0)

        return joint_ll, ac_ll, imputations

    eval_fn = jax.jit(hk.transform_with_state(eval_fn).apply)

    prng = hk.PRNGSequence(91)

    joint_lls, ac_lls, imputations, bs, data = [], [], [], [], []

    for i in range(num_masks):
        joint_lls.append([])
        ac_lls.append([])
        imputations.append([])
        bs.append([])

        for batch in tqdm(
            ds.as_numpy_iterator(),
            desc=f"Trial {i + 1}/{num_masks}",
            total=ds.cardinality().numpy().item(),
        ):
            (joint, ac, imp), _ = eval_fn(
                model_state.params, model_state.state, prng.next(), batch
            )

            joint_lls[-1].append(jax.device_get(joint))
            ac_lls[-1].append(jax.device_get(ac))
            imputations[-1].append(jax.device_get(imp))
            bs[-1].append(batch["mask"])

            if i == 0:
                data.append(batch["features"])

        joint_lls[-1] = np.concatenate(joint_lls[-1], axis=-1)
        ac_lls[-1] = np.concatenate(ac_lls[-1], axis=-1)
        imputations[-1] = np.concatenate(imputations[-1], axis=1)
        bs[-1] = np.concatenate(bs[-1], axis=0)

        if i == 0:
            data = np.concatenate(data, axis=0)

    joint_lls = np.asarray(joint_lls)
    ac_lls = np.asarray(ac_lls)
    imputations = np.asarray(imputations)
    bs = np.asarray(bs)

    classes_per_feature = np.asarray(model_config["classes_per_feature"])

    discrete_imputations = imputations[..., classes_per_feature > 1]
    continuous_imputations = imputations[..., classes_per_feature <= 1]
    discrete_data = data[..., classes_per_feature > 1]
    continuous_data = data[..., classes_per_feature <= 1]
    discrete_b = bs[..., classes_per_feature > 1]
    continuous_b = bs[..., classes_per_feature <= 1]

    continuous_imputations = np.mean(continuous_imputations, axis=1)
    discrete_imputations = np.squeeze(mode(discrete_imputations, axis=1).mode, axis=1)

    continuous_data_std = np.std(continuous_data, axis=0, keepdims=True)

    error = (np.expand_dims(continuous_data, 0) - continuous_imputations) ** 2 * (
        1 - continuous_b
    )
    mse = np.sum(error, axis=-2) / np.count_nonzero(1 - continuous_b, axis=-2)
    nrmse = np.sqrt(mse) / continuous_data_std

    correct = np.equal(discrete_imputations, np.expand_dims(discrete_data, 0)) * (
        1 - discrete_b
    )
    accuracy = np.sum(correct, axis=(-1, -2)) / np.count_nonzero(
        1.0 - discrete_b, axis=(-1, -2)
    )

    results = {
        "joint_ll": np.mean(joint_lls).item(),
        "ac_ll": np.mean(ac_lls).item(),
        "ac_ll_std": np.std(np.mean(ac_lls, axis=1)).item(),
        "nrmse": np.mean(nrmse).item(),
        "nrmse_std": np.std(np.mean(nrmse, axis=1)).item(),
        "accuracy": np.mean(accuracy).item(),
        "accuracy_std": np.std(accuracy).item(),
    }

    run.log(results)


if __name__ == "__main__":
    main()
