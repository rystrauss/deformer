import json
import os

import click
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from bax import Trainer
from bax.callbacks import CheckpointCallback, WandbCallback, LearningRateLoggerCallback

import wandb
from models import DEformer
from utils import add_weight_decay

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
tf.config.set_visible_devices([], "GPU")


def load_datasets(dataset, batch_size):
    train_ds = tfds.load(dataset, split="train")
    val_ds = tfds.load(dataset, split="validation")

    def format_data(d):
        return d["features"]

    train_ds = train_ds.map(format_data).cache()
    val_ds = val_ds.map(format_data).cache()

    num_features = train_ds.element_spec.shape[0]

    train_ds = train_ds.shuffle(30000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.batch(batch_size, drop_remainder=True)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_features


@click.command()
@click.option(
    "--dataset", type=click.STRING, required=True, help="The dataset to train on."
)
@click.option(
    "--steps",
    type=click.INT,
    default=2000000,
    help="The number of training iterations.",
)
@click.option(
    "--batch_size", type=click.INT, default=128, help="The (per-device) batch size."
)
@click.option(
    "--num_layers",
    type=click.INT,
    default=6,
    help="The number of self-attention layers.",
)
@click.option(
    "--mlp_hidden_units",
    type=click.STRING,
    default="128,256,512",
    help="Comma-separated integers specifying the number of hidden units in each"
    "layer of the pre-attention MLPs. Note that the size of final layer will be used as"
    "the hidden dimension of the model.",
)
@click.option(
    "--index_embedding_dim",
    type=click.INT,
    default=16,
    help="The dimensionality of the index embeddings.",
)
@click.option(
    "--mixture_components",
    type=click.INT,
    default=100,
    help="The number of GMM mixture components.",
)
@click.option(
    "--num_heads",
    type=click.INT,
    default=8,
    help="The number of heads used in multi-head attention.",
)
@click.option("--dropout", type=click.FLOAT, default=0.2, help="The dropout rate.")
@click.option(
    "--widening_factor", type=click.INT, default=4, help="The MLP widening factor."
)
@click.option(
    "--data_noise",
    type=click.FLOAT,
    default=0.0,
    help="The scale of Gaussian noise to add to the data during training.",
)
@click.option(
    "--lr",
    type=click.FLOAT,
    default=5e-5,
    help="The initial learning rate. Note that this will be scaled according "
    "to the batch size.",
)
@click.option(
    "--weight_decay", type=click.FLOAT, default=0.0, help="Weight decay strength."
)
@click.option(
    "--validation_freq", type=click.INT, default=5000, help="The validation frequency."
)
@click.option(
    "--lr_boundaries",
    type=click.STRING,
    default="1500000",
    help="Comma-separated integers specifying the iteration boundaries at which the "
    "learning rate will be decayed by a factor of 10.",
)
@click.option(
    "--offline", is_flag=True, help="If flag is set, W&B will run in offline mode."
)
def main(
    dataset,
    steps,
    batch_size,
    num_layers,
    mlp_hidden_units,
    index_embedding_dim,
    mixture_components,
    num_heads,
    dropout,
    widening_factor,
    data_noise,
    lr,
    weight_decay,
    validation_freq,
    lr_boundaries,
    offline,
):
    mlp_hidden_units = tuple(map(int, mlp_hidden_units.split(",")))
    lr_boundaries = tuple(map(int, lr_boundaries.split(",")))
    config = locals()
    del config["offline"]

    run = wandb.init(
        project="deformer",
        job_type="train_deformer",
        mode="disabled" if offline else "online",
        config=config,
    )

    train_ds, val_ds, num_features = load_datasets(dataset, batch_size)

    model_config = dict(
        num_features=num_features,
        num_layers=num_layers,
        mlp_hidden_units=mlp_hidden_units,
        index_embedding_dim=index_embedding_dim,
        mixture_components=mixture_components,
        num_heads=num_heads,
        dropout=dropout,
        widening_factor=widening_factor,
    )

    def loss_fn(step, is_training, x):
        model = DEformer(**model_config)

        order = jax.random.uniform(hk.next_rng_key(), x.shape[:2])
        order = jnp.argsort(order, axis=-1)

        if data_noise != 0:
            x += (
                jax.random.normal(hk.next_rng_key(), shape=x.shape)
                * data_noise
                * is_training
            )

        dist = model(x, order, is_training)
        lls = jnp.sum(dist.log_prob(x), axis=-1)

        loss = -jnp.mean(lls)

        return loss, {}

    boundaries_and_scales = {i: 0.1 for i in lr_boundaries}
    scaled_lr = (lr * batch_size * jax.local_device_count()) / 256
    schedule = optax.piecewise_constant_schedule(scaled_lr, boundaries_and_scales)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        add_weight_decay(
            weight_decay, exclude_names=["index_embeddings", "class_embeddings"]
        ),
        optax.scale_by_schedule(schedule),
        optax.scale(-1),
    )

    trainer = Trainer(
        loss_fn,
        optimizer,
        num_devices=jax.local_device_count(),
    )

    callbacks = [
        CheckpointCallback(os.path.join(run.dir, "state.pkl")),
        LearningRateLoggerCallback(schedule),
        WandbCallback(run),
    ]

    train_state = trainer.fit(
        train_ds,
        steps,
        val_dataset=val_ds,
        validation_freq=validation_freq,
        callbacks=callbacks,
    )

    with open(os.path.join(run.dir, "model_config.json"), "w") as fp:
        json.dump(model_config, fp)

    state_artifact = wandb.Artifact(f"{dataset}_deformer", type="deformer_model")
    state_artifact.add_file(os.path.join(run.dir, "state.pkl"))
    state_artifact.add_file(os.path.join(run.dir, "model_config.json"))
    run.log_artifact(state_artifact)


if __name__ == "__main__":
    main()
