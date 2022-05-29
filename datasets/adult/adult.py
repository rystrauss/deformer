import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_LINK = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
TEST_LINK = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


class Adult(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="The UCI Adult dataset, with preprocessing as described in "
            "'Arbitrary Conditional Distributions with Energy'.",
            features=tfds.features.FeaturesDict(
                {"features": tfds.features.Tensor(shape=(14,), dtype=tf.float32)}
            ),
            supervised_keys=None,
            homepage="https://archive.ics.uci.edu/ml/datasets/adult",
            citation=None,
            metadata=tfds.core.MetadataDict(
                classes_per_feature=[0, 8, 0, 16, 0, 7, 14, 6, 5, 2, 0, 0, 0, 2]
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        train_path = dl_manager.download(TRAIN_LINK)
        test_path = dl_manager.download(TEST_LINK)

        train, val, test = _load_and_preprocess_adult(train_path, test_path)

        return {
            "train": self._generate_examples(train),
            "validation": self._generate_examples(val),
            "test": self._generate_examples(test),
        }

    def _generate_examples(self, data):
        for i, x in enumerate(data):
            yield i, dict(features=x)


def _load_and_preprocess_adult(train_path, test_path):
    train = pd.read_csv(
        train_path,
        header=None,
        na_values="?",
        sep=", ",
        engine="python",
        dtype={
            1: "category",
            3: "category",
            5: "category",
            6: "category",
            7: "category",
            8: "category",
            9: "category",
            13: "category",
            14: "category",
        },
    )

    test = pd.read_csv(
        test_path,
        header=None,
        na_values="?",
        sep=", ",
        engine="python",
        skiprows=1,
        dtype={
            1: "category",
            3: "category",
            5: "category",
            6: "category",
            7: "category",
            8: "category",
            9: "category",
            13: "category",
            14: "category",
        },
    )

    train = train.dropna()
    test = test.dropna()

    test[14] = test[14].apply(lambda s: s[:-1])

    for i in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
        mapping = {j: i for i, j in enumerate(train[i].cat.categories)}
        train[i] = train[i].cat.rename_categories(mapping)
        test[i] = test[i].cat.rename_categories(mapping)

    for i in [0, 2, 4, 10, 11, 12]:
        mu = train[i].mean()
        std = train[i].std()
        train[i] = (train[i] - mu) / std
        test[i] = (test[i] - mu) / std

    train = train[train[13] == 38]
    test = test[test[13] == 38]

    train[13] = train[14]
    train = train.drop(14, axis=1)

    test[13] = test[14]
    test = test.drop(14, axis=1)

    train = train.sample(frac=1, random_state=91)

    p = int(0.8 * len(train))

    val = train.iloc[p:]
    train = train.iloc[:p]

    train = train.to_numpy().astype(np.float32)
    val = val.to_numpy().astype(np.float32)
    test = test.to_numpy().astype(np.float32)

    return train, val, test
