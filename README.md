[1]: https://arxiv.org/abs/2106.06989
[2]: https://github.com/airalcorn2/deformer
[3]: https://arxiv.org/abs/2102.04426
[4]: https://arxiv.org/pdf/1909.06319.pdf
[5]: https://arxiv.org/abs/2004.02441

# DEformer

This repository provides a JAX implementation of the [DEformer][1] model (the authors'
original implementation can be found [here][2]).

DEformer is an order-agnostic autoregressive model, meaning that it can factorize
the joint likelihood in any order across the features. This is in contrast to typical
autoregressive models which always represent the joint likelihood as e.g.:
$$p(\mathbf{x}) = \prod_{i=1}^D p(x_i \mid \mathbf{x}_{<i})$$

While DEformer is ostensibly trained to only maximize the likelihood of the joint
distribution, the fact that it can do so using any ordering in the chain rule means
that it is actually capable of modeling a far wider range of distributions. We have the
ability to obtain likelihoods (and samples) for any arbitrary (multi-dimensional)
conditional distribution over the features. In other words, we can do
_arbitrary conditioning_.

This repository contains code for training and evaluating DEformer on standard
benchmarks for both joint density estimation and arbitrary conditional density
estimation. We also provide a new variant of DEformer that can operate on datasets with
a mixture of continuous and discrete features. We find that DEformer achieves
state-of-the-art performance for arbitrary conditioning tasks, relative to recent
models such as [ACE][3] and [ACFlow][4], and that DEformer is also comparable to recent
single-order auto-regressive models for joint density estimation (e.g. [TraDE][5]).
See results below.

## Results

### Joint Likelihoods
|                      | Power |  Gas   | Hepmass | Miniboone  |  BSDS   |
|:---------------------|:-----:|:------:|:-------:|:----------:|:-------:|
| DEformer             | 0.541 | 13.167 | **-11.983** | **-9.323** | 157.035 |
| ACE                  | 0.576 | 12.201 | -15.041 |  -11.407   | 156.400 |
| ACE Proposal         | 0.488 | 11.840 | -15.697 |  -12.109   | 154.349 |
| TraDE (single-order) | **0.73**  | **13.27**  | -12.01  |   -9.49    | **160.01**  |

### Arbitrary Conditioning Likelihoods
|                      |     Power      |  Gas   | Hepmass | Miniboone  |  BSDS   |
|:---------------------|:--------------:|:------:|:-------:|:----------:|:-------:|
| DEformer             | **0.641 ± 0.002**  | **10.272 ± 0.006** | **-0.899 ± 0.005** | **1.758 ± 0.044** | **87.01 ± 0.016** |
| ACE                  | 0.631 ± 0.002  | 9.643 ± 0.005 | -3.859 ± 0.005 |  0.310 ± 0.054   | 86.701 ± 0.008 |
| ACE Proposal         | 0.583 ± 0.003  | 9.484 ± 0.005 | -4.417 ± 0.005 |  -0.241 ± 0.056   | 85.228 ± 0.009 |

### Imputation Normalized Root-Mean-Square Error
|                      |     Power      |  Gas   | Hepmass | Miniboone  |  BSDS   |
|:---------------------|:--------------:|:------:|:-------:|:----------:|:-------:|
| DEformer             | **0.623 ± 0.002**  | **0.232 ± 0.022** | **0.514 ± 0.000** | **0.239 ± 0.002** | **0.300 ± 0.000** |
| ACE                  | 0.828 ± 0.002  | 0.335 ± 0.027 | 0.830 ± 0.001 |  0.432 ± 0.003   | 0.525 ± 0.000 |
| ACE Proposal         | 0.828 ± 0.002  | 0.312 ± 0.033 | 0.832 ± 0.001 |  0.436 ± 0.004   | 0.535 ± 0.000 |

### Mixed Continuous-Discrete Data (UCI Adult)
|              |  LL   | NRMSE | Accuray |
|:-------------|:-----:|:-----:|:-------:|
| DEformer     | **2.66**  | **0.88**  |  **0.70**   |
| ACE          | 2.38  | 0.90  |  0.69   |
| ACE Proposal | 2.24  | 0.89  |  0.69   |
| VAEAC        | -7.25 | 0.91  |  0.67   |

## Setup

This code relies on several packages, which we detail in this section. The most
important step is making sure you have versions of CUDA software and JAX that are
compatible with each other. These versions can vary depending on
your machine and as the software continues to be updated. Please refer to the
[JAX installation instructions](https://github.com/google/jax#installation) for details.
If using TPUs, you will want to install the TPU version of JAX.

After installing JAX and CUDA (is using GPUs), the remaining dependencies can be
installed via `pip`:
```
pip install tensorflow tensorflow_probability tensorflow_datasets
pip install dm-haiku optax chex>=0.1.3 distrax
pip install bax==0.1.11
pip install wandb click gdown numpy scipy pandas einops tqdm
```

### Datasets

The benchmark UCI datasets are provided as TensorFlow Datasets in
[`datasets`](datasets). Currently, the following datasets are provided:
- Adult
- Power
- Gas
- Miniboone
- Hepmass
- BSDS

Before trying to use a dataset, you should build it by navigating to the directory of
the  dataset you wish to use, then running `tfds build`. Note that `gdown` needs to
be installed before some of the datasets can be built.

If you would like to add your own new dataset, please see
[this guide](https://www.tensorflow.org/datasets/add_dataset) for instructions on how
to create one. You can also refer to the existing directories inside
[`datasets`](datasets) for examples.

### Weights and Biases

This code uses [Weights and Biases](https://wandb.ai/site) for saving experiments and
artifacts. Using Weights and Biases requires first making an account. Once you have an
account, you then need to run
```
wandb login
```
on your machine before running any of the code in this repository.

## Usage

The `train_deformer.py` and `train_continuous_discrete_deformer.py`
scripts can be used to train new models on some data. The former script
assumes that all features are continuous, whereas the latter is for a dataset like
[`adult`](datasets/adult) which has both continuous and discrete features.
An example command for training the model is:
```
python train_deformer.py --dataset gas --batch_size 256
```
Note that DEformer models are best trained with large total batch sizes (e.g. 2048)
and can be pretty compute intensive to train to convergence. If possible, it is best
to train these model with multiple accelerators. Also, note that the `--batch_size`
argument for the script is the _per-device_ batch size, so if you are training on e.g.
8 TPUs and want a total batch size of 2048, then you should set `--batch_size=256`.
The training scripts will automatically use all visible accelerators. The training
scripts will automatically save the model config and model weights to W&B as an
artifact.

After training a model, you can then evaluate it with `eval_deformer.py` or
`eval_continuous_discrete_deformer.py`. These scripts will produce the metrics given in
the tables above. An example command for evaluating a model is:
```
python eval_deformer.py --dataset gas --model_artifact gas_deformer:v0
```
where you can replace `gas_deformer:v0` with the name of the model artifact (in W&B)
that you wish to evaluate.