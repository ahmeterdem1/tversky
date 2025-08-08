# Tversky Neural Networks

This repository is *an* implementation of Tversky Neural Networks in PyTorch, 
following the recent paper [Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity](https://arxiv.org/abs/2506.11035).

Intersection and set difference operations are implemented as separate classes.
Paper proposes to use a lot of aggregation functions for intersection, but 
only several of them are implemented here.

Both set difference methods are implemented.

A LeNet like backbone and a single Tversky layer is combined to form a simple classifier,
to be trained and tested on MNIST. The model architecture does not follow the exact
implementation of the paper (number of layers, hyperparameters, etc.). Short tests on
MNIST and CIFAR10 have been conducted here to try to reproduce the paper. Results are
averaged over 5 random runs, 32 batch size. 

| Model (Initialization) | Dataset  | Accuracy        |
|------------------------|----------|-----------------|
| Base Model             | MNIST    | 0.9875 ± 0.0012 |
| Tversky (Normal)       | MNIST    | 0.9822 ± 0.0029 |
| Tversky (Uniform)      | MNIST    | 0.9856 ± 0.0021 |
| Tversky (Orthogonal)   | MNIST    | 0.9865 ± 0.0014 |
| Base Model             | CIFAR10  | 0.6723 ± 0.0060 |
| Tversky (Normal)       | CIFAR10  | 0.6307 ± 0.0130 |
| Tversky (Uniform)      | CIFAR10  | 0.6452 ± 0.0164 |
| Tversky (Orthogonal)   | CIFAR10  | 0.4393 ± 0.2771 |

As it is also reported in the paper (Table 3), Tversky layers as a replacement for
a "classification head" did not improve results. Test setups are very different
from the paper.
