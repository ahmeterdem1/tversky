# Tversky Neural Networks

This repository is *an* implementation of Tversky Neural Networks in PyTorch, 
following the recent paper [Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity](https://arxiv.org/abs/2506.11035).

Intersection and set difference operations are implemented as separate classes.
Paper proposes to use a lot of aggregation functions for intersection, but 
only several of them are implemented here.

Both set difference methods are implemented.

A LeNet like backbone and a single Tversky layer is combined to form a simple classifier,
to be trained and tested on MNIST. The model architecture does not follow the exact
implementation of the paper (number of layers, hyperparameters, etc.). The resulting
layer after only 5 epochs of training achieves 97.88% accuracy on the test set.
Therefore, we were able to reproduce the results of the paper for this specific case. 
