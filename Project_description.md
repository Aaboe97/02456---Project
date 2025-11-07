# Project Description

**Implementing a Neural Network from Scratch with NumPy: Training, Optimization, and Experiment Tracking with Weights & Biases (WandB)**

## Objective

The goal of this project is to design and train a fully-connected feedforward neural network (FFN) from scratch using only NumPy, without relying on deep learning libraries such as TensorFlow or PyTorch. Implement forward and backward propagation, gradient descent optimization, and evaluate the model on small-scale classification datasets — including both tabular and image-based tasks.

## Datasets

Two open-source datasets are recommended:

- **Fashion-MNIST**
- **CIFAR-10**

These datasets are small enough for CPU-based NumPy training and sufficiently complex to illustrate overfitting, regularization, and optimizer effects. But if you find any interesting open-source data in Kaggle feel free to use them.

## Methodology

Implement a flexible FFNN class with the following configurable hyperparameters:

- `num_epochs`
- `num_hidden_layers` 
- `n_hidden_units`
- `learning_rate`
- `optimizer`
- `batch_size`
- `l2_coeff`
- `weights_init`
- `activation`
- `loss`
- etc.

## Implementation Stages

1. **Forward pass**: matrix multiplications + activation functions
2. **Loss computation**: MSE or cross-entropy with L2 regularization
3. **Backward pass**: manual derivative calculation and weight updates
4. **Training loop**: mini-batch gradient descent
5. **Evaluation**: compute accuracy, loss curves, and confusion matrices

## Experiment Logging with WandB

Each training run will be logged to Weights & Biases, including:

- Learning curves (train_loss, val_loss, accuracy, val_acc)
- Parameter histograms and gradient norms
- Hyperparameter sweeps (random or Bayesian) across architectures and optimizers
- Summary reports comparing activation functions and initializations

---

**This project will be supervised by Viswanathan Sankar (viswa@dtu.dk)**
