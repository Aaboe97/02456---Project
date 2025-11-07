#%% JAXNet Shared Functions Module
"""
Shared functions for JAXNet neural network implementations.
Contains all common functionality used by both M10 (MNIST) and E26 (EMNIST) networks.
JAX version of PyNet.py with identical structure and functionality.
"""

import time
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial

class JAXNetBase:
    """Base class containing all shared neural network functionality"""
    
    def __init__(self, num_features, hidden_units, num_output, weights_init='he', activation='relu', loss='cross_entropy', seed=42):
        """
        Initialize neural network with configurable architecture.
        
        Args:
            num_features: Number of input features
            hidden_units: List of hidden layer sizes [layer1, layer2, ...]
            num_output: Number of output classes
            weights_init: Weight initialization method ('he', 'xavier', 'normal')
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            loss: Loss function ('cross_entropy', 'mse', 'mae')
            seed: Random seed for weight initialization
        """
        
        # Build layer sizes: input → hidden layers → output
        layer_sizes = [num_features] + hidden_units + [num_output]
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights_init = weights_init
        self.loss = loss
        self.seed = seed
        
        # Initialize weights for each layer
        self.W = []
        key = random.PRNGKey(seed)
        
        for i in range(len(layer_sizes) - 1):
            key, subkey = random.split(key)
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Weight initialization
            if weights_init == 'he':
                # He initialization (good for ReLU)
                w = random.normal(subkey, (input_size + 1, output_size)) * jnp.sqrt(2 / input_size)
            elif weights_init == 'xavier':
                # Xavier initialization (good for tanh/sigmoid)
                w = random.normal(subkey, (input_size + 1, output_size)) * jnp.sqrt(1 / input_size)
            elif weights_init == 'normal':
                # Standard normal initialization
                w = random.normal(subkey, (input_size + 1, output_size)) * 0.01
            else:
                raise ValueError(f"Unknown weights_init: {weights_init}")
            
            self.W.append(w)


    def forward(self, X, W):
        """
        Forward pass through the network

        Args:
            X: Input data
            W: Weights
        Returns:
            y: Output predictions
            h: List of hidden layer activations
        """
        h = []
        a = X
        for l in range(len(W) - 1):
            a = jnp.vstack([a, jnp.ones((1, a.shape[1]))])  # Add bias term
            z = W[l].T @ a
            a = self.activation_function(z)  # Use configurable activation
            h.append(a)
        a = jnp.vstack([a, jnp.ones((1, a.shape[1]))])  # Add bias term
        y_hat = W[-1].T @ a
        y = self.softmax(y_hat)  # Output layer always uses softmax for classification
        return y, h
    

    def backward(self, X, T, W, h, eta, y_pred=None, use_clipping=True, max_grad_norm=25.0):
        """
        Backward pass with optional gradient clipping and stability checks.
        
        Args:
            X: Input data
            T: Target labels
            W: Weights
            h: Hidden activations from forward pass
            eta: Learning rate
            y_pred: Pre-computed predictions (optional, for efficiency)
            use_clipping: Whether to use gradient clipping (default True)
            max_grad_norm: Maximum gradient norm for clipping (default 25.0)
        """
        m = X.shape[1]
        
        if y_pred is None:  # Use pre-computed predictions if available, otherwise compute them
            y, _ = self.forward(X, W)
        else:
            y = y_pred
            
        delta = self.loss_derivative(y, T)  # Use configurable loss derivative
        
        # Create new weights list (JAX arrays are immutable)
        new_W = []
        
        for l in range(len(W) - 1, 0, -1):
            a_prev = jnp.vstack([h[l-1], jnp.ones((1, h[l-1].shape[1]))])  # Add bias term
            Q = a_prev @ delta.T
            
            # Optional gradient clipping
            if use_clipping:
                grad_norm = jnp.linalg.norm(Q)
                Q = jnp.where(grad_norm > max_grad_norm, Q * (max_grad_norm / grad_norm), Q)
            
            new_W.insert(0, W[l] - (eta / m) * Q)
            delta = W[l][:-1, :] @ delta
            delta = delta * self.activation_derivative(h[l-1])  # Use configurable activation derivative
            
        a_prev = jnp.vstack([X, jnp.ones((1, X.shape[1]))])  # Add bias term
        Q = a_prev @ delta.T
        
        # Optional gradient clipping for first layer
        if use_clipping:
            grad_norm = jnp.linalg.norm(Q)
            Q = jnp.where(grad_norm > max_grad_norm, Q * (max_grad_norm / grad_norm), Q)
        
        new_W.insert(0, W[0] - (eta / m) * Q)
        loss = self.loss_function(y, T)
        return new_W, loss
    

    def softmax(self, y_hat):
        """Compute softmax probabilities"""
        y_hat = y_hat - jnp.max(y_hat, axis=0, keepdims=True)  # prevent overflow
        exp_scores = jnp.exp(y_hat)
        return exp_scores / jnp.sum(exp_scores, axis=0, keepdims=True)
    

    def activation_function(self, z):
        """Apply activation function"""
        if self.activation == 'relu':
            return jnp.maximum(0, z)
        elif self.activation == 'tanh':
            return jnp.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + jnp.exp(-jnp.clip(z, -500, 500)))  # Clip to prevent overflow
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        

    def activation_derivative(self, a):
        """Calculate derivative of activation function"""
        if self.activation == 'relu':
            return (a > 0).astype(jnp.float32)
        elif self.activation == 'tanh':
            return 1 - a**2
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        

    def loss_function(self, y_pred, y_true):
        """Calculate loss based on configured loss function"""
        epsilon = 1e-12  # Prevent log(0)
        
        if self.loss == 'cross_entropy':
            # Categorical Cross-Entropy Loss
            return -jnp.sum(jnp.log(jnp.sum(y_pred * y_true, axis=0) + epsilon))
        elif self.loss == 'mse':
            # Mean Squared Error Loss
            return 0.5 * jnp.sum((y_pred - y_true) ** 2)
        elif self.loss == 'mae':
            # Mean Absolute Error Loss
            return jnp.sum(jnp.abs(y_pred - y_true))
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        

    def loss_derivative(self, y_pred, y_true):
        """Calculate derivative of loss function for backpropagation"""
        if self.loss == 'cross_entropy':
            # For cross-entropy with softmax: derivative is simply (y_pred - y_true)
            return y_pred - y_true
        elif self.loss == 'mse':
            # MSE derivative: (y_pred - y_true)
            return y_pred - y_true
        elif self.loss == 'mae':
            # MAE derivative: sign(y_pred - y_true)
            return jnp.sign(y_pred - y_true)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")


# Shared utility functions
def calculate_accuracy(net, X, T, W):
    """Calculate accuracy percentage"""
    y, _ = net.forward(X, W)
    predictions = jnp.argmax(y, axis=0)
    true_labels = jnp.argmax(T, axis=0)
    return jnp.mean(predictions == true_labels) * 100
    

def train(net, X, T, W, epochs, eta, batchsize=32, use_clipping=True, max_grad_norm=25.0):
    """
    Training loop for neural network.
    
    Args:
        net: Neural network instance
        X, T: Training data and labels
        W: Initial weights
        epochs: Number of training epochs
        eta: Learning rate
        batchsize: Mini-batch size
        use_clipping: Whether to use gradient clipping
        max_grad_norm: Maximum gradient norm for clipping
    """
    losses = []
    accuracies = []  # Track training accuracy
    epoch_times = []  # Track computation time per epoch
    
    # Print header for nicely formatted table
    print("-" * 70)
    print(f"{'Epoch':<10} {'Accuracy':<10} {'Gain':<10} {'Time':<10} {'ETA'}")
    print("-" * 70)
    
    start_total = time.time()

    m = X.shape[1]
    key = random.PRNGKey(42)  # For batch shuffling
    
    for epoch in range(epochs):
        epoch_start = time.time()  # Start timing this epoch
        
        key, subkey = random.split(key)
        order = random.permutation(subkey, m)
        epoch_loss = 0
        
        for i in range(0, m, batchsize):
            batch = order[i:i+batchsize]
            X_batch = X[:, batch]
            T_batch = T[:, batch]
            y_batch, h = net.forward(X_batch, W)
            W, loss = net.backward(X_batch, T_batch, W, h, eta, y_batch, use_clipping, max_grad_norm)
            epoch_loss += loss

        # Calculate training accuracy for this epoch
        train_accuracy = float(calculate_accuracy(net, X, T, W))  # Convert to Python float
        accuracies.append(train_accuracy)
        losses.append(float(epoch_loss))  # Convert to Python float
        
        # Calculate gain compared to last epoch
        if epoch > 0:
            gain = train_accuracy - accuracies[-2]  # Current - previous
            if gain > 0:
                gain_str = f"+{gain:.2f}%"
            elif gain < 0:
                gain_str = f"{gain:.2f}%"  # Already has negative sign
            else:
                gain_str = " 0.00%"
        else:
            gain_str = "baseline"
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Calculate ETA (estimated time remaining)
        if epoch > 0:
            avg_time_per_epoch = sum(epoch_times) / len(epoch_times)  # Use pure Python for averaging
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs
            if eta_seconds > 60:
                eta_str = f"{int(eta_seconds//60)}min {int(eta_seconds%60)}sec"
            else:
                eta_str = f"{int(eta_seconds)}sec"
        else:
            eta_str = "calculating..."
        
        # Format epoch info for output
        epoch_str = f"{epoch+1}/{epochs}"
        accuracy_str = f"{train_accuracy:.2f}%"
        time_str = f"{epoch_time:.2f}sec"
        
        # Show progress 
        print(f"{epoch_str:<10} {accuracy_str:<10} {gain_str:<10} {time_str:<10} {eta_str}")
    
    total_time = time.time() - start_total
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print("-" * 70)
    print(f"Total training time: {total_time:.1f}sec")
    print(f"Average per epoch: {avg_epoch_time:.2f}sec")
    print("-" * 70)
    return W, losses, accuracies

def evaluate_model(net, X_test, T_test, y_test, W, train_accuracies):
    """Evaluate model performance and print results"""
    # Make predictions and calculate accuracy
    y_test_pred, _ = net.forward(X_test.T, W)
    y_pred = jnp.argmax(y_test_pred, axis=0)
    test_accuracy = float(jnp.mean(y_pred == y_test))  # Convert to Python float

    # Calculate test loss using the configurable loss function
    test_loss = float(net.loss_function(y_test_pred, T_test.T) / X_test.shape[0])  # Average per sample

    print(f"\n================== Final Results ==================")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss (avg per sample): {test_loss:.4f}")
    print(f"Training Accuracy Improvement: {(train_accuracies[-1] - train_accuracies[0]):.1f}% points")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    
    return y_pred, test_accuracy, test_loss
