#%% 1. Import Required Libraries and Configuration

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import time


# Dataset Configuration
num_features = 28 * 28  # MNIST: 28x28 pixels
num_classes = 10        # MNIST: digits 0-9

# Architecture Configuration
num_hidden_layers = 2   # Number of hidden layers
hidden_units = [32, 32] # Units per hidden layer [layer1, layer2, ...]
activation = 'relu'     # Activation function: 'relu', 'tanh', 'sigmoid'
weights_init = 'he'     # Weight initialization: 'he', 'xavier', 'normal'

# Training Configuration  
num_epochs = 100        # Number of training epochs
learning_rate = 0.001   # Learning rate for gradient descent
batch_size = 32         # Mini-batch size
loss = 'cross_entropy'  # Loss function: 'cross_entropy', 'mse', 'mae'




#%% 2. Load MNIST Data

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize inputs 
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# One-hot encode labels
T_train = to_categorical(y_train, num_classes=10)
T_test = to_categorical(y_test, num_classes=10)





#%% 3. Initialize Network Parameters

class PyNet_M10:
    def __init__(self, num_features, hidden_units, num_output, weights_init='he', activation='relu', loss='cross_entropy'):
        """
        Initialize neural network with configurable architecture.
        
        Args:
            num_features: Number of input features
            hidden_units: List of hidden layer sizes [layer1, layer2, ...]
            num_output: Number of output classes
            weights_init: Weight initialization method ('he', 'xavier', 'normal')
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            loss: Loss function ('cross_entropy', 'mse', 'mae')
        """
        
        # Build layer sizes: input → hidden layers → output
        layer_sizes = [num_features] + hidden_units + [num_output]
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights_init = weights_init
        self.loss = loss
        
            
        # Initialize weights for each layer
        self.W = []
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Weight initialization
            if weights_init == 'he':
                # He initialization (good for ReLU)
                w = np.random.randn(input_size + 1, output_size) * np.sqrt(2 / input_size)
            elif weights_init == 'xavier':
                # Xavier initialization (good for tanh/sigmoid)
                w = np.random.randn(input_size + 1, output_size) * np.sqrt(1 / input_size)
            elif weights_init == 'normal':
                # Standard normal initialization
                w = np.random.randn(input_size + 1, output_size) * 0.01
            else:
                raise ValueError(f"Unknown weights_init: {weights_init}")
            
            self.W.append(w)

    
    def forward(self, X, W):
        h = []
        a = X
        for l in range(len(W) - 1):
            a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
            z = W[l].T @ a
            a = self.activate(z)  # Use configurable activation
            h.append(a)
        a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
        y_hat = W[-1].T @ a
        y = self.softmax(y_hat)  # Output layer always uses softmax for classification
        return y, h


    def backward(self, X, T, W, h, eta, y_pred=None):
        
        m = X.shape[1] 

        if y_pred is None: # Use pre-computed predictions if available, otherwise compute them
            y, _ = self.forward(X, W)
        else:
            y = y_pred

        delta = self.loss_derivative(y, T)  # Use configurable loss derivative
        for l in range(len(W) - 1, 0, -1):
            a_prev = np.vstack([h[l-1], np.ones(h[l-1].shape[1])])  # Add bias term
            Q = a_prev @ delta.T
            W[l] -= (eta / m) * Q
            delta = W[l][:-1, :] @ delta
            delta *= self.activation_derivative(h[l-1])  # Use configurable activation derivative
        a_prev = np.vstack([X, np.ones(X.shape[1])])  # Add bias term
        Q = a_prev @ delta.T
        W[0] -= eta * Q
        loss = self.calculate_loss(y, T)
        return W, loss
    

    def softmax(self, y_hat):
        """Compute softmax probabilities"""
        y_hat = y_hat - np.max(y_hat, axis=0, keepdims=True)  # prevent overflow
        exp_scores = np.exp(y_hat)
        return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    

    def activate(self, z):
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        

    def activation_derivative(self, a):
        """Calculate derivative of activation function"""
        if self.activation == 'relu':
            return a > 0
        elif self.activation == 'tanh':
            return 1 - a**2
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    def calculate_loss(self, y_pred, y_true):
        """Calculate loss based on configured loss function"""
        epsilon = 1e-12  # Prevent log(0)
        
        if self.loss == 'cross_entropy':
            # Categorical Cross-Entropy Loss
            return -np.sum(np.log(np.sum(y_pred * y_true, axis=0) + epsilon))
        elif self.loss == 'mse':
            # Mean Squared Error Loss
            return 0.5 * np.sum((y_pred - y_true) ** 2)
        elif self.loss == 'mae':
            # Mean Absolute Error Loss
            return np.sum(np.abs(y_pred - y_true))
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
            return np.sign(y_pred - y_true)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    

net = PyNet_M10(num_features, hidden_units, num_classes, weights_init, activation, loss)






#%% 4. Training Loop

def calculate_accuracy(X, T, W):
    """Calculate accuracy percentage"""
    y, _ = net.forward(X, W)
    predictions = np.argmax(y, axis=0)
    true_labels = np.argmax(T, axis=0)
    return np.mean(predictions == true_labels) * 100

def train(X, T, W, epochs, eta, batchsize=32):
    
    losses = []
    accuracies = []  # Track training accuracy
    epoch_times = []  # Track computation time per epoch
    
    # Print header for nicely formatted table
    print("-" * 51)
    print(f"{'Epoch':<10} {'Accuracy':<10} {'Time':<10} {'ETA'}")
    print("-" * 51)
    
    start_total = time.time()
    
    m = X.shape[1]
    for epoch in range(epochs):
        epoch_start = time.time()  # Start timing this epoch
        
        order = np.random.permutation(m)
        epoch_loss = 0
        for i in range(0, m, batchsize):
            batch = order[i:i+batchsize]
            X_batch = X[:, batch]
            T_batch = T[:, batch]
            y_batch, h = net.forward(X_batch, W)
            W, loss = net.backward(X_batch, T_batch, W, h, eta, y_batch)  # Pass pre-computed predictions
            epoch_loss += loss

        # Calculate training accuracy for this epoch
        train_accuracy = calculate_accuracy(X, T, W)
        accuracies.append(train_accuracy)
        losses.append(epoch_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Calculate ETA (estimated time remaining)
        if epoch > 0:
            avg_time_per_epoch = np.mean(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs
            if eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.1f}min"
            else:
                eta_str = f"{eta_seconds:.0f}sec"
        else:
            eta_str = "calculating..."
        
        # Format epoch info for output
        epoch_str = f"{epoch+1}/{epochs}"
        accuracy_str = f"{train_accuracy:.2f}%"
        time_str = f"{epoch_time:.2f}sec"
        
        # Show progress 
        print(f"{epoch_str:<10} {accuracy_str:<10} {time_str:<10} {eta_str}")
    
    total_time = time.time() - start_total
    avg_epoch_time = np.mean(epoch_times)
    
    print("-" * 51)
    print(f"Total training time: {total_time:.1f}sec")
    print(f"Average per epoch: {avg_epoch_time:.2f}sec")
    print("-" * 51)
    return W, losses, accuracies


# Train the Model
net.W, losses, train_accuracies = train(X_train.T, T_train.T, net.W, num_epochs, learning_rate, batch_size)




#%% 5. Evaluate the Model

def predict(X, W):
    y, _ = net.forward(X, W)
    return np.argmax(y, axis=0)

# Test the model
y_pred = predict(X_test.T, net.W)
accuracy = np.mean(y_pred == y_test)

# Calculate test loss using the configurable loss function
y_test_pred, _ = net.forward(X_test.T, net.W)
test_loss = net.calculate_loss(y_test_pred, T_test.T) / X_test.shape[0]  # Average per sample

print(f"\n================== Final Results ==================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss (avg per sample): {test_loss:.4f}")
print(f"Training Accuracy Improvement: {(train_accuracies[-1] - train_accuracies[0]):.1f}% points")
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
