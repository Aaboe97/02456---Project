#%% 1. Import Required Libraries

import jax
import jax.numpy as jnp
from jax import random, grad, jit
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import time

def softmax(y_hat):
    y_hat = y_hat - jnp.max(y_hat, axis=0, keepdims=True)  # prevent overflow
    exp_scores = jnp.exp(y_hat)
    return exp_scores / jnp.sum(exp_scores, axis=0, keepdims=True)




#%% 2. Load MNIST Data

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize inputs 
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# One-hot encode labels
T_train = to_categorical(y_train, num_classes=10)
T_test = to_categorical(y_test, num_classes=10)

# Convert to JAX arrays
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
T_train = jnp.array(T_train)
T_test = jnp.array(T_test)


#%% 3. Initialize Network Parameters

def init(key, dims):
    keys = random.split(key, len(dims) - 1)
    W = []
    for i in range(len(dims) - 1):
        w = random.normal(keys[i], (dims[i] + 1, dims[i + 1])) * jnp.sqrt(2.0 / dims[i])
        W.append(w)
    return W

dims = [784, 32, 32, 10]  # Input layer (784), hidden layer (32, 32), output layer (10)
key = random.PRNGKey(42)
W = init(key, dims)




#%% 4. Define Forward Pass

def forward(X, W):
    h = []
    a = X
    for l in range(len(W) - 1):
        a = jnp.vstack([a, jnp.ones((1, a.shape[1]))])  # Add bias term
        z = W[l].T @ a
        a = jnp.maximum(0, z)  # ReLU activation
        h.append(a)
    a = jnp.vstack([a, jnp.ones((1, a.shape[1]))])  # Add bias term
    y_hat = W[-1].T @ a
    y = softmax(y_hat)  # Softmax
    return y, h




#%% 5. Define Loss Function and Gradients

def loss_fn(W, X, T):
    """Loss function for gradient computation"""
    y, _ = forward(X, W)
    epsilon = 1e-12
    loss = -jnp.sum(jnp.log(jnp.sum(y * T, axis=0) + epsilon))
    return loss

# Create optimized functions with JIT compilation
grad_fn = jit(grad(loss_fn))
forward_fn = jit(forward)
loss_fn_jit = jit(loss_fn)

@jit
def update_weights(W, grads, eta, m):
    """JIT-compiled weight update - match PyNet-M10 exactly"""
    W_new = []
    for i, (w, g) in enumerate(zip(W, grads)):
        if i == 0:  # First layer - different update rule like PyNet
            W_new.append(w - eta * g)
        else:       # Hidden layers - divide by batch size
            W_new.append(w - (eta / m) * g)
    return W_new

def backward(X, T, W, h, eta):
    """Backward pass using JAX gradients with JIT compilation"""
    m = X.shape[1]
    
    # Compute gradients using JIT-compiled function
    grads = grad_fn(W, X, T)
    
    # Update weights using JIT-compiled function - now matches PyNet exactly
    W_new = update_weights(W, grads, eta, m)
    
    # Calculate loss for tracking using JIT-compiled function
    loss = loss_fn_jit(W_new, X, T)
    
    return W_new, loss




#%% 6. Training Loop

@jit
def calculate_accuracy_jit(X, T, W):
    """JIT-compiled accuracy calculation"""
    y, _ = forward(X, W)
    predictions = jnp.argmax(y, axis=0)
    true_labels = jnp.argmax(T, axis=0)
    return jnp.mean(predictions == true_labels) * 100

def calculate_accuracy(X, T, W):
    """Calculate accuracy percentage"""
    return calculate_accuracy_jit(X, T, W)

def train(X, T, W, epochs, eta, batchsize=32):
    m = X.shape[1]
    losses = []
    accuracies = []  # Track training accuracy
    epoch_times = []  # Track computation time per epoch
    
    # Print header for nicely formatted table
    print("-" * 51)
    print(f"{'Epoch':<10} {'Accuracy':<10} {'Time':<10} {'ETA'}")
    print("-" * 51)
    
    # Warmup: trigger JIT compilation with a small batch
    print("Warming up JIT compilation...")
    warmup_batch = X[:, :32]
    warmup_T = T[:, :32]
    _ = forward_fn(warmup_batch, W)
    _ = grad_fn(W, warmup_batch, warmup_T)
    print("JIT compilation complete!")
    
    start_total = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()  # Start timing this epoch
        
        # Use NumPy for random permutation, then convert back to JAX  
        order = np.random.permutation(m)
        epoch_loss = 0
        for i in range(0, m, batchsize):
            batch_end = min(i + batchsize, m)
            batch_indices = order[i:batch_end]
            X_batch = X[:, batch_indices]
            T_batch = T[:, batch_indices]
            _, h = forward_fn(X_batch, W)  # Use JIT-compiled forward
            W, loss = backward(X_batch, T_batch, W, h, eta)
            epoch_loss += loss

        # Calculate training accuracy for this epoch
        train_accuracy = float(calculate_accuracy(X, T, W))  # Convert to Python float
        accuracies.append(train_accuracy)
        losses.append(float(epoch_loss))  # Convert to Python float
        
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





#%% 7. Train the Model

epochs = 100
eta = 0.001
W, losses, train_accuracies = train(X_train.T, T_train.T, W, epochs, eta)




#%% 8. Evaluate the Model

def predict(X, W):
    y, _ = forward(X, W)
    return jnp.argmax(y, axis=0)

def calculate_test_loss(X, T, W):
    """Calculate average cross-entropy loss on test set"""
    loss = loss_fn(W, X, T)
    return loss / X.shape[1]  # Average loss per sample

# Test the model
y_pred = predict(X_test.T, W)
accuracy = jnp.mean(y_pred == jnp.argmax(T_test.T, axis=0))
test_loss = calculate_test_loss(X_test.T, T_test.T, W)

print(f"\n================== Final Results ==================")
print(f"Test Accuracy: {float(accuracy * 100):.2f}%")
print(f"Test Loss (avg per sample): {float(test_loss):.4f} (0.0 is perfect, 2.3 is random guessing)")
print(f"Training Accuracy Improvement: {(train_accuracies[-1] - train_accuracies[0]):.1f}% points")
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")