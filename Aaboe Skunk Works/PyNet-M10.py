#%% 1. Import Required Libraries

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import time

def softmax(y_hat):
    y_hat = y_hat - np.max(y_hat, axis=0, keepdims=True)  # prevent overflow
    exp_scores = np.exp(y_hat)
    return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)




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

def init(dims):
    W = []
    for i in range(len(dims) - 1):
        W.append(np.random.randn(dims[i] + 1, dims[i + 1]) * np.sqrt(2 / (dims[i])))
    return W

dims = [784, 32, 32, 10]  # Input layer (784), hidden layer (128), output layer (10)
W = init(dims)




#%% 4. Define Forward Pass

def forward(X, W):
    h = []
    a = X
    for l in range(len(W) - 1):
        a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
        z = W[l].T @ a
        a = np.maximum(0, z)  # ReLU activation
        h.append(a)
    a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
    y_hat = W[-1].T @ a
    y = softmax(y_hat)  # new stable version
  # Softmax
    return y, h




#%% 5. Define Backward Pass

def backward(X, T, W, h, eta):
    m = X.shape[1]
    y, _ = forward(X, W)
    delta = y - T
    for l in range(len(W) - 1, 0, -1):
        a_prev = np.vstack([h[l-1], np.ones(h[l-1].shape[1])])  # Add bias term
        Q = a_prev @ delta.T
        W[l] -= (eta / m) * Q
        delta = W[l][:-1, :] @ delta
        delta *= h[l-1] > 0  # ReLU derivative
    a_prev = np.vstack([X, np.ones(X.shape[1])])  # Add bias term
    Q = a_prev @ delta.T
    W[0] -= eta * Q
    epsilon = 1e-12
    loss = -np.sum(np.log(np.sum(y * T, axis=0) + epsilon))
    return W, loss




#%% 6. Training Loop

def calculate_accuracy(X, T, W):
    """Calculate accuracy percentage"""
    y, _ = forward(X, W)
    predictions = np.argmax(y, axis=0)
    true_labels = np.argmax(T, axis=0)
    return np.mean(predictions == true_labels) * 100

def train(X, T, W, epochs, eta, batchsize=32):
    m = X.shape[1]
    losses = []
    accuracies = []  # Track training accuracy
    epoch_times = []  # Track computation time per epoch
    
    # Print header for nicely formatted table
    print("-" * 51)
    print(f"{'Epoch':<10} {'Accuracy':<10} {'Time':<10} {'ETA'}")
    print("-" * 51)
    
    start_total = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()  # Start timing this epoch
        
        order = np.random.permutation(m)
        epoch_loss = 0
        for i in range(0, m, batchsize):
            batch = order[i:i+batchsize]
            X_batch = X[:, batch]
            T_batch = T[:, batch]
            _, h = forward(X_batch, W)
            W, loss = backward(X_batch, T_batch, W, h, eta)
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





#%% 7. Train the Model

epochs = 100
eta = 0.001
W, losses, train_accuracies = train(X_train.T, T_train.T, W, epochs, eta)




#%% 8. Evaluate the Model

def predict(X, W):
    y, _ = forward(X, W)
    return np.argmax(y, axis=0)

def calculate_test_loss(X, T, W):
    """Calculate average cross-entropy loss on test set"""
    y, _ = forward(X, W)
    epsilon = 1e-12
    loss = -np.sum(np.log(np.sum(y * T, axis=0) + epsilon))
    return loss / X.shape[1]  # Average loss per sample

# Test the model
y_pred = predict(X_test.T, W)
accuracy = np.mean(y_pred == y_test)
test_loss = calculate_test_loss(X_test.T, T_test.T, W)

print(f"\n================== Final Results ==================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss (avg per sample): {test_loss:.4f} (0.0 is perfect, 2.3 is random guessing)")
print(f"Training Accuracy Improvement: {(train_accuracies[-1] - train_accuracies[0]):.1f}% points")
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
