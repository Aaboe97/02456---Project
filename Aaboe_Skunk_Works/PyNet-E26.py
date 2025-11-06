#%% 1. Import Required Libraries
import time
import warnings
import numpy as np
import tensorflow_datasets as tfds
from keras.utils import to_categorical
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress runtime warnings for mental stability, I'm okay...


def softmax(y_hat):
    """Numerically stable softmax"""
    
    y_hat = np.clip(y_hat, -500, 500) # Clip extreme values to prevent overflow
    y_hat = y_hat - np.max(y_hat, axis=0, keepdims=True)  # prevent overflow
    exp_scores = np.exp(y_hat)
    probs = exp_scores / (np.sum(exp_scores, axis=0, keepdims=True) + 1e-15) # Add small epsilon to prevent division by zero
    return probs




#%% 2. Load EMNIST Letters Data

# Load EMNIST Letters dataset using TensorFlow Datasets (more reliable)
print("Loading EMNIST Letters dataset...")
# Load the dataset
ds_train, ds_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

# Convert to numpy arrays
def preprocess_data(ds):
    images, labels = [], []
    for image, label in ds:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

print("Converting to numpy arrays...")
X_train, y_train = preprocess_data(ds_train)
X_test, y_test = preprocess_data(ds_test)

# Reshape and normalize inputs (same as MNIST)
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# EMNIST letters uses labels 1-26, need to convert to 0-25 for one-hot encoding
print(f"Original label range: {y_train.min()}-{y_train.max()}")
y_train = y_train - 1  # Convert 1-26 to 0-25
y_test = y_test - 1    # Convert 1-26 to 0-25
print(f"Adjusted label range: {y_train.min()}-{y_train.max()}")

# One-hot encode labels (now 0-25 for A-Z)
T_train = to_categorical(y_train, num_classes=26)
T_test = to_categorical(y_test, num_classes=26)

print(f"✅ Successfully loaded!")
print(f"📊 Training samples: {X_train.shape[0]:,}")
print(f"📊 Test samples: {X_test.shape[0]:,}")
print(f"🏷️  Classes: A-Z (26 total)")
print(f"🖼️  Image shape: 28x28 → {X_train.shape[1]} features")


#%% 3. Initialize Network Parameters

def init(dims):
    """Initialize weights with Xavier/He initialization for better stability"""
    W = []
    for i in range(len(dims) - 1):
        # Xavier initialization for better convergence
        fan_in = dims[i]
        fan_out = dims[i + 1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W.append(np.random.uniform(-limit, limit, (dims[i] + 1, dims[i + 1])))
    return W

dims = [784, 32, 32, 26]  # Input layer (784), hidden layers (32, 32), output layer (26 for A-Z)
W = init(dims)




#%% 4. Define Forward Pass

def forward(X, W):
    """Forward pass with numerical stability checks"""
    h = []
    a = X
    for l in range(len(W) - 1):
        a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
        z = W[l].T @ a
        # Clip to prevent extreme values
        z = np.clip(z, -100, 100)
        a = np.maximum(0, z)  # ReLU activation
        h.append(a)
    a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
    y_hat = W[-1].T @ a
    y_hat = np.clip(y_hat, -100, 100)  # Prevent extreme values
    y = softmax(y_hat)
    return y, h




#%% 5. Define Backward Pass

# def backward(X, T, W, h, eta):
#     m = X.shape[1]
#     y, _ = forward(X, W)
#     delta = y - T
#     for l in range(len(W) - 1, 0, -1):
#         a_prev = np.vstack([h[l-1], np.ones(h[l-1].shape[1])])  # Add bias term
#         Q = a_prev @ delta.T
#         Q = np.clip(Q, -5, 5)  # Gradient clipping
#         W[l] -= (eta / m) * Q
        
#         # Essential: Check for NaN/Inf and recover
#         if np.any(np.isnan(W[l])) or np.any(np.isinf(W[l])):
#             W[l] = np.random.uniform(-0.1, 0.1, W[l].shape)
        
#         delta = W[l][:-1, :] @ delta
#         delta *= h[l-1] > 0  # ReLU derivative
        
#     a_prev = np.vstack([X, np.ones(X.shape[1])])  # Add bias term
#     Q = a_prev @ delta.T
#     Q = np.clip(Q, -5, 5)  # Gradient clipping
#     W[0] -= (eta / m) * Q
    
#     # Essential: Check first layer too
#     if np.any(np.isnan(W[0])) or np.any(np.isinf(W[0])):
#         W[0] = np.random.uniform(-0.1, 0.1, W[0].shape)
    
#     epsilon = 1e-15
#     loss = -np.sum(np.log(np.sum(y * T, axis=0) + epsilon))
    
#     return W, loss



def backward(X, T, W, h, eta):
    """Backward pass with gradient clipping and stability checks"""
    m = X.shape[1]
    y, _ = forward(X, W)
    delta = y - T
    
    # Gradient clipping threshold
    max_grad_norm = 5.0
    
    for l in range(len(W) - 1, 0, -1):
        a_prev = np.vstack([h[l-1], np.ones(h[l-1].shape[1])])  # Add bias term
        Q = a_prev @ delta.T
        
        # Gradient clipping
        grad_norm = np.linalg.norm(Q)
        if grad_norm > max_grad_norm:
            Q = Q * (max_grad_norm / grad_norm)
        
        W[l] -= (eta / m) * Q
        
        # Check for NaN/Inf in weights
        if np.any(np.isnan(W[l])) or np.any(np.isinf(W[l])):
            print(f"Warning: NaN/Inf detected in W[{l}], reinitializing...")
            W[l] = np.random.uniform(-0.1, 0.1, W[l].shape)
        
        delta = W[l][:-1, :] @ delta
        delta *= h[l-1] > 0  # ReLU derivative
        
    a_prev = np.vstack([X, np.ones(X.shape[1])])  # Add bias term
    Q = a_prev @ delta.T
    
    # Gradient clipping for first layer
    grad_norm = np.linalg.norm(Q)
    if grad_norm > max_grad_norm:
        Q = Q * (max_grad_norm / grad_norm)
    
    W[0] -= (eta / m) * Q
    
    # Check for NaN/Inf in first layer weights
    if np.any(np.isnan(W[0])) or np.any(np.isinf(W[0])):
        print(f"Warning: NaN/Inf detected in W[0], reinitializing...")
        W[0] = np.random.uniform(-0.1, 0.1, W[0].shape)
    
    # Stable loss calculation
    epsilon = 1e-15
    y_clipped = np.clip(y, epsilon, 1 - epsilon)
    loss = -np.sum(np.log(np.sum(y_clipped * T, axis=0) + epsilon))
    
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
eta = 0.001  # Back to original learning rate for better accuracy
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
print(f"Test Loss (avg per sample): {test_loss:.4f} (0.0 is perfect, 3.3 is random guessing for 26 classes)")
print(f"Training Accuracy Improvement: {(train_accuracies[-1] - train_accuracies[0]):.1f}% points")
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")

# Convert some predictions to letters for demonstration
def number_to_letter(num):
    return chr(ord('A') + num)

print(f"\nSample predictions:")
sample_indices = np.random.choice(len(y_test), 5, replace=False)
for i in sample_indices:
    true_letter = number_to_letter(y_test[i])  # y_test is already 0-25 range
    pred_letter = number_to_letter(y_pred[i])
    print(f"True: {true_letter}, Predicted: {pred_letter}")
