#%%########### 1. Import Required Libraries and Configuration ##############    

import jax.numpy as jnp
from keras.datasets import mnist
from keras.utils import to_categorical
from JAXNet import JAXNetBase, train, evaluate_model

# Dataset Configuration
num_features = 28 * 28  # MNIST: 28x28 pixels
num_classes = 10        # MNIST: digits 0-9

# Architecture Configuration
hidden_units = [32, 32]    # Units per hidden layer [layer1, layer2, ...]
activation = 'relu'        # Activation function: 'relu', 'tanh', 'sigmoid'
weights_init = 'he'        # Weight initialization: 'he', 'xavier', 'normal'

# Training Configuration  
num_epochs = 100           # Number of training epochs
learning_rate = 0.001      # Learning rate for gradient descent
batch_size = 32            # Mini-batch size
loss = 'cross_entropy'     # Loss function: 'cross_entropy', 'mse', 'mae'





#%%######################### 2. Load MNIST Data ############################

# Load MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize inputs 
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Convert to JAX arrays
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)

# One-hot encode labels
T_train = to_categorical(y_train, num_classes=10)
T_test = to_categorical(y_test, num_classes=10)

# Convert to JAX arrays
T_train = jnp.array(T_train)
T_test = jnp.array(T_test)

print(f"Successfully loaded!")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print(f"Classes: 0-9 (10 total)")
print(f"Image shape: 28x28 → {X_train.shape[1]} features")




#%%################ 3. Initialize MNIST Neural Network #####################

# Create MNIST-specific network class
class JAXNet_M10(JAXNetBase):
    """MNIST-specific neural network using shared base functionality"""
    pass

# Initialize network
net = JAXNet_M10(num_features, hidden_units, num_classes, weights_init, activation, loss)




#%%########################### 4. Training Loop ############################

# Train the model (using minimal clipping for MNIST stability)
net.W, losses, train_accuracies = train(
    net, X_train.T, T_train.T, net.W,
    num_epochs, learning_rate, batch_size,
    use_clipping=False, max_grad_norm=1.0
)

#%%########################## 5. Evaluate Model ############################

# Evaluate and display results
y_pred, test_accuracy, test_loss = evaluate_model(
    net, X_test, T_test, y_test, net.W, train_accuracies
)
