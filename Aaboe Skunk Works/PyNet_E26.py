#%%########### 1. Import Required Libraries and Configuration ############## 

import warnings
import numpy as np
import tensorflow_datasets as tfds
from keras.utils import to_categorical
from PyNet import PyNetBase, train, evaluate_model
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress runtime warnings for mental stability

# Dataset Configuration
num_features = 28 * 28  # EMNIST: 28x28 pixels
num_classes = 26        # EMNIST: letters A-Z

# Architecture Configuration
hidden_units = [32, 32]    # Units per hidden layer [layer1, layer2, ...]
activation = 'relu'        # Activation function: 'relu', 'tanh', 'sigmoid'
weights_init = 'he'        # Weight initialization: 'he', 'xavier', 'normal'

# Training Configuration  
num_epochs = 100           # Number of training epochs
learning_rate = 0.001      # Learning rate for gradient descent
batch_size = 32            # Mini-batch size
loss = 'cross_entropy'     # Loss function: 'cross_entropy', 'mse', 'mae'




#%%######################## 2. Load EMNIST Data ############################

# Load EMNIST Letters dataset using TensorFlow Datasets
print("Loading EMNIST Letters dataset...")
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

print(f"Successfully loaded!")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print(f"Classes: A-Z (26 total)")
print(f"Image shape: 28x28 → {X_train.shape[1]} features")




#%%################ 3. Initialize EMNIST Neural Network #####################

# Create EMNIST-specific network class
class PyNet_E26(PyNetBase):
    """EMNIST Letters-specific neural network using shared base functionality"""
    pass

# Initialize network
net = PyNet_E26(num_features, hidden_units, num_classes, weights_init, activation, loss)




#%%########################### 4. Training Loop ############################

# Train the model (using gradient clipping for EMNIST stability)
net.W, losses, train_accuracies = train(
    net, X_train.T, T_train.T, net.W, 
    num_epochs, learning_rate, batch_size,
    use_clipping=True, max_grad_norm=50.0
)




#%%########################## 5. Evaluate Model ############################

# Evaluate and display results
y_pred, test_accuracy, test_loss = evaluate_model(
    net, X_test, T_test, y_test, net.W, train_accuracies
)

# Convert some predictions to letters for demonstration  
def number_to_letter(num):
    return chr(ord('A') + num)

print(f"\n📝 Sample Letter Predictions:")
sample_indices = np.random.choice(len(y_test), 5, replace=False)
for i in sample_indices:
    true_letter = number_to_letter(y_test[i])  # y_test is already 0-25 range
    pred_letter = number_to_letter(y_pred[i])
    print(f"True: {true_letter}, Predicted: {pred_letter}")
