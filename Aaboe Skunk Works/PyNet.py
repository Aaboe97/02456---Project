#%% PyNet Shared Functions Module
import time
import numpy as np
import wandb

class PyNetBase:
    """Base class containing all shared neural network functionality"""
    
    def __init__(self, num_features, hidden_units, num_output, weights_init='he', activation='relu', loss='cross_entropy', optimizer='sgd', l2_coeff=0.0, dropout_p=None):
        """
        Initialize neural network with configurable architecture.
        
        Args:
            num_features: Number of input features
            hidden_units: List of hidden layer sizes [layer1, layer2, ...]
            num_output: Number of output classes
            weights_init: Weight initialization method ('he', 'xavier', 'normal')
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            loss: Loss function ('cross_entropy', 'mse', 'mae')
            optimizer: Optimizer type ('sgd', 'adam', 'rmsprop')
            l2_coeff: L2 regularization coefficient (weight_decay)
            dropout_p: List of dropout probabilities for each hidden layer (None = no dropout)
        """
        
        # Build layer sizes: input → hidden layers → output
        layer_sizes = [num_features] + hidden_units + [num_output]
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights_init = weights_init
        self.loss = loss
        self.optimizer = optimizer
        self.l2_coeff = l2_coeff
        self.dropout_p = dropout_p
        
        # Validate dropout_p if provided
        num_hidden = len(hidden_units)
        if dropout_p is not None:
            if len(dropout_p) != num_hidden:
                raise ValueError(f"dropout_p must have {num_hidden} values (one per hidden layer)")
            self.dropout_p = dropout_p
        else:
            self.dropout_p = [0.0] * num_hidden  # No dropout by default
        
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
        
        # Initialize optimizer state
        if optimizer == 'adam':
            self.m = [np.zeros_like(w) for w in self.W]  # First moment estimates
            self.v = [np.zeros_like(w) for w in self.W]  # Second moment estimates
            self.t = 0  # Time step counter
        elif optimizer == 'rmsprop':
            self.v = [np.zeros_like(w) for w in self.W]  # Moving average of squared gradients


    def forward(self, X, W, dropout_on=False):
        """
        Forward pass through the network with optional dropout

        Args:
            X: Input data
            W: Weights
            dropout_on: Whether to apply dropout (True during training, False during inference)
        Returns:
            y: Output predictions
            h: List of hidden layer activations
            masks: List of dropout masks (one per hidden layer)
        """
        h = []
        masks = []
        a = X
        num_hidden = len(W) - 1
        
        # Loop through hidden layers
        for l in range(num_hidden):
            a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
            z = W[l].T @ a
            a = self._activation_function(z)  # Use configurable activation
            
            # Apply dropout if enabled
            if dropout_on and self.dropout_p[l] > 0.0:
                p = self.dropout_p[l]
                # Inverted dropout: scale active neurons to maintain expected activation
                mask = (np.random.rand(*a.shape) > p).astype(float) / (1.0 - p)
                a *= mask
            else:
                mask = np.ones_like(a)  # No dropout: all neurons active
            
            h.append(a)
            masks.append(mask)
        
        # Output layer (no dropout)
        a = np.vstack([a, np.ones(a.shape[1])])  # Add bias term
        y_hat = W[-1].T @ a
        y = self._softmax(y_hat)  # Output layer always uses softmax for classification
        return y, h, masks
    

    def backward(self, X, T, W, h, masks, eta, y_pred=None, use_clipping=True, max_grad_norm=25.0):
        """
        Backward pass with configurable optimizers, L2 regularization, gradient clipping, and dropout.
        
        Args:
            X: Input data
            T: Target labels
            W: Weights
            h: Hidden activations from forward pass
            masks: Dropout masks from forward pass
            eta: Learning rate
            y_pred: Pre-computed predictions (optional, for efficiency)
            use_clipping: Whether to use gradient clipping (default True)
            max_grad_norm: Maximum gradient norm for clipping (default 25.0)
        """
        m = X.shape[1]
        
        if y_pred is None:  # Use pre-computed predictions if available, otherwise compute them
            y, _, _ = self.forward(X, W, dropout_on=False)
        else:
            y = y_pred
        
        # Increment Adam time step once per backward pass
        if self.optimizer == 'adam':
            self.t += 1
            
        delta = self._loss_derivative(y, T)  # Use configurable loss derivative
        
        # Backpropagate through hidden layers (in reverse)
        for l in range(len(W) - 1, 0, -1):
            a_prev = np.vstack([h[l-1], np.ones(h[l-1].shape[1])])  # Add bias term
            Q = a_prev @ delta.T
            
            # Add L2 regularization to gradient (don't regularize biases - last row)
            if self.l2_coeff > 0:
                Q[:-1, :] += self.l2_coeff * W[l][:-1, :]  # Only regularize weights, not biases
            
            # Optional gradient clipping
            if use_clipping:
                grad_norm = np.linalg.norm(Q)
                if grad_norm > max_grad_norm:
                    Q *= max_grad_norm / grad_norm
            
            # Apply optimizer update
            W = self._apply_optimizer_update(W, l, Q, eta, m)
            
            # Backpropagate delta
            delta = W[l][:-1, :] @ delta
            delta *= self._activation_derivative(h[l-1])  # Use configurable activation derivative
            delta *= masks[l-1]  # Apply dropout mask (only gradients through active neurons)
            
        # First layer gradient
        a_prev = np.vstack([X, np.ones(X.shape[1])])  # Add bias term
        Q = a_prev @ delta.T
        
        # Add L2 regularization to first layer gradient
        if self.l2_coeff > 0:
            Q[:-1, :] += self.l2_coeff * W[0][:-1, :]  # Only regularize weights, not biases
        
        # Optional gradient clipping for first layer
        if use_clipping:
            grad_norm = np.linalg.norm(Q)
            if grad_norm > max_grad_norm:
                Q = Q * (max_grad_norm / grad_norm)
        
        # Apply optimizer update to first layer
        W = self._apply_optimizer_update(W, 0, Q, eta, m)
        loss = self._loss_function(y, T)
        return W, loss
    
    
    def _apply_optimizer_update(self, W, layer_idx, gradient, eta, batch_size):
        """Helper method to apply optimizer-specific weight updates"""
        
        if self.optimizer == 'sgd': 
            # Standard SGD (Stochastic Gradient Descent) update
            W[layer_idx] -= (eta / batch_size) * gradient
            
        elif self.optimizer == 'adam':
            # Adam (Adaptive Moment Estimation) optimizer update
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8
            
            # Update biased first moment estimate
            self.m[layer_idx] = beta1 * self.m[layer_idx] + (1 - beta1) * gradient
            # Update biased second raw moment estimate  
            self.v[layer_idx] = beta2 * self.v[layer_idx] + (1 - beta2) * gradient**2
            # Compute bias-corrected first moment estimate
            m_hat = self.m[layer_idx] / (1 - beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[layer_idx] / (1 - beta2**self.t)
            # Update weights
            denominator = np.sqrt(v_hat) + epsilon
            update = (eta / batch_size) * m_hat / denominator
            # Clip extreme updates to prevent instability
            update = np.clip(update, -1.0, 1.0)
            W[layer_idx] -= update
            
        elif self.optimizer == 'rmsprop':
            # RMSprop (Root Mean Square Propagation) optimizer update
            alpha, epsilon = 0.99, 1e-8
            # Update moving average of squared gradients
            self.v[layer_idx] = alpha * self.v[layer_idx] + (1 - alpha) * gradient**2
            # Update weights
            W[layer_idx] -= (eta / batch_size) * gradient / (np.sqrt(self.v[layer_idx]) + epsilon)
        
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        return W
    

    def _softmax(self, y_hat):
        """Compute softmax probabilities"""
        y_hat = y_hat - np.max(y_hat, axis=0, keepdims=True)  # prevent overflow
        exp_scores = np.exp(y_hat)
        return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    

    def _activation_function(self, z):
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        

    def _activation_derivative(self, a):
        """Calculate derivative of activation function"""
        if self.activation == 'relu':
            return a > 0
        elif self.activation == 'tanh':
            return 1 - a**2
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        

    def _loss_function(self, y_pred, y_true):
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
        

    def _loss_derivative(self, y_pred, y_true):
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
    
    


# Shared utility functions
def calculate_accuracy(net, X, T, W):
    """Calculate accuracy percentage (always with dropout OFF)"""
    y, _, _ = net.forward(X, W, dropout_on=False)
    predictions = np.argmax(y, axis=0)
    true_labels = np.argmax(T, axis=0)
    return np.mean(predictions == true_labels) * 100
    

def train(net, X, T, W, epochs, eta, batchsize=32, use_clipping=True, max_grad_norm=25.0, use_wandb=False, wandb_project=None, wandb_config=None, wandb_mode="online"):
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
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_config: Dictionary of hyperparameters to log to W&B
        wandb_mode: W&B mode - "online", "offline", or "disabled"
    """
    losses = []
    accuracies = []  # Track training accuracy
    epoch_times = []  # Track computation time per epoch
    
    # Initialize W&B if enabled
    if use_wandb and wandb_project:
        wandb.init(project=wandb_project, config=wandb_config, mode=wandb_mode)
    
    # Print header for nicely formatted table
    print("-" * 70)
    print(f"{'Epoch':<10} {'Accuracy':<10} {'Gain':<10} {'Time':<10} {'ETA'}")
    print("-" * 70)
    
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
            # Forward pass with dropout enabled during training
            y_batch, h, masks = net.forward(X_batch, W, dropout_on=True)
            # Backward pass with dropout masks
            W, loss = net.backward(X_batch, T_batch, W, h, masks, eta, y_batch, use_clipping, max_grad_norm)
            epoch_loss += loss

        # Calculate training accuracy for this epoch
        train_accuracy = calculate_accuracy(net, X, T, W)
        accuracies.append(train_accuracy)
        losses.append(epoch_loss)
        
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
            avg_time_per_epoch = np.mean(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs
            if eta_seconds > 60:
                eta_str = f"{np.floor(eta_seconds/60):.0f}min {eta_seconds%60:.0f}sec"
            else:
                eta_str = f"{eta_seconds:.0f}sec"
        else:
            eta_str = "calculating..."
        
        # Format epoch info for output
        epoch_str = f"{epoch+1}/{epochs}"
        accuracy_str = f"{train_accuracy:.2f}%"
        time_str = f"{epoch_time:.2f}sec"
        
        # Show progress 
        print(f"{epoch_str:<10} {accuracy_str:<10} {gain_str:<10} {time_str:<10} {eta_str}")
        
        # Log to W&B if enabled
        if use_wandb and wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": float(epoch_loss),
                "train_accuracy": float(train_accuracy),
                "epoch_time": float(epoch_time)
            })
    
    total_time = time.time() - start_total
    avg_epoch_time = np.mean(epoch_times)
    
    print("-" * 70)
    print(f"Total training time: {total_time:.1f}sec")
    print(f"Average per epoch: {avg_epoch_time:.2f}sec")
    print("-" * 70)
    
    # Don't finish W&B here - let evaluate_model do it after logging test metrics
    
    return W, losses, accuracies

def evaluate_model(net, X_test, T_test, y_test, W, train_accuracies, use_wandb=False):
    """
    Evaluate model performance and print results
    
    Args:
        net: Neural network instance
        X_test, T_test: Test data and labels
        y_test: Test labels (not one-hot encoded)
        W: Trained weights
        train_accuracies: List of training accuracies from training
        use_wandb: Whether to log test metrics to W&B
    """
    # Make predictions and calculate accuracy (dropout OFF for evaluation)
    y_test_pred, _, _ = net.forward(X_test.T, W, dropout_on=False)
    y_pred = np.argmax(y_test_pred, axis=0)
    test_accuracy = np.mean(y_pred == y_test)

    # Calculate test loss using the configurable loss function
    test_loss = net._loss_function(y_test_pred, T_test.T) / X_test.shape[0]  # Average per sample

    print(f"\n================== Final Results ==================")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss (avg per sample): {test_loss:.4f}")
    print(f"Training Accuracy Improvement: {(train_accuracies[-1] - train_accuracies[0]):.1f}% points")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    
    # Log test metrics to W&B if enabled and run is still active
    if use_wandb and wandb.run is not None:
        wandb.log({
            "test_accuracy": float(test_accuracy * 100),
            "test_loss": float(test_loss)
        })
        wandb.finish(quiet=False)  # Finish the W&B run after logging test metrics (quiet=False shows summary)
    
    return y_pred, test_accuracy, test_loss



