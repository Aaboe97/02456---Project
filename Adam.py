import numpy as np                     
import matplotlib.pyplot as plt        
from IPython.display import clear_output  


# ====================== NN components ======================

def relu(z):
    # ReLU activation: zero out negative values, keep positives
    return np.maximum(0, z)

def relu_derivative(z):
    # Derivative of ReLU: 1 where z > 0, else 0
    return (z > 0).astype(float)

def softmax(z):
    # Shift logits by max for numerical stability (avoid overflow)
    z = z - np.max(z, axis=0, keepdims=True)
    # Exponentiate shifted logits
    exp_z = np.exp(z)
    # Normalize so columns sum to 1 → probabilities per sample
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def init_weights(dims, seed=0):
    """
    dims: [D, H1, ..., C]
    Bias is folded into each W as an extra input row.
    """
    rng = np.random.default_rng(seed)  # Random generator for reproducibility
    W = []                             # List to hold weight matrices
    for l in range(len(dims) - 1):
        fan_in = dims[l] + 1           # Inputs to this layer + 1 for bias
        fan_out = dims[l+1]            # Neurons in next layer
        # He-style init: N(0, 2/fan_in) for ReLU stability
        W_l = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
        W.append(W_l)                  # Store weights for this layer
    return W                           # W[l] has shape (fan_in, fan_out)


# ====================== Adam optimizer ======================

# Hyperparamters:
# default recommendations from the Adam
# beta1 - Moment of gradient -> 0.9 (“about 90% previous direction, 10% new gradient”)
# beta2 - how fast we track gradient magnitude - controls the exponential moving average of the squared gradients - beta2 very close to 1 → long, smooth memory (good). -> 0.999 
# epsilon - tiny number for numerical stability


def init_adam(W, beta1=0.9, beta2=0.999, eps=1e-8):
    # First moment (mean of gradients) for each W[l], initialized to zeros
    m = [np.zeros_like(Wl) for Wl in W]
    # Second moment (mean of squared gradients) for each W[l]
    v = [np.zeros_like(Wl) for Wl in W]
    # Store everything in a dict so we can carry state between steps
    return {"m": m, "v": v, "t": 0, "beta1": beta1, "beta2": beta2, "eps": eps}

def adam_step(W, grads, state, lr):
    # Increment time step t (needed for bias correction)
    state["t"] += 1
    t = state["t"]
    b1, b2, eps = state["beta1"], state["beta2"], state["eps"]

    # Loop over all layers
    for l in range(len(W)):
        g = grads[l]   # Current gradient for this layer (same shape as W[l])

        # Update biased first moment estimate (exponential moving average of grads)
        state["m"][l] = b1 * state["m"][l] + (1 - b1) * g

        # Update biased second moment estimate (EMA of squared grads)
        state["v"][l] = b2 * state["v"][l] + (1 - b2) * (g ** 2)

        # Bias-corrected first moment (compensate for init at 0)
        m_hat = state["m"][l] / (1 - b1**t)

        # Bias-corrected second moment
        v_hat = state["v"][l] / (1 - b2**t)

        # Adam parameter update: scale step by 1 / (sqrt(v_hat)+eps)
        W[l] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Return updated weights and updated state
    return W, state


# ====================== Forward / Backward ======================

def forward(X, W):
    """
    X: (D, N)  features x samples (already transposed before calling)
    Returns:
      y      : (C, N) softmax outputs
      a_list : activations per layer
      z_list : pre-activations per layer
    """
    a_list = [X]   # Store activations; start with input
    z_list = []    # Store pre-activations (z = W^T * [a;1])
    a = X          # Current activation

    # ----- Hidden layers -----
    for l in range(len(W) - 1):
        # Add bias row of ones: shape (fan_in, N)
        a_bias = np.vstack([a, np.ones((1, a.shape[1]))])
        # Linear transform: z = W^T * [a;1], shape (H_l, N)
        z = W[l].T @ a_bias
        # Apply ReLU non-linearity
        a = relu(z)
        # Save for backprop
        z_list.append(z)
        a_list.append(a)

    # ----- Output layer -----
    a_bias = np.vstack([a, np.ones((1, a.shape[1]))])
    # Pre-activation for output: (C, N)
    zL = W[-1].T @ a_bias
    # Softmax to convert to probabilities
    y = softmax(zL)
    # Store final pre-activation
    z_list.append(zL)

    # Return outputs + caches for backward pass
    return y, a_list, z_list

def backward(X, T, W):
    """
    X: (D, N)  input (features x samples)
    T: (C, N)  one-hot targets
    Returns:
      grads : list of dL/dW with same shapes as W
      loss  : mean cross-entropy over the batch
    """
    N = X.shape[1]                 # Number of samples in this batch
    # Run forward to get predictions and caches
    y, a_list, z_list = forward(X, W)

    # Cross-entropy loss: average over samples
    loss = -np.sum(T * np.log(y + 1e-15)) / N

    # Allocate gradient containers matching W
    grads = [np.zeros_like(Wl) for Wl in W]

    # ----- Output layer gradients -----
    # For softmax + cross-entropy: delta = (y - T)/N
    delta = (y - T) / N            # Shape (C, N)

    # Last hidden activations
    a_last = a_list[-1]            # Shape (H_{L-1}, N)
    # Add bias row for gradient wrt W[-1]
    a_last_bias = np.vstack([a_last, np.ones((1, N))])
    # dL/dW_last = [a_last;1] @ delta^T  → shape (H_{L-1}+1, C)
    grads[-1] = a_last_bias @ delta.T

    # ----- Hidden layers (backprop) -----
    # Iterate layers in reverse (excluding output layer)
    for l in range(len(W) - 2, -1, -1):
        # Take W of next layer, drop its bias row (we don't backprop through bias)
        W_next = W[l+1][:-1, :]        # Shape (H_l, fan_out_next)
        # ReLU derivative on that layer's pre-activation
        dz = relu_derivative(z_list[l])
        # Backpropagate delta: (H_l, N)
        delta = (W_next @ delta) * dz

        # Activation from previous layer (or input X)
        a_prev = a_list[l]             # Shape (prev_dim, N)
        # Add bias row to match W[l] shape
        a_prev_bias = np.vstack([a_prev, np.ones((1, N))])
        # dL/dW_l = [a_prev;1] @ delta^T
        grads[l] = a_prev_bias @ delta.T

    # Return gradients for all layers + loss
    return grads, loss


# ====================== Helpers ======================

def predict(X_col, W):
    """
    X_col: (D, N) features x samples
    Returns predicted class index per sample.
    """
    y, _, _ = forward(X_col, W)
    # Take argmax over classes for each sample
    return np.argmax(y, axis=0)

def accuracy(X_col, T_col, W):
    """
    Compute fraction of correct predictions.
    X_col: (D, N)
    T_col: (C, N) one-hot
    """
    y_pred = predict(X_col, W)          # Predicted labels
    y_true = np.argmax(T_col, axis=0)   # True labels from one-hot
    return np.mean(y_pred == y_true)    # Mean of correct comparisons


# ====================== Training function ======================

def train_ffnn(
    X_train,
    T_train,
    dims,
    epochs=50,
    lr=1e-3,
    batch_size=32,
    use_adam=True,
    seed=0,
    plot=True,
):
    """
    X_train: (N, D)  samples x features
    T_train: (N, C)  samples x one-hot labels
    dims:    [D, H1, ..., C]
    """
    rng = np.random.default_rng(seed)   

    # Convert to internal convention: columns = samples
    X = X_train.T                       # (D, N)
    T = T_train.T                       # (C, N)
    D, N = X.shape
    C = T.shape[0]

    # Sanity checks: architecture matches data
    assert dims[0] == D
    assert dims[-1] == C

    # Initialize weights
    W = init_weights(dims, seed=seed)

    # Initialize Adam state 
    adam_state = init_adam(W) if use_adam else None

    # Store epoch losses for plotting
    losses = []

    # Enable interactive plotting (for live updates)
    if plot:
        plt.ion()

    # ----- Training loop over epochs -----
    for epoch in range(epochs):
        # Random permutation of sample indices for this epoch
        idx = rng.permutation(N)
        epoch_loss = 0.0

        # Mini-batch loop
        for start in range(0, N, batch_size):
            # Indices for this mini-batch
            batch_idx = idx[start:start+batch_size]

            # Slice mini-batch data
            Xb = X[:, batch_idx]       # (D, B)
            Tb = T[:, batch_idx]       # (C, B)

            # Compute gradients & batch loss with current weights
            grads, loss = backward(Xb, Tb, W)

            # Accumulate weighted loss (loss * batch_size)
            epoch_loss += loss * Xb.shape[1]

            # Update parameters
            if use_adam:
                # Adam update uses running moments
                W, adam_state = adam_step(W, grads, adam_state, lr)
            else:
                # Plain gradient descent: W = W - lr * grad
                for l in range(len(W)):
                    W[l] -= lr * grads[l]

        # Convert accumulated loss to mean over all N samples
        epoch_loss /= N
        losses.append(epoch_loss)

        # Live plot: redraw loss curve after each epoch
        if plot:
            clear_output(wait=True)               
            plt.figure()
            plt.plot(range(1, len(losses) + 1), losses, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training loss over epochs")
            plt.grid(True)
            plt.show()

    # Turn off interactive mode when done
    if plot:
        plt.ioff()

    # Compute final training accuracy on full training set
    final_acc = accuracy(X, T, W) * 100.0
    print(f"Final training accuracy: {final_acc:.2f}%")

    # Return trained weights + loss history
    return W, losses
