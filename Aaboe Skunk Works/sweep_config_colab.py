# Google Colab-friendly sweep configuration
# Use this in Google Colab instead of sweep_config.yaml

import os
import wandb
os.environ["WANDB_START_METHOD"] = "thread"

sweep_config = {
    'method': 'grid',  # Options: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        # Architecture parameters
        'hidden_units': {
            'value': [128, 128] # Changed 'values' to 'value' to pass the list as a single parameter
        },
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid']
        },
        'weights_init': {
            'values': ['he', 'xavier', 'normal']
        },

        # Optimizer parameters
        'optimizer': {
            'values': ['adam']
        },
        'learning_rate': {
            'values': [0.001]
        },

        # Regularization parameters
        'dropout_p_value': {
            'values': [0.3]
        },
        'l2_coeff': {
            'values': [1e-8]
        },

        # Training parameters
        'batch_size': {
            'values': [512]
        },

        # Fixed parameters
        'num_epochs': {
            'value': 100  # Use 'value' (not 'values') for fixed params
        },
        'loss': {
            'value': 'cross_entropy'
        },
        'use_grad_clipping': {
            'value': False  # Python False instead of YAML false
        },
        'max_grad_norm': {
            'value': 50.0
        },
        'seed': {
            'value': 42
        }
    }
}