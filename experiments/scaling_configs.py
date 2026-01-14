"""Scaling experiment configurations for nano-llm.

This module defines different model configurations for studying scaling behavior.
"""

# Nano model: Minimal baseline
NANO_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 32,
    'num_layers': 2,
    'ffn_hidden_dim': 64,
    'context_length': 32,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/nano',
    'checkpoint_interval': 500
}

# Small model: 2x capacity
SMALL_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 64,
    'num_layers': 4,
    'ffn_hidden_dim': 128,
    'context_length': 32,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/small',
    'checkpoint_interval': 500
}

# Medium model: 4x capacity
MEDIUM_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 128,
    'num_layers': 6,
    'ffn_hidden_dim': 256,
    'context_length': 32,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/medium',
    'checkpoint_interval': 500
}

# Context scaling: Short context
SHORT_CONTEXT_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 64,
    'num_layers': 4,
    'ffn_hidden_dim': 128,
    'context_length': 16,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/context_short',
    'checkpoint_interval': 500
}

# Context scaling: Long context
LONG_CONTEXT_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 64,
    'num_layers': 4,
    'ffn_hidden_dim': 128,
    'context_length': 64,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/context_long',
    'checkpoint_interval': 500
}

# Depth scaling: Shallow
SHALLOW_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 64,
    'num_layers': 2,
    'ffn_hidden_dim': 128,
    'context_length': 32,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/depth_shallow',
    'checkpoint_interval': 500
}

# Depth scaling: Deep
DEEP_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 64,
    'num_layers': 8,
    'ffn_hidden_dim': 128,
    'context_length': 32,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/depth_deep',
    'checkpoint_interval': 500
}

# Width scaling: Narrow
NARROW_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 32,
    'num_layers': 4,
    'ffn_hidden_dim': 64,
    'context_length': 32,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/width_narrow',
    'checkpoint_interval': 500
}

# Width scaling: Wide
WIDE_CONFIG = {
    'vocab_size': 65,
    'embedding_dim': 128,
    'num_layers': 4,
    'ffn_hidden_dim': 256,
    'context_length': 32,
    'learning_rate': 1e-3,
    'batch_size': 1,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/width_wide',
    'checkpoint_interval': 500
}

# Experiment suites
CAPACITY_SCALING_SUITE = [
    ('nano', NANO_CONFIG),
    ('small', SMALL_CONFIG),
    ('medium', MEDIUM_CONFIG)
]

CONTEXT_SCALING_SUITE = [
    ('context_short', SHORT_CONTEXT_CONFIG),
    ('context_medium', SMALL_CONFIG),
    ('context_long', LONG_CONTEXT_CONFIG)
]

DEPTH_SCALING_SUITE = [
    ('depth_shallow', SHALLOW_CONFIG),
    ('depth_medium', SMALL_CONFIG),
    ('depth_deep', DEEP_CONFIG)
]

WIDTH_SCALING_SUITE = [
    ('width_narrow', NARROW_CONFIG),
    ('width_medium', SMALL_CONFIG),
    ('width_wide', WIDE_CONFIG)
]
