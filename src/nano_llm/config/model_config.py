from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for language model and training."""
    
    # Model architecture
    vocab_size: int
    embedding_dim: int
    num_layers: int
    ffn_hidden_dim: int
    context_length: int
    
    # Training hyperparameters
    learning_rate: float
    batch_size: int
    num_steps: int
    
    # Reproducibility
    seed: int
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.ffn_hidden_dim > 0, "ffn_hidden_dim must be positive"
        assert self.context_length > 0, "context_length must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_steps > 0, "num_steps must be positive"