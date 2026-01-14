import torch
import torch.nn as nn
from .attention import SingleHeadSelfAttention
from .mlp import FeedForward


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization, attention, and feed-forward."""
    
    def __init__(self, embedding_dim: int, ffn_hidden_dim: int):
        """Initialize Transformer block.
        
        Args:
            embedding_dim: Dimension of token embeddings
            ffn_hidden_dim: Dimension of feed-forward hidden layer
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        
        # Pre-normalization layers
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        
        # Attention and feed-forward sublayers
        self.attention = SingleHeadSelfAttention(embedding_dim)
        self.ffn = FeedForward(embedding_dim, ffn_hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Transformer block with residual connections.
        
        Args:
            x: Input embeddings of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Output embeddings of shape [batch_size, seq_len, embedding_dim]
        """
        # Attention sublayer with residual connection
        x = x + self.attention(self.ln1(x))
        
        # Feed-forward sublayer with residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x