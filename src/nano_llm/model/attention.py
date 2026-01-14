import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadSelfAttention(nn.Module):
    """Single-head causal self-attention mechanism."""
    
    def __init__(self, embedding_dim: int):
        """Initialize single-head self-attention.
        
        Args:
            embedding_dim: Dimension of token embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.
        
        Args:
            x: Input embeddings of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Attention output of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        # Compute Q, K, V projections
        Q = self.query(x)  # [batch_size, seq_len, embedding_dim]
        K = self.key(x)    # [batch_size, seq_len, embedding_dim]
        V = self.value(x)  # [batch_size, seq_len, embedding_dim]
        
        # Compute attention scores (scaled dot-product)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        scores = scores / (embedding_dim ** 0.5)
        
        # Apply causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [batch_size, seq_len, embedding_dim]
        
        return output
