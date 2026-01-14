import torch
import torch.nn as nn
from .mlp import MLPLanguageModel
from .attention import SingleHeadSelfAttention
from .block import TransformerBlock


class NeuralBigramModel(nn.Module):
    """Neural bigram language model using embeddings and linear projection."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize neural bigram model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for next token prediction.
        
        Args:
            token_ids: Input token IDs of shape [batch_size] or [batch_size, seq_len]
            
        Returns:
            Logits of shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        """
        embeddings = self.token_embedding(token_ids)
        logits = self.lm_head(embeddings)
        return logits


class FixedContextMLPModel(nn.Module):
    """Fixed-context MLP language model wrapper."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int, hidden_dim: int):
        """Initialize fixed-context MLP model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            context_length: Number of previous tokens to condition on
            hidden_dim: Dimension of hidden layer
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        
        self.mlp_model = MLPLanguageModel(vocab_size, embedding_dim, context_length, hidden_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for next token prediction.
        
        Args:
            token_ids: Input token IDs of shape [batch_size, context_length]
            
        Returns:
            Logits of shape [batch_size, vocab_size]
        """
        return self.mlp_model(token_ids)


class AttentionLanguageModel(nn.Module):
    """Self-attention based language model."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize attention language model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SingleHeadSelfAttention(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for next token prediction.
        
        Args:
            token_ids: Input token IDs of shape [batch_size, seq_len]
            
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        embeddings = self.token_embedding(token_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Apply self-attention
        attn_output = self.attention(embeddings)  # [batch_size, seq_len, embedding_dim]
        
        # Project to vocabulary
        logits = self.lm_head(attn_output)  # [batch_size, seq_len, vocab_size]
        
        return logits


class TransformerLanguageModel(nn.Module):
    """Transformer-based language model with stacked blocks."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int, ffn_hidden_dim: int):
        """Initialize Transformer language model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            num_layers: Number of Transformer blocks to stack
            ffn_hidden_dim: Dimension of feed-forward hidden layer
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.ffn_hidden_dim = ffn_hidden_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Stack Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, ffn_hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for next token prediction.
        
        Args:
            token_ids: Input token IDs of shape [batch_size, seq_len]
            
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.token_embedding(token_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
