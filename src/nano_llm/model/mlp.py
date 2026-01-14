import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network used inside Transformer blocks."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """Initialize feed-forward network.
        
        Args:
            embedding_dim: Dimension of input and output
            hidden_dim: Dimension of hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embedding_dim]
        """
        return self.fc2(self.relu(self.fc1(x)))


class MLPLanguageModel(nn.Module):
    """MLP-based language model with fixed-length context window."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int, hidden_dim: int):
        """Initialize MLP language model.
        
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
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # MLP takes concatenated embeddings as input
        input_dim = context_length * embedding_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for next token prediction.
        
        Args:
            token_ids: Input token IDs of shape [batch_size, context_length]
            
        Returns:
            Logits of shape [batch_size, vocab_size]
        """
        batch_size = token_ids.shape[0]
        
        # Embed tokens: [batch_size, context_length, embedding_dim]
        embeddings = self.token_embedding(token_ids)
        
        # Flatten embeddings: [batch_size, context_length * embedding_dim]
        flat_embeddings = embeddings.view(batch_size, -1)
        
        # MLP forward pass
        hidden = self.relu(self.fc1(flat_embeddings))
        logits = self.fc2(hidden)
        
        return logits