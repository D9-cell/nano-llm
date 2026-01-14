import math
import random
from typing import List, Dict, Tuple
from collections import defaultdict


class BigramLM:
    """Statistical bigram language model using Maximum Likelihood Estimation."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.bigram_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.bigram_probs: Dict[int, Dict[int, float]] = {}
    
    def train(self, tokens: List[int]) -> None:
        """Estimate bigram probabilities from token sequence.
        
        Args:
            tokens: List of integer token IDs
        """
        # Count bigrams
        for i in range(len(tokens) - 1):
            prev_token = tokens[i]
            next_token = tokens[i + 1]
            self.bigram_counts[prev_token][next_token] += 1
        
        # Convert counts to probabilities
        self.bigram_probs = {}
        for prev_token, next_counts in self.bigram_counts.items():
            total = sum(next_counts.values())
            self.bigram_probs[prev_token] = {
                next_token: count / total 
                for next_token, count in next_counts.items()
            }
    
    def get_prob(self, prev_token: int, next_token: int) -> float:
        """Get conditional probability P(next_token | prev_token).
        
        Args:
            prev_token: Previous token ID
            next_token: Next token ID
            
        Returns:
            Conditional probability
        """
        if prev_token in self.bigram_probs:
            return self.bigram_probs[prev_token].get(next_token, 0.0)
        return 0.0
    
    def compute_loss(self, tokens: List[int]) -> float:
        """Compute average cross-entropy loss over sequence.
        
        Args:
            tokens: List of integer token IDs
            
        Returns:
            Average cross-entropy loss in nats
        """
        if len(tokens) < 2:
            return 0.0
        
        total_nll = 0.0
        count = 0
        
        for i in range(len(tokens) - 1):
            prev_token = tokens[i]
            next_token = tokens[i + 1]
            prob = self.get_prob(prev_token, next_token)
            
            if prob > 0:
                total_nll += -math.log(prob)
            else:
                # Unseen bigram: assign high penalty
                total_nll += 10.0
            count += 1
        
        return total_nll / count if count > 0 else 0.0
    
    def compute_negative_log_likelihood(self, tokens: List[int]) -> float:
        """Compute total negative log-likelihood of sequence.
        
        Args:
            tokens: List of integer token IDs
            
        Returns:
            Total negative log-likelihood
        """
        if len(tokens) < 2:
            return 0.0
        
        nll = 0.0
        for i in range(len(tokens) - 1):
            prev_token = tokens[i]
            next_token = tokens[i + 1]
            prob = self.get_prob(prev_token, next_token)
            
            if prob > 0:
                nll += -math.log(prob)
            else:
                nll += 10.0
        
        return nll
    
    def sample(self, start_token: int, length: int, seed: int = None) -> List[int]:
        """Generate token sequence using learned bigram probabilities.
        
        Args:
            start_token: Initial token ID
            length: Number of tokens to generate (including start_token)
            seed: Random seed for reproducibility
            
        Returns:
            List of generated token IDs
        """
        if seed is not None:
            random.seed(seed)
        
        tokens = [start_token]
        current_token = start_token
        
        for _ in range(length - 1):
            if current_token not in self.bigram_probs:
                # No data for this token, sample uniformly
                next_token = random.randint(0, self.vocab_size - 1)
            else:
                # Sample from learned distribution
                next_tokens = list(self.bigram_probs[current_token].keys())
                next_probs = list(self.bigram_probs[current_token].values())
                next_token = random.choices(next_tokens, weights=next_probs)[0]
            
            tokens.append(next_token)
            current_token = next_token
        
        return tokens


def train_bigram_model(tokens: List[int], vocab_size: int) -> BigramLM:
    """Train a bigram language model on token sequence.
    
    Args:
        tokens: List of integer token IDs
        vocab_size: Size of vocabulary
        
    Returns:
        Trained BigramLM instance
    """
    model = BigramLM(vocab_size)
    model.train(tokens)
    return model
