import torch
import torch.nn.functional as F
from typing import List, Optional


def generate(
    model: torch.nn.Module,
    start_tokens: List[int],
    max_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None
) -> List[int]:
    """Generate tokens autoregressively using the model.
    
    Args:
        model: Trained language model
        start_tokens: Initial token sequence to condition on
        max_length: Maximum number of tokens to generate (including start_tokens)
        temperature: Temperature for sampling (higher = more random, lower = more deterministic)
        top_k: If specified, only sample from top k most likely tokens
        seed: Random seed for reproducibility
        
    Returns:
        List of generated token IDs
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    
    # Initialize with start tokens
    generated = list(start_tokens)
    
    with torch.no_grad():
        for _ in range(max_length - len(start_tokens)):
            # Prepare input (current sequence)
            input_tokens = torch.tensor([generated], dtype=torch.long)
            
            # Forward pass
            logits = model(input_tokens)  # [1, seq_len, vocab_size]
            
            # Get logits for the last position
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                # Get top-k logits and indices
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                
                # Create mask for top-k tokens
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask[top_k_indices] = top_k_logits
                next_token_logits = mask
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Append to generated sequence
            generated.append(next_token)
    
    return generated


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None
) -> str:
    """Generate text from a prompt string.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Starting text prompt
        max_length: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: If specified, only sample from top k most likely tokens
        seed: Random seed for reproducibility
        
    Returns:
        Generated text string
    """
    # Encode prompt
    start_tokens = tokenizer.encode(prompt)
    
    # Generate tokens
    generated_tokens = generate(
        model=model,
        start_tokens=start_tokens,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        seed=seed
    )
    
    # Decode to text
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text