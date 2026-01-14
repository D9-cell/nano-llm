import math
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple


def load_text(file_path: str) -> str:
    """Load raw text from file."""
    return Path(file_path).read_text(encoding='utf-8')


def compute_char_frequencies(text: str) -> Dict[str, int]:
    """Compute character frequency counts."""
    return dict(Counter(text))


def compute_probabilities(frequencies: Dict[str, int]) -> Dict[str, float]:
    """Normalize frequencies into probabilities."""
    total = sum(frequencies.values())
    return {char: count / total for char, count in frequencies.items()}


def compute_entropy(probabilities: Dict[str, float]) -> float:
    """Compute Shannon entropy of character distribution.
    
    H(X) = -Σ p(x) * log2(p(x))
    """
    return -sum(p * math.log2(p) for p in probabilities.values() if p > 0)


def analyze_text_distribution(file_path: str) -> Dict:
    """Analyze character-level probability distribution of text.
    
    Args:
        file_path: Path to raw text file
        
    Returns:
        Dictionary containing:
            - frequencies: Character counts
            - probabilities: Normalized probabilities
            - entropy: Shannon entropy in bits
            - vocab_size: Number of unique characters
            - total_chars: Total character count
    """
    text = load_text(file_path)
    frequencies = compute_char_frequencies(text)
    probabilities = compute_probabilities(frequencies)
    entropy = compute_entropy(probabilities)
    
    return {
        'frequencies': frequencies,
        'probabilities': probabilities,
        'entropy': entropy,
        'vocab_size': len(frequencies),
        'total_chars': len(text)
    }
