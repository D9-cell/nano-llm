from pathlib import Path
from typing import List, Dict


class CharTokenizer:
    """Character-level tokenizer with deterministic vocabulary construction."""
    
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._vocab_size: int = 0
    
    def build_vocab(self, file_path: str) -> None:
        """Build vocabulary from raw text file.
        
        Args:
            file_path: Path to raw text file
        """
        text = Path(file_path).read_text(encoding='utf-8')
        unique_chars = sorted(set(text))
        
        self.stoi = {char: idx for idx, char in enumerate(unique_chars)}
        self.itos = {idx: char for char, idx in self.stoi.items()}
        self._vocab_size = len(unique_chars)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of integer token IDs
        """
        return [self.stoi[char] for char in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text.
        
        Args:
            tokens: List of integer token IDs
            
        Returns:
            Decoded text string
        """
        return ''.join(self.itos[token] for token in tokens)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size
