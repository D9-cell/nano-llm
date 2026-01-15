"""CLI entry point for text generation."""
import sys
from pathlib import Path
from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.train.train import load_checkpoint
from src.nano_llm.inference.generate import generate_text


def main():
    """Generate text using a trained model."""
    # Default configuration
    data_path = 'data/raw/chat.txt'
    checkpoint_path = 'checkpoints/nano/checkpoint_final.pt'
    prompt = "The "
    max_length = 200
    temperature = 0.8
    top_k = 40
    seed = 42
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = CharTokenizer()
    
    if not Path(data_path).exists():
        print(f"Error: Training data not found at {data_path}")
        print("Please ensure you have trained a model first.")
        sys.exit(1)
    
    tokenizer.build_vocab(data_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    print("Creating model...")
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=32,
        num_layers=2,
        ffn_hidden_dim=64
    )
    
    # Load checkpoint
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using the instructions in TRAINING.md")
        sys.exit(1)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_info = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint from step {checkpoint_info['step']}")
    print(f"Checkpoint loss: {checkpoint_info['loss']:.4f}")
    
    model.eval()
    
    # Generate text
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("-" * 60)
    
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        seed=seed
    )
    
    print(output)
    print("-" * 60)
    print(f"\nGenerated {len(output)} characters")


if __name__ == '__main__':
    main()
