"""Main entry point for training module."""
import sys
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.utils.seed import set_seed
from src.nano_llm.train.train import train_with_config

def main():
    """Run nano model training."""
    print("Starting Nano-LLM Training...")
    
    # Use chat.txt if available, otherwise sample.txt
    data_path = 'data/raw/chat.txt'
    if not Path(data_path).exists():
        data_path = 'data/raw/sample.txt'
        print(f"Using {data_path} for training")
    
    # Build tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(data_path)
    
    # Load and encode data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total tokens in dataset: {len(tokens)}")
    
    # For nano model, train on chunks to handle large datasets
    # Optimized chunk_size for balance between coverage and speed
    chunk_size = 2000  # Reduced to 2000 for faster training
    num_chunks = min(len(tokens) // chunk_size, 50)  # Increased to 50 chunks
    
    if num_chunks == 0:
        num_chunks = 1
        chunk_size = min(len(tokens), 2000)
    
    print(f"Training on {num_chunks} chunks of up to {chunk_size} tokens each")
    
    # Configuration - optimized for better convergence
    config = {
        'vocab_size': tokenizer.vocab_size,
        'embedding_dim': 64,  # Increased from 32 for better representation
        'num_layers': 3,  # Balanced at 3 layers to prevent crashes
        'ffn_hidden_dim': 128,  # Increased from 64 for better learning
        'learning_rate': 3e-4,  # Reduced for more stable convergence
        'num_steps': 200,  # 200 steps per chunk (10000 total with 50 chunks)
        'seed': 42,
        'checkpoint_dir': 'checkpoints/nano',
        'checkpoint_interval': None  # Only save final to save time
    }
    
    # Set seed
    set_seed(config['seed'])
    
    # Create model
    model = TransformerLanguageModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        ffn_hidden_dim=config['ffn_hidden_dim']
    )
    
    # Train on each chunk
    print("Training in progress...")
    total_loss = 0.0
    total_grad_norm = 0.0
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        
        print(f"\nChunk {i+1}/{num_chunks}: {len(chunk_tokens)} tokens")
        
        # Train on this chunk
        results = train_with_config(model, chunk_tokens, config)
        
        total_loss += results['final_loss']
        total_grad_norm += results['final_grad_norm']
        
        # Print progress
        print(f"  Final loss: {results['final_loss']:.4f}")
        print(f"  Avg loss so far: {total_loss / (i + 1):.4f}")
    
    # Average results
    avg_loss = total_loss / num_chunks
    avg_grad_norm = total_grad_norm / num_chunks
    
    # Print results
    print(f"Average final loss: {avg_loss:.4f}")
    print(f"Average final gradient norm: {avg_grad_norm:.4f}")
    print(f"Checkpoint saved to: {config['checkpoint_dir']}/checkpoint_final.pt")
    print("Training completed successfully!")

if __name__ == '__main__':
    main()