from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.utils.seed import set_seed
from src.nano_llm.train.train import train_with_config

# 1. Build tokenizer
tokenizer = CharTokenizer()
tokenizer.build_vocab('data/raw/chat.txt')

# 2. Load and encode data
with open('data/raw/chat.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokens = tokenizer.encode(text)

# Limit sequence length for training (nano model can't handle 10M tokens)
max_seq_len = 1000  # Use first 1000 tokens for nano model
tokens = tokens[:max_seq_len]

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Total tokens used for training: {len(tokens)}")

# 3. Configuration (NANO model)
config = {
    'vocab_size': tokenizer.vocab_size,
    'embedding_dim': 32,
    'num_layers': 2,
    'ffn_hidden_dim': 64,
    'learning_rate': 1e-3,
    'num_steps': 1000,
    'seed': 42,
    'checkpoint_dir': 'checkpoints/nano',
    'checkpoint_interval': 500
}

# 4. Set seed for reproducibility
set_seed(config['seed'])

# 5. Create model
model = TransformerLanguageModel(
    vocab_size=config['vocab_size'],
    embedding_dim=config['embedding_dim'],
    num_layers=config['num_layers'],
    ffn_hidden_dim=config['ffn_hidden_dim']
)

# 6. Train
print("Starting training...")
results = train_with_config(model, tokens, config)

# 7. Print results
print(f"Final loss: {results['final_loss']:.4f}")
print(f"Final gradient norm: {results['final_grad_norm']:.4f}")
print(f"Checkpoint saved to: {config['checkpoint_dir']}/checkpoint_final.pt")