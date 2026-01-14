# nano-llm

A minimal, character-level Transformer language model built from first principles.

## What is this?

`nano-llm` is an educational implementation of a decoder-only Transformer architecture trained on character-level text prediction. It demonstrates how language models work by implementing every component from scratch using PyTorch.

**This project is:**
- A character-level autoregressive language model
- A learning resource for understanding Transformers
- A platform for experimenting with model scaling

**This project is NOT:**
- A chat interface or conversational AI
- A question-answering system
- A retrieval-augmented system
- A production-ready language model
- Trained on external knowledge bases

## Architecture

- **Tokenization:** Character-level (no external dependencies)
- **Model:** Multi-layer Transformer with causal self-attention
- **Components:** Embedding layers, multi-head attention, feed-forward networks, residual connections, layer normalization
- **Training:** Next-token prediction with cross-entropy loss and AdamW optimizer
- **Inference:** Autoregressive generation with temperature and top-k sampling

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nano-llm.git
cd nano-llm

# Install dependencies
pip install torch numpy
# Or use uv:
uv sync
```

### Train a Nano Model

```bash
# Create a training script
cat > train_nano.py << 'EOF'
from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.utils.seed import set_seed
from src.nano_llm.train.train import train_with_config

# Build tokenizer
tokenizer = CharTokenizer()
tokenizer.build_vocab('data/raw/sample.txt')

# Load and encode data
with open('data/raw/sample.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokens = tokenizer.encode(text)

# Configuration
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

# Set seed and create model
set_seed(config['seed'])
model = TransformerLanguageModel(
    vocab_size=config['vocab_size'],
    embedding_dim=config['embedding_dim'],
    num_layers=config['num_layers'],
    ffn_hidden_dim=config['ffn_hidden_dim']
)

# Train
print("Starting training...")
results = train_with_config(model, tokens, config)
print(f"Final loss: {results['final_loss']:.4f}")
EOF

# Run training
python train_nano.py
```

### Generate Text

```bash
# After training, run inference
uv run python -m nano_llm.inference.generate

# Or create a custom generation script
cat > generate.py << 'EOF'
from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.train.train import load_checkpoint
from src.nano_llm.inference.generate import generate_text

# Build tokenizer (must use same data as training)
tokenizer = CharTokenizer()
tokenizer.build_vocab('data/raw/sample.txt')

# Create and load model
model = TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=32,
    num_layers=2,
    ffn_hidden_dim=64
)
load_checkpoint('checkpoints/nano/checkpoint_final.pt', model)
model.eval()

# Generate text
output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="The quick",
    max_length=200,
    temperature=0.8,
    top_k=40,
    seed=42
)
print(output)
EOF

python generate.py
```

### Run Scaling Experiments

```bash
# Run pre-configured experiments
uv run python experiments/run_scaling_experiments.py

# Results saved to experiments/results/
```

## Scaling Philosophy

The model scales through **configuration changes only** — no code modifications required.

| Model | embedding_dim | num_layers | ffn_hidden_dim | Parameters | Training Steps |
|-------|--------------|------------|----------------|------------|----------------|
| Nano  | 32           | 2          | 64             | ~10K       | 1,000          |
| Small | 64           | 4          | 128            | ~50K       | 3,000          |
| Medium| 128          | 6          | 256            | ~300K      | 10,000         |

See [TRAINING.md](TRAINING.md) for complete scaling guide.

## Documentation

**[TRAINING.md](TRAINING.md)** — Complete training runbook covering:
- Step-by-step training instructions
- Configuration reference
- Scaling from nano → small → medium
- Troubleshooting guide
- Checkpoint management
- Inference examples

## Project Structure

```
nano-llm/
├── src/nano_llm/          # Core implementation
│   ├── config/            # Model configuration
│   ├── inference/         # Generation and sampling
│   ├── model/             # Transformer architecture
│   ├── tokenizer/         # Character-level tokenizer
│   ├── train/             # Training loops and utilities
│   └── utils/             # Helper functions
├── experiments/           # Scaling experiments
├── data/raw/              # Training data
├── checkpoints/           # Saved model weights (gitignored)
├── TRAINING.md            # Complete training guide
└── README.md              # This file
```

## Limitations

- **Character-level only:** No subword or BPE tokenization
- **No batching:** Processes one sequence at a time
- **No distributed training:** Single GPU/CPU only
- **No chat interface:** Text generation only
- **Limited context:** Maximum sequence length defined at training
- **No instruction following:** Purely predictive, not instruction-tuned

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

This is an educational project. Contributions that improve clarity, fix bugs, or enhance documentation are welcome. Please do not submit changes that alter the core architecture or training philosophy.

## Citation

If you use this code for educational purposes, please cite:

```bibtex
@software{nano_llm,
  title = {nano-llm: A minimal character-level Transformer},
  author = {nano-llm contributors},
  year = {2026},
  url = {https://github.com/yourusername/nano-llm}
}
```
