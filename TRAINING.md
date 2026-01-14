# TRAINING.md — Nano-LLM Training Runbook

**Last Updated:** Phase 10  
**Purpose:** Complete guide to training and scaling character-level Transformer language models  
**Audience:** Users with basic Python knowledge, no ML background required

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start: Training a Nano Model](#quick-start-training-a-nano-model)
4. [Understanding the Output](#understanding-the-output)
5. [Loading and Using a Trained Model](#loading-and-using-a-trained-model)
6. [Scaling Guide: Nano → Small → Medium](#scaling-guide-nano--small--medium)
7. [Configuration Reference](#configuration-reference)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**What is nano-llm?**
A minimal, character-level Transformer language model built from first principles.

**Architecture:**

- Character-level tokenizer (no external dependencies)
- Multi-layer Transformer with causal self-attention
- Stateless model design
- Feed-forward networks with residual connections and layer normalization

**Training Approach:**

- Next-token prediction (autoregressive language modeling)
- Cross-entropy loss minimization
- AdamW optimizer
- Checkpoint-based training with resumption support

**Current State:**

- No CLI interface
- Training via Python scripts
- Configuration-driven scaling (no code changes needed)

---

## Prerequisites

### System Requirements

- Python 3.8+
- PyTorch (installed via `pip install torch`)
- NumPy (installed automatically with PyTorch)

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

### Project Structure

```structure
nano-llm/
├── data/raw/              # Training data goes here
├── checkpoints/           # Saved model checkpoints
├── experiments/           # Scaling configurations and runners
└── src/nano_llm/          # Core implementation (DO NOT MODIFY)
```

---

## Quick Start: Training a Nano Model

### Step 1: Prepare Training Data

Place a text file in `data/raw/`:

```bash
# Example: Use the provided sample
cat data/raw/sample.txt

# Or add your own data
cp /path/to/your/text.txt data/raw/my_data.txt
```

**Data Requirements:**

- UTF-8 encoded plain text
- Minimum 10KB for nano models
- Larger is better (reduces overfitting)

### Step 2: Create a Training Script

Create `train_nano.py` in the project root:

```python
from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.utils.seed import set_seed
from src.nano_llm.train.train import train_with_config

# 1. Build tokenizer
tokenizer = CharTokenizer()
tokenizer.build_vocab('data/raw/sample.txt')

# 2. Load and encode data
with open('data/raw/sample.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokens = tokenizer.encode(text)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Total tokens: {len(tokens)}")

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
```

### Step 3: Run Training

```bash
python train_nano.py
```

**Expected Output:**

```output
Vocabulary size: 42
Total tokens: 523
Starting training...
Final loss: 2.1234
Final gradient norm: 0.5678
Checkpoint saved to: checkpoints/nano/checkpoint_final.pt
```

### Step 4: Verify Checkpoints

```bash
ls checkpoints/nano/
# Output: checkpoint_step_500.pt  checkpoint_final.pt
```

---

## Understanding the Output

### Loss Behavior

**What is loss?**

- Measures how well the model predicts the next character
- Lower is better
- Typical nano model: Final loss 1.5-3.0

**Healthy training:**

- Loss decreases consistently
- No sudden spikes
- Gradient norm stays stable (0.1-5.0 range)

**Warning signs:**

- Loss increases: Learning rate too high
- Loss = NaN: Gradient explosion (reduce learning rate)
- Loss plateaus early: Model too small or data too easy

### Gradient Norm

**What is gradient norm?**

- Measures the size of weight updates during training
- Indicates training stability

**Healthy range:** 0.1 - 5.0

- Too low (<0.01): Learning stalled
- Too high (>10.0): Risk of instability
- NaN or Inf: Training failure

### Checkpoints

Each checkpoint file contains:

- `model_state_dict`: All model weights
- `optimizer_state_dict`: Optimizer state for resumption
- `step`: Training step number
- `loss`: Loss value at checkpoint
- `config`: Full configuration snapshot

**Resume training from checkpoint:**

```python
results = train_with_config(
    model, 
    tokens, 
    config, 
    resume_from='checkpoints/nano/checkpoint_step_500.pt'
)
```

---

## Loading and Using a Trained Model

### Load a Checkpoint

```python
import torch
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.train.train import load_checkpoint

# Create model with SAME configuration as training
model = TransformerLanguageModel(
    vocab_size=42,  # Must match training
    embedding_dim=32,
    num_layers=2,
    ffn_hidden_dim=64
)

# Load checkpoint
checkpoint_info = load_checkpoint('checkpoints/nano/checkpoint_final.pt', model)

print(f"Loaded checkpoint from step {checkpoint_info['step']}")
print(f"Checkpoint loss: {checkpoint_info['loss']:.4f}")

model.eval()  # IMPORTANT: Set to evaluation mode
```

### Generate Text

```python
from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.inference.generate import generate_text

# Rebuild tokenizer (must use same data as training)
tokenizer = CharTokenizer()
tokenizer.build_vocab('data/raw/sample.txt')

# Generate text
output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="The quick",
    max_length=100,
    temperature=0.8,
    top_k=40,
    seed=42
)

print(output)
```

### Temperature Control

**Temperature** controls randomness:

- `temperature=0.5`: Very deterministic, repetitive
- `temperature=1.0`: Balanced (default)
- `temperature=1.5`: Creative, more random
- `temperature=2.0`: Very random, may be incoherent

### Top-K Sampling

**Top-k** limits vocabulary during sampling:

- `top_k=None`: Sample from all characters
- `top_k=10`: Only sample from top 10 most likely
- `top_k=1`: Greedy decoding (always pick most likely)

**Recommended:** `top_k=40, temperature=0.8`

---

## Scaling Guide: Nano → Small → Medium

### Overview

**Scaling Philosophy:**

- Same architecture, different hyperparameters
- NO code changes required
- Change only configuration values

### Nano Model (Baseline)

**Configuration:**

```python
config_nano = {
    'vocab_size': tokenizer.vocab_size,
    'embedding_dim': 32,
    'num_layers': 2,
    'ffn_hidden_dim': 64,
    'learning_rate': 1e-3,
    'num_steps': 1000,
    'checkpoint_dir': 'checkpoints/nano'
}
```

**Data Requirements:** 10-50KB of text  
**Training Time:** ~1-2 minutes on CPU  
**Parameters:** ~10,000

### Small Model (2x Scale)

**Configuration:**

```python
config_small = {
    'vocab_size': tokenizer.vocab_size,
    'embedding_dim': 64,        # 2x
    'num_layers': 4,            # 2x
    'ffn_hidden_dim': 128,      # 2x
    'learning_rate': 1e-3,
    'num_steps': 3000,          # 3x (needs more training)
    'checkpoint_dir': 'checkpoints/small'
}
```

**Data Requirements:** 100KB - 1MB of text  
**Training Time:** ~5-10 minutes on CPU  
**Parameters:** ~50,000

**Critical:**

- Use MORE data than nano (risk of overfitting)
- Increase `num_steps` proportionally
- Change `checkpoint_dir` to avoid overwriting nano

### Medium Model (4x Scale)

**Configuration:**

```python
config_medium = {
    'vocab_size': tokenizer.vocab_size,
    'embedding_dim': 128,       # 4x from nano
    'num_layers': 6,            # 3x from nano
    'ffn_hidden_dim': 256,      # 4x from nano
    'learning_rate': 5e-4,      # REDUCED for stability
    'num_steps': 10000,         # 10x from nano
    'checkpoint_dir': 'checkpoints/medium'
}
```

**Data Requirements:** 1-10MB of text (REQUIRED)  
**Training Time:** 30-60 minutes on CPU (GPU recommended)  
**Parameters:** ~300,000

**Critical:**

- MUST use large dataset (will overfit on small data)
- Reduce learning rate to 5e-4 or 3e-4
- Consider GPU training (CPU becomes very slow)

### Scaling Checklist

**Before scaling:**

- [ ] Nano model trains successfully
- [ ] Loss decreases to reasonable value
- [ ] Generated text is better than random
- [ ] Have larger dataset ready

**When scaling to Small:**

- [ ] Change `embedding_dim` to 64
- [ ] Change `num_layers` to 4
- [ ] Change `ffn_hidden_dim` to 128
- [ ] Increase `num_steps` to 3000+
- [ ] Update `checkpoint_dir`
- [ ] Use dataset 10x larger than nano

**When scaling to Medium:**

- [ ] Change `embedding_dim` to 128
- [ ] Change `num_layers` to 6
- [ ] Change `ffn_hidden_dim` to 256
- [ ] Reduce `learning_rate` to 5e-4
- [ ] Increase `num_steps` to 10000+
- [ ] Update `checkpoint_dir`
- [ ] Use dataset 100x larger than nano
- [ ] Consider GPU training

---

## Configuration Reference

### Required Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `vocab_size` | Number of unique characters | Automatic from tokenizer |
| `embedding_dim` | Token embedding dimension | 32 (nano), 64 (small), 128 (medium) |
| `num_layers` | Number of Transformer blocks | 2 (nano), 4 (small), 6 (medium) |
| `ffn_hidden_dim` | Feed-forward hidden size | 64 (nano), 128 (small), 256 (medium) |
| `learning_rate` | Optimizer learning rate | 1e-3 (small), 5e-4 (medium) |
| `num_steps` | Training iterations | 1000 (nano), 3000 (small), 10000 (medium) |
| `seed` | Random seed | 42 (recommended) |
| `checkpoint_dir` | Checkpoint save location | 'checkpoints/nano' |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size (not fully implemented) | 1 |
| `checkpoint_interval` | Save every N steps | None (only final) |

### Parameter Relationships

**Model size** = f(embedding_dim, num_layers, ffn_hidden_dim)

- Doubling embedding_dim ~quadruples parameters
- Adding layers increases parameters linearly
- Doubling ffn_hidden_dim ~doubles parameters

**Training time** = f(model_size, num_steps, sequence_length)

- Larger models = slower per step
- More steps = linear time increase
- Longer sequences = quadratic attention cost

---

## Troubleshooting

### Problem: "Token sequence must have at least 2 tokens"

**Cause:** Data file is empty or too small  
**Solution:** Use a larger text file (minimum 100 characters)

### Problem: Loss = NaN

**Cause:** Gradient explosion  
**Solution:**

- Reduce learning_rate to 5e-4 or 1e-4
- Check data quality (remove special characters if needed)
- Reduce model size

### Problem: Loss not decreasing

**Possible causes:**

1. Learning rate too low → Increase to 1e-3
2. Model too small → Increase embedding_dim or num_layers
3. Data too complex → Try simpler/cleaner text
4. Data too small → Add more training data

### Problem: Generated text is gibberish

**Possible causes:**

1. Not trained long enough → Increase num_steps
2. Model too small → Scale to larger configuration
3. Temperature too high → Reduce to 0.5-1.0
4. Data quality poor → Use cleaner, more structured text

### Problem: Training very slow

**Solutions:**

1. Reduce num_steps (accept slightly worse quality)
2. Reduce embedding_dim or num_layers
3. Use GPU: `model.cuda()` and `tokens_tensor = tokens_tensor.cuda()`
4. Reduce data size (truncate sequence length)

### Problem: Out of memory

**Solutions:**

1. Reduce embedding_dim
2. Reduce num_layers
3. Reduce sequence length (truncate tokens list)
4. Use smaller dataset

---

## Advanced: Running Scaling Experiments

The project includes pre-configured scaling experiments.

### Run All Scaling Experiments

```bash
python experiments/run_scaling_experiments.py
```

**Output:**

- `experiments/results/capacity_scaling.json`
- `experiments/results/context_scaling.json`
- `experiments/results/depth_scaling.json`
- `experiments/results/width_scaling.json`

### Interpret Results

Each JSON file contains:

```json
{
  "experiment_name": "small",
  "num_parameters": 52345,
  "training_time_seconds": 120.5,
  "results": {
    "final_val_loss": 2.34,
    "final_val_perplexity": 10.38
  }
}
```

**Compare experiments by:**

- Parameters vs. validation loss (scaling efficiency)
- Training time vs. model size (computational cost)
- Perplexity (lower is better)

---

## Best Practices

### For Nano Models

- Use clean, simple text (children's books, Wikipedia articles)
- Keep data small (10-50KB)
- Train for 1000 steps minimum
- Expect loss ~2.0-3.0

### For Small Models

- Use diverse, well-structured text
- Require 100KB-1MB minimum
- Train for 3000-5000 steps
- Expect loss ~1.5-2.5

### For Medium Models

- Use large, high-quality corpus
- Require 1-10MB minimum
- Train for 10000+ steps
- Use GPU if available
- Expect loss ~1.0-2.0

### General

- Always set a seed for reproducibility
- Save checkpoints frequently
- Monitor gradient norms (should be 0.1-5.0)
- Test generation early and often
- Keep separate checkpoint directories for each experiment

---

## Summary

**Key Takeaways:**

1. Training requires: data preparation, tokenizer building, config creation, and running training script
2. Scaling requires ONLY changing configuration values, no code changes
3. Larger models need more data, more steps, and more time
4. Checkpoints enable resumption and model reuse
5. Generation quality depends on model size, training duration, and data quality

**Next Steps:**

1. Train a nano model on sample data
2. Verify generation works
3. Scale to small with more data
4. Experiment with temperature and top-k sampling
5. If results are good, scale to medium

**Remember:**

- DO NOT modify files in `src/nano_llm/`
- Change ONLY configuration values
- Keep architecture fixed
- Scale via hyperparameters, not code
