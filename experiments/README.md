# Scaling Experiments

This directory contains configurations and scripts for running scaling experiments with nano-llm.

## Experiment Suites

### 1. Capacity Scaling
Tests how model performance scales with overall capacity (parameters).

- **Nano**: 2 layers, 32-dim embeddings
- **Small**: 4 layers, 64-dim embeddings  
- **Medium**: 6 layers, 128-dim embeddings

### 2. Context Scaling
Tests how context length affects model performance.

- **Short**: 16 tokens
- **Medium**: 32 tokens
- **Long**: 64 tokens

### 3. Depth Scaling
Tests how number of layers affects performance (constant width).

- **Shallow**: 2 layers
- **Medium**: 4 layers
- **Deep**: 8 layers

### 4. Width Scaling
Tests how embedding dimension affects performance (constant depth).

- **Narrow**: 32-dim embeddings
- **Medium**: 64-dim embeddings
- **Wide**: 128-dim embeddings

## Running Experiments

```bash
# Run all scaling experiments
python experiments/run_scaling_experiments.py

# Or import and run specific suites
python -c "from experiments.run_scaling_experiments import run_capacity_scaling_experiments; run_capacity_scaling_experiments('data/raw/sample.txt')"
```

## Results

Results are saved as JSON files in `experiments/results/` with the following structure:

```json
{
  "experiment_name": "nano",
  "num_parameters": 12345,
  "config": {...},
  "training_time_seconds": 123.45,
  "results": {
    "train_loss_history": [...],
    "val_loss_history": [...],
    "val_perplexity_history": [...],
    "final_val_perplexity": 42.0
  }
}
```

## Analysis

Compare experiments by:
- Number of parameters vs final validation loss
- Training time vs model size
- Context length vs perplexity on long sequences
- Depth vs width trade-offs at similar parameter counts
