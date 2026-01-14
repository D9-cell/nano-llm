"""Script to run scaling experiments for nano-llm.

This script runs multiple experiments to study how model performance
scales with different architectural choices.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.utils.seed import set_seed
from src.nano_llm.train.train import run_multiple_experiments, save_experiment_results
from experiments.scaling_configs import (
    CAPACITY_SCALING_SUITE,
    CONTEXT_SCALING_SUITE,
    DEPTH_SCALING_SUITE,
    WIDTH_SCALING_SUITE
)


def load_and_prepare_data(data_path: str, train_split: float = 0.8):
    """Load data and split into train/validation.
    
    Args:
        data_path: Path to raw text file
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (tokenizer, train_tokens, val_tokens)
    """
    # Initialize tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(data_path)
    
    # Load and encode text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    
    # Split into train/validation
    split_idx = int(len(tokens) * train_split)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    return tokenizer, train_tokens, val_tokens


def run_capacity_scaling_experiments(data_path: str):
    """Run experiments varying model capacity."""
    print("Running capacity scaling experiments...")
    
    tokenizer, train_tokens, val_tokens = load_and_prepare_data(data_path)
    
    experiments = []
    for name, config in CAPACITY_SCALING_SUITE:
        config['vocab_size'] = tokenizer.vocab_size
        
        def make_model(cfg=config):
            set_seed(cfg['seed'])
            return TransformerLanguageModel(
                vocab_size=cfg['vocab_size'],
                embedding_dim=cfg['embedding_dim'],
                num_layers=cfg['num_layers'],
                ffn_hidden_dim=cfg['ffn_hidden_dim']
            )
        
        experiments.append({
            'name': name,
            'model_factory': make_model,
            'config': config
        })
    
    results = run_multiple_experiments(
        experiments=experiments,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        eval_interval=100
    )
    
    save_experiment_results(results, 'experiments/results/capacity_scaling.json')
    print(f"Completed {len(results)} capacity scaling experiments")
    
    return results


def run_context_scaling_experiments(data_path: str):
    """Run experiments varying context length."""
    print("Running context scaling experiments...")
    
    tokenizer, train_tokens, val_tokens = load_and_prepare_data(data_path)
    
    experiments = []
    for name, config in CONTEXT_SCALING_SUITE:
        config['vocab_size'] = tokenizer.vocab_size
        
        def make_model(cfg=config):
            set_seed(cfg['seed'])
            return TransformerLanguageModel(
                vocab_size=cfg['vocab_size'],
                embedding_dim=cfg['embedding_dim'],
                num_layers=cfg['num_layers'],
                ffn_hidden_dim=cfg['ffn_hidden_dim']
            )
        
        experiments.append({
            'name': name,
            'model_factory': make_model,
            'config': config
        })
    
    results = run_multiple_experiments(
        experiments=experiments,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        eval_interval=100
    )
    
    save_experiment_results(results, 'experiments/results/context_scaling.json')
    print(f"Completed {len(results)} context scaling experiments")
    
    return results


def run_depth_scaling_experiments(data_path: str):
    """Run experiments varying model depth."""
    print("Running depth scaling experiments...")
    
    tokenizer, train_tokens, val_tokens = load_and_prepare_data(data_path)
    
    experiments = []
    for name, config in DEPTH_SCALING_SUITE:
        config['vocab_size'] = tokenizer.vocab_size
        
        def make_model(cfg=config):
            set_seed(cfg['seed'])
            return TransformerLanguageModel(
                vocab_size=cfg['vocab_size'],
                embedding_dim=cfg['embedding_dim'],
                num_layers=cfg['num_layers'],
                ffn_hidden_dim=cfg['ffn_hidden_dim']
            )
        
        experiments.append({
            'name': name,
            'model_factory': make_model,
            'config': config
        })
    
    results = run_multiple_experiments(
        experiments=experiments,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        eval_interval=100
    )
    
    save_experiment_results(results, 'experiments/results/depth_scaling.json')
    print(f"Completed {len(results)} depth scaling experiments")
    
    return results


def run_width_scaling_experiments(data_path: str):
    """Run experiments varying model width."""
    print("Running width scaling experiments...")
    
    tokenizer, train_tokens, val_tokens = load_and_prepare_data(data_path)
    
    experiments = []
    for name, config in WIDTH_SCALING_SUITE:
        config['vocab_size'] = tokenizer.vocab_size
        
        def make_model(cfg=config):
            set_seed(cfg['seed'])
            return TransformerLanguageModel(
                vocab_size=cfg['vocab_size'],
                embedding_dim=cfg['embedding_dim'],
                num_layers=cfg['num_layers'],
                ffn_hidden_dim=cfg['ffn_hidden_dim']
            )
        
        experiments.append({
            'name': name,
            'model_factory': make_model,
            'config': config
        })
    
    results = run_multiple_experiments(
        experiments=experiments,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        eval_interval=100
    )
    
    save_experiment_results(results, 'experiments/results/width_scaling.json')
    print(f"Completed {len(results)} width scaling experiments")
    
    return results


if __name__ == '__main__':
    data_path = 'data/raw/sample.txt'
    
    print("=" * 60)
    print("Nano-LLM Scaling Experiments")
    print("=" * 60)
    
    # Run all experiment suites
    run_capacity_scaling_experiments(data_path)
    run_context_scaling_experiments(data_path)
    run_depth_scaling_experiments(data_path)
    run_width_scaling_experiments(data_path)
    
    print("\nAll scaling experiments completed!")
    print("Results saved to experiments/results/")
