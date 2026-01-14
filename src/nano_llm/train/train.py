import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import math
import time
import time


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in model.
    
    Args:
        model: Model instance
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_neural_bigram(
    model: torch.nn.Module,
    tokens: List[int],
    num_epochs: int,
    learning_rate: float,
    seed: int = None
) -> Dict:
    """Train neural bigram language model.
    
    Args:
        model: NeuralBigramModel instance
        tokens: List of integer token IDs
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing training history
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Convert tokens to tensors
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    # Prepare bigram pairs: (x[:-1], x[1:])
    if len(tokens_tensor) < 2:
        raise ValueError("Token sequence must have at least 2 tokens")
    
    inputs = tokens_tensor[:-1]
    targets = tokens_tensor[1:]
    
    loss_history = []
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        
        # Compute loss
        loss = F.cross_entropy(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else None
    }


def train_mlp_model(
    model: torch.nn.Module,
    tokens: List[int],
    context_length: int,
    num_epochs: int,
    learning_rate: float,
    seed: int = None
) -> Dict:
    """Train MLP language model with fixed context window.
    
    Args:
        model: FixedContextMLPModel instance
        tokens: List of integer token IDs
        context_length: Number of previous tokens to use as context
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing training history
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Convert tokens to tensors
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    # Create sliding window batches
    if len(tokens_tensor) < context_length + 1:
        raise ValueError(f"Token sequence must have at least {context_length + 1} tokens")
    
    # Build input-target pairs using sliding windows
    inputs_list = []
    targets_list = []
    
    for i in range(len(tokens_tensor) - context_length):
        context = tokens_tensor[i:i + context_length]
        target = tokens_tensor[i + context_length]
        inputs_list.append(context)
        targets_list.append(target)
    
    inputs = torch.stack(inputs_list)
    targets = torch.tensor(targets_list, dtype=torch.long)
    
    loss_history = []
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        
        # Compute loss
        loss = F.cross_entropy(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else None
    }


def train_attention_model(
    model: torch.nn.Module,
    tokens: List[int],
    num_epochs: int,
    learning_rate: float,
    seed: int = None
) -> Dict:
    """Train attention-based language model.
    
    Args:
        model: AttentionLanguageModel instance
        tokens: List of integer token IDs
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing training history
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Convert tokens to tensors
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < 2:
        raise ValueError("Token sequence must have at least 2 tokens")
    
    # Prepare full sequence input and targets
    # Input: [0, 1, 2, ..., n-1], Target: [1, 2, 3, ..., n]
    inputs = tokens_tensor[:-1].unsqueeze(0)  # [1, seq_len]
    targets = tokens_tensor[1:].unsqueeze(0)  # [1, seq_len]
    
    loss_history = []
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)  # [1, seq_len, vocab_size]
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # [seq_len, vocab_size]
        targets_flat = targets.view(-1)  # [seq_len]
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else None
    }


def compute_loss(model: torch.nn.Module, tokens: List[int]) -> float:
    """Compute cross-entropy loss on token sequence.
    
    Args:
        model: NeuralBigramModel instance
        tokens: List of integer token IDs
        
    Returns:
        Cross-entropy loss
    """
    model.eval()
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < 2:
        return 0.0
    
    inputs = tokens_tensor[:-1]
    targets = tokens_tensor[1:]
    
    with torch.no_grad():
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
    
    return loss.item()


def compute_mlp_loss(model: torch.nn.Module, tokens: List[int], context_length: int) -> float:
    """Compute cross-entropy loss on token sequence for MLP model.
    
    Args:
        model: FixedContextMLPModel instance
        tokens: List of integer token IDs
        context_length: Number of previous tokens to use as context
        
    Returns:
        Cross-entropy loss
    """
    model.eval()
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < context_length + 1:
        return 0.0
    
    # Build input-target pairs using sliding windows
    inputs_list = []
    targets_list = []
    
    for i in range(len(tokens_tensor) - context_length):
        context = tokens_tensor[i:i + context_length]
        target = tokens_tensor[i + context_length]
        inputs_list.append(context)
        targets_list.append(target)
    
    inputs = torch.stack(inputs_list)
    targets = torch.tensor(targets_list, dtype=torch.long)
    
    with torch.no_grad():
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
    
    return loss.item()


def compute_attention_loss(model: torch.nn.Module, tokens: List[int]) -> float:
    """Compute cross-entropy loss on token sequence for attention model.
    
    Args:
        model: AttentionLanguageModel instance
        tokens: List of integer token IDs
        
    Returns:
        Cross-entropy loss
    """
    model.eval()
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < 2:
        return 0.0
    
    inputs = tokens_tensor[:-1].unsqueeze(0)  # [1, seq_len]
    targets = tokens_tensor[1:].unsqueeze(0)  # [1, seq_len]
    
    with torch.no_grad():
        logits = model(inputs)  # [1, seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
    
    return loss.item()


def train_transformer_model(
    model: torch.nn.Module,
    tokens: List[int],
    num_epochs: int,
    learning_rate: float,
    seed: int = None
) -> Dict:
    """Train Transformer language model.
    
    Args:
        model: TransformerLanguageModel instance
        tokens: List of integer token IDs
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing training history
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Convert tokens to tensors
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < 2:
        raise ValueError("Token sequence must have at least 2 tokens")
    
    # Prepare full sequence input and targets
    inputs = tokens_tensor[:-1].unsqueeze(0)  # [1, seq_len]
    targets = tokens_tensor[1:].unsqueeze(0)  # [1, seq_len]
    
    loss_history = []
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)  # [1, seq_len, vocab_size]
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # [seq_len, vocab_size]
        targets_flat = targets.view(-1)  # [seq_len]
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else None
    }


def compute_transformer_loss(model: torch.nn.Module, tokens: List[int]) -> float:
    """Compute cross-entropy loss on token sequence for Transformer model.
    
    Args:
        model: TransformerLanguageModel instance
        tokens: List of integer token IDs
        
    Returns:
        Cross-entropy loss
    """
    model.eval()
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < 2:
        return 0.0
    
    inputs = tokens_tensor[:-1].unsqueeze(0)  # [1, seq_len]
    targets = tokens_tensor[1:].unsqueeze(0)  # [1, seq_len]
    
    with torch.no_grad():
        logits = model(inputs)  # [1, seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
    
    return loss.item()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: Dict,
    checkpoint_dir: str,
    checkpoint_name: str = "checkpoint.pt"
) -> None:
    """Save training checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        step: Current training step
        loss: Current loss value
        config: Configuration dictionary
        checkpoint_dir: Directory to save checkpoint
        checkpoint_name: Name of checkpoint file
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    
    save_path = checkpoint_path / checkpoint_name
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load state into
        optimizer: Optional optimizer instance to load state into
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0.0),
        'config': checkpoint.get('config', {})
    }


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute L2 norm of model gradients.
    
    Args:
        model: Model instance
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def evaluate_model(
    model: torch.nn.Module,
    tokens: List[int]
) -> Dict[str, float]:
    """Evaluate model on token sequence.
    
    Args:
        model: Model instance
        tokens: List of integer token IDs
        
    Returns:
        Dictionary containing loss and perplexity
    """
    model.eval()
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < 2:
        return {'loss': 0.0, 'perplexity': 1.0}
    
    inputs = tokens_tensor[:-1].unsqueeze(0)  # [1, seq_len]
    targets = tokens_tensor[1:].unsqueeze(0)  # [1, seq_len]
    
    with torch.no_grad():
        logits = model(inputs)  # [1, seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
    
    loss_value = loss.item()
    perplexity = math.exp(loss_value)
    
    return {
        'loss': loss_value,
        'perplexity': perplexity
    }


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity (exp(loss))
    """
    return math.exp(loss)


def train_and_evaluate(
    model: torch.nn.Module,
    train_tokens: List[int],
    val_tokens: List[int],
    config: Dict,
    eval_interval: int,
    resume_from: Optional[str] = None
) -> Dict:
    """Train model with periodic evaluation.
    
    Args:
        model: Model instance
        train_tokens: Training token sequence
        val_tokens: Validation token sequence
        config: Configuration dictionary
        eval_interval: Evaluate every N steps
        resume_from: Optional checkpoint to resume from
        
    Returns:
        Dictionary containing training and evaluation history
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    start_step = 0
    if resume_from is not None:
        checkpoint_info = load_checkpoint(resume_from, model, optimizer)
        start_step = checkpoint_info['step'] + 1
    
    # Convert tokens to tensors
    train_tensor = torch.tensor(train_tokens, dtype=torch.long)
    
    if len(train_tensor) < 2:
        raise ValueError("Token sequence must have at least 2 tokens")
    
    # Prepare training data
    inputs = train_tensor[:-1].unsqueeze(0)
    targets = train_tensor[1:].unsqueeze(0)
    
    train_loss_history = []
    grad_norm_history = []
    val_loss_history = []
    val_perplexity_history = []
    eval_steps = []
    
    for step in range(start_step, config['num_steps']):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        grad_norm = compute_gradient_norm(model)
        optimizer.step()
        
        # Record metrics
        train_loss_history.append(loss.item())
        grad_norm_history.append(grad_norm)
        
        # Evaluate periodically
        if (step + 1) % eval_interval == 0:
            eval_metrics = evaluate_model(model, val_tokens)
            val_loss_history.append(eval_metrics['loss'])
            val_perplexity_history.append(eval_metrics['perplexity'])
            eval_steps.append(step + 1)
        
        # Save checkpoint if interval is specified
        checkpoint_interval = config.get('checkpoint_interval')
        if checkpoint_interval is not None and (step + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                loss=loss.item(),
                config=config,
                checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
                checkpoint_name=f"checkpoint_step_{step + 1}.pt"
            )
    
    # Final evaluation
    final_eval = evaluate_model(model, val_tokens)
    
    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=config['num_steps'] - 1,
        loss=train_loss_history[-1] if train_loss_history else 0.0,
        config=config,
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        checkpoint_name="checkpoint_final.pt"
    )
    
    return {
        'train_loss_history': train_loss_history,
        'grad_norm_history': grad_norm_history,
        'val_loss_history': val_loss_history,
        'val_perplexity_history': val_perplexity_history,
        'eval_steps': eval_steps,
        'final_train_loss': train_loss_history[-1] if train_loss_history else None,
        'final_val_loss': final_eval['loss'],
        'final_val_perplexity': final_eval['perplexity']
    }


def run_experiment(
    model: torch.nn.Module,
    train_tokens: List[int],
    val_tokens: List[int],
    config: Dict,
    experiment_name: str,
    eval_interval: int = 100,
    resume_from: Optional[str] = None
) -> Dict:
    """Run single scaling experiment with detailed logging.
    
    Args:
        model: Model instance
        train_tokens: Training token sequence
        val_tokens: Validation token sequence
        config: Configuration dictionary
        experiment_name: Name of the experiment
        eval_interval: Evaluate every N steps
        resume_from: Optional checkpoint to resume from
        
    Returns:
        Dictionary containing experiment results and metadata
    """
    start_time = time.time()
    
    # Count parameters
    num_params = count_parameters(model)
    
    # Run training with evaluation
    results = train_and_evaluate(
        model=model,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        config=config,
        eval_interval=eval_interval,
        resume_from=resume_from
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Compile experiment metadata
    experiment_results = {
        'experiment_name': experiment_name,
        'num_parameters': num_params,
        'config': config,
        'training_time_seconds': training_time,
        'results': results
    }
    
    return experiment_results


def run_multiple_experiments(
    experiments: List[Dict],
    train_tokens: List[int],
    val_tokens: List[int],
    eval_interval: int = 100
) -> List[Dict]:
    """Run multiple scaling experiments sequentially.
    
    Args:
        experiments: List of experiment configurations, each containing:
            - name: Experiment name
            - model_factory: Callable that returns model instance
            - config: Configuration dictionary
        train_tokens: Training token sequence
        val_tokens: Validation token sequence
        eval_interval: Evaluate every N steps
        
    Returns:
        List of experiment results
    """
    all_results = []
    
    for exp in experiments:
        exp_name = exp['name']
        model = exp['model_factory']()
        config = exp['config']
        
        results = run_experiment(
            model=model,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            config=config,
            experiment_name=exp_name,
            eval_interval=eval_interval
        )
        
        all_results.append(results)
    
    return all_results


def save_experiment_results(results: List[Dict], output_path: str) -> None:
    """Save experiment results to JSON file.
    
    Args:
        results: List of experiment results
        output_path: Path to save results
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def train_with_config(
    model: torch.nn.Module,
    tokens: List[int],
    config: Dict,
    resume_from: Optional[str] = None
) -> Dict:
    """Train model using configuration object.
    
    Args:
        model: Model instance
        tokens: List of integer token IDs
        config: Configuration dictionary containing training parameters
        resume_from: Optional path to checkpoint to resume from
        
    Returns:
        Dictionary containing training history
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    start_step = 0
    if resume_from is not None:
        checkpoint_info = load_checkpoint(resume_from, model, optimizer)
        start_step = checkpoint_info['step'] + 1
    
    # Convert tokens to tensors
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    if len(tokens_tensor) < 2:
        raise ValueError("Token sequence must have at least 2 tokens")
    
    # Prepare full sequence input and targets
    inputs = tokens_tensor[:-1].unsqueeze(0)  # [1, seq_len]
    targets = tokens_tensor[1:].unsqueeze(0)  # [1, seq_len]
    
    loss_history = []
    grad_norm_history = []
    
    model.train()
    for step in range(start_step, config['num_steps']):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm before optimizer step
        grad_norm = compute_gradient_norm(model)
        
        optimizer.step()
        
        # Record metrics
        loss_history.append(loss.item())
        grad_norm_history.append(grad_norm)
        
        # Save checkpoint if interval is specified
        checkpoint_interval = config.get('checkpoint_interval')
        if checkpoint_interval is not None and (step + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                loss=loss.item(),
                config=config,
                checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
                checkpoint_name=f"checkpoint_step_{step + 1}.pt"
            )
    
    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=config['num_steps'] - 1,
        loss=loss_history[-1] if loss_history else 0.0,
        config=config,
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        checkpoint_name="checkpoint_final.pt"
    )
    
    return {
        'loss_history': loss_history,
        'grad_norm_history': grad_norm_history,
        'final_loss': loss_history[-1] if loss_history else None,
        'final_grad_norm': grad_norm_history[-1] if grad_norm_history else None
    }
