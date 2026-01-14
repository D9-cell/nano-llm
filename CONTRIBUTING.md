# Contributing to nano-llm

Thank you for your interest in contributing to nano-llm!

## Project Philosophy

nano-llm is an **educational project** designed to teach how Transformer language models work. Contributions should prioritize clarity, simplicity, and educational value over performance or feature completeness.

## What We Welcome

- **Bug fixes**: Corrections to errors in code or documentation
- **Documentation improvements**: Clarifications, examples, or better explanations
- **Educational enhancements**: Comments, diagrams, or explanations that improve understanding
- **Reproducibility fixes**: Issues with setup, installation, or running experiments

## What We Do NOT Accept

- **Architecture changes**: The current Transformer design is intentionally simple and should remain unchanged
- **New features**: Chat interfaces, retrieval, instruction tuning, etc., are out of scope
- **Performance optimizations**: This is not a production system; clarity trumps speed
- **New dependencies**: The project should remain minimal (PyTorch + NumPy only)
- **Experimental features**: Keep the codebase focused and stable

## How to Contribute

1. **Open an issue first**: Describe the bug or improvement you'd like to make
2. **Fork the repository**: Create your own copy to work on
3. **Make focused changes**: Keep pull requests small and targeted
4. **Test your changes**: Ensure training and inference still work
5. **Update documentation**: If your change affects usage, update README.md or TRAINING.md
6. **Submit a pull request**: Reference the issue number and explain your changes

## Code Style

- Follow PEP 8 conventions
- Use type hints where helpful for clarity
- Add comments only when they improve understanding (avoid obvious comments)
- Keep functions focused and single-purpose
- Maintain consistency with existing code style

## Testing Changes

Before submitting, verify:

```bash
# Train a nano model
python train_nano.py

# Run inference
uv run python -m nano_llm.inference.generate

# Run scaling experiments
uv run python experiments/run_scaling_experiments.py
```

All commands should complete without errors.

## Documentation

If your change affects how users interact with the project:

- Update README.md for high-level changes
- Update TRAINING.md for training-related changes
- Provide clear examples of new behavior
- Ensure commands are copy-pasteable and work from a fresh clone

## Questions?

Open an issue for discussion before starting work on significant changes.

Thank you for helping make nano-llm a better educational resource!
