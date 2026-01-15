"""Interactive CLI chat interface for nano-llm."""
import sys
from pathlib import Path
from src.nano_llm.tokenizer.tokenizer import CharTokenizer
from src.nano_llm.model.model import TransformerLanguageModel
from src.nano_llm.train.train import load_checkpoint
from src.nano_llm.inference.generate import generate_text


def main():
    """Start an interactive chat session with the trained model."""
    # Default configuration
    data_path = 'data/raw/chat.txt'
    checkpoint_path = 'checkpoints/nano/checkpoint_final.pt'
    max_tokens_per_response = 150
    temperature = 0.8
    top_k = 40
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = CharTokenizer()
    
    if not Path(data_path).exists():
        print(f"Error: Training data not found at {data_path}")
        print("Please ensure you have trained a model first.")
        sys.exit(1)
    
    tokenizer.build_vocab(data_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Get supported characters for input filtering
    supported_chars = set(tokenizer.stoi.keys())
    
    # Create model
    print("Creating model...")
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,  # Updated to match new training config
        num_layers=3,  # Updated to match new training config (reduced from 4 to 3)
        ffn_hidden_dim=128  # Updated to match new training config
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
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize conversation buffer
    conversation_buffer = ""
    
    # Start interactive loop
    print("\n" + "=" * 60)
    print("Interactive Chat Session Started")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 60 + "\n")
    
    while True:
        # Get user input
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nEnding chat session...")
            break
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit']:
            print("Ending chat session...")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Normalize input: convert to lowercase and filter unsupported characters
        normalized_input = ''.join(c for c in user_input.lower() if c in supported_chars)
        
        # Skip if normalization resulted in empty string
        if not normalized_input:
            print("(Input contains only unsupported characters, skipping)\n")
            continue
        
        # Append normalized user input to conversation buffer
        conversation_buffer += f"user: {normalized_input}\nassistant: "
        
        # Generate response using the conversation buffer as prompt
        try:
            generated_output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=conversation_buffer,
                max_length=len(tokenizer.encode(conversation_buffer)) + max_tokens_per_response,
                temperature=temperature,
                top_k=top_k,
                seed=None  # Allow randomness across turns
            )
            
            # Extract only the newly generated portion (after the prompt)
            response = generated_output[len(conversation_buffer):]
            
            # Find natural stopping point (newline or end of response)
            # Take response up to first "user:" to avoid the model generating both sides
            if "\nuser:" in response:
                response = response.split("\nuser:")[0]
            
            # Clean up trailing whitespace
            response = response.strip()
            
            # Print the assistant's response
            print(f"Assistant: {response}\n")
            
            # Update conversation buffer with the actual response
            conversation_buffer += response + "\n"
            
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Try a different input or restart the session.\n")
            # Remove the failed prompt from buffer
            conversation_buffer = conversation_buffer.rsplit("user:", 1)[0]
    
    print("Goodbye!")


if __name__ == '__main__':
    main()
