import torch
from transformers import AutoTokenizer
from src.model.transformer import Transformer

def generate_text(model_path, prompt, max_new_tokens=50):
    """
    Generate text using a trained Transformer model
    
    Args:
        model_path: Path to the saved model
        prompt: Text prompt to start generation from
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated text including the prompt
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_args = checkpoint['model_args']
    model = Transformer(
        src_vocab_size=model_args['src_vocab_size'],
        tgt_vocab_size=model_args['tgt_vocab_size'],
        d_model=model_args['d_model'],
        num_heads=model_args['num_heads'],
        num_layers=model_args['num_layers'],
        d_ff=model_args['d_ff'],
        max_seq_length=model_args['max_seq_length'],
        dropout=0.0  # No dropout during evaluation
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer (assuming we saved tokenizer info with model)
    tokenizer_name = checkpoint.get('tokenizer_name')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=max_new_tokens
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text with Transformer")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    generated_text = generate_text(
        model_path=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens
    )
    
    print(f"Generated text:\n{generated_text}")
