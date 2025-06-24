import os
import torch
import sentencepiece as spm
import argparse

from src.model.transformer import Transformer

def generate_translation(model_path, tokenizer_path, prompt, max_new_tokens=50, device='cpu'):
    """
    Generate translation from English to Vietnamese using trained Transformer Seq2Seq
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']

    # Load tokenizer SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(tokenizer_path, "tokenizer.model"))

    vocab_size = model_args['vocab_size']

    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=model_args['d_model'],
        num_heads=model_args['num_heads'],
        num_layers=model_args['num_layers'],
        d_ff=model_args['d_ff'],
        max_seq_length=model_args['max_seq_length'],
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Encode input prompt
    src_ids = sp.encode(prompt, out_type=int)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate translation
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    with torch.no_grad():
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        src_emb = model.dropout(model.positional_encoding(model.encoder_embedding(src_tensor)))
        for enc_layer in model.encoder_layers:
            src_emb = enc_layer(src_emb, src_mask)
        enc_output = src_emb

        ys = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            tgt_mask = torch.tril(torch.ones((1, 1, ys.size(1), ys.size(1)), device=device)).bool()
            tgt_emb = model.dropout(model.positional_encoding(model.decoder_embedding(ys)))
            dec_output = tgt_emb
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            out = model.fc(dec_output[:, -1])
            next_token = out.argmax(-1).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        output_ids = ys[0, 1:].tolist()  # remove BOS

    translation = sp.decode(output_ids)
    return translation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate translation with trained Transformer Seq2Seq model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (best_model.pt)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to trained tokenizer")
    parser.add_argument("--prompt", type=str, required=True, help="English sentence to translate")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU inference")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    
    translation = generate_translation(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        device=device
    )

    print(f"English Input: {args.prompt}")
    print(f"Vietnamese Translation: {translation}")
