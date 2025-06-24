import os
import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from tqdm import tqdm
import sentencepiece as spm
import sacrebleu

from src.model.transformer import Transformer
from src.data.prepare_data import get_dataloaders

def evaluate(args):
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model_args = checkpoint['model_args']

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.data_dir, "tokenizer.model"))

    # Create model
    model = Transformer(
        vocab_size=model_args['d_model'],
        d_model=model_args['d_model'],
        num_heads=model_args['num_heads'],
        num_layers=model_args['num_layers'],
        d_ff=model_args['d_ff'],
        max_seq_length=model_args['max_seq_length'],
        dropout=0.0  # no dropout during evaluation
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model.to(device)
    
    # Create dataloaders
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Evaluation
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass
            logits = model(src, tgt[:, :-1])
            
            # Compute loss
            loss = criterion(logits.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
            mask = tgt[:, 1:] != 0
            
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
            
            # Count correct predictions
            predictions = logits.argmax(-1)
            correct_predictions += ((predictions == tgt[:, 1:]) & mask).sum().item()
            
            # Generate hypothesis sentences (greedy decode only for BLEU)
            for i in range(src.size(0)):
                pred_ids = greedy_decode(model, src[i:i+1], sp, device, max_len=model_args['max_seq_length'])
                ref_text = sp.decode(tgt[i].cpu().tolist())
                hyp_text = sp.decode(pred_ids)
                references.append([ref_text])
                hypotheses.append(hyp_text)
    
    # Compute perplexity and accuracy
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    accuracy = correct_predictions / total_tokens
    bleu = sacrebleu.corpus_bleu(hypotheses, references).score

    # Print evaluation results
    print(f"Evaluation results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  BLEU: {bleu:.2f}")
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "bleu": bleu
    }

def greedy_decode(model, src, sp, device, max_len=100):
    model.eval()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    src_emb = model.dropout(model.positional_encoding(model.encoder_embedding(src)))
    for enc_layer in model.encoder_layers:
        src_emb = enc_layer(src_emb, src_mask)
    enc_output = src_emb

    ys = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = model._generate_tgt_mask(ys.size(1)).to(device)
        tgt_emb = model.dropout(model.positional_encoding(model.decoder_embedding(ys)))
        dec_output = tgt_emb
        for dec_layer in model.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        out = model.fc(dec_output[:, -1])
        next_word = out.argmax(-1).unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        if next_word.item() == eos_id:
            break

    return ys[0, 1:].cpu().tolist()  # Bỏ BOS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GPT-mini model")
    
    # Data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with processed data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    args = parser.parse_args()
    
    evaluate(args)
