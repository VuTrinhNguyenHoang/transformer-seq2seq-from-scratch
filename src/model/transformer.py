import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer
from .decoder_layer import DecoderLayer
from .positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_length: int, dropout: float):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate(self, src, max_length: int = 100, temperature: float = 1.0):
        """
        Generate a translation using beam search.
        
        Args:
            src (Tensor): Source sequence tensor [batch_size, src_seq_len]
            max_length (int): Maximum length of the generated sequence
            beam_size (int): Beam size for beam search
            temperature (float): Temperature for softmax, higher values = more diversity
            
        Returns:
            Tensor: Generated sequence tensor [batch_size, seq_len]
        """
        device = src.device
        batch_size = src.size(0)

        # Encode the source sequence
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Initialize with BOS token (assuming 1 is the BOS token)
        ys = torch.ones(batch_size, 1).fill_(1).long().to(device)

        for i in range(max_length-1):
            # Create mask for target
            tgt_mask = (ys != 0).unsqueeze(1).unsqueeze(3)
            seq_length = ys.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
            tgt_mask = tgt_mask & nopeak_mask

            # Decode 
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(ys)))
            
            dec_output = tgt_embedded
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

            out = self.fc(dec_output[:, -1])
            out = out / temperature

            # Get the most probable token
            prob = torch.softmax(out, dim=-1)
            next_word = torch.argmax(prob, dim=1)

            # Append to the sequence
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

            if (next_word == 2).sum() == batch_size:
                break
        
        return ys

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(src.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    