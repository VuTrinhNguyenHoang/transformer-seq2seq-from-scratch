import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer
from .decoder_layer import DecoderLayer
from .positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_length: int, dropout: float):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=tgt.device)).bool()
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
    
    def generate(self, src, bos_id, eos_id, max_length=100):
        """
        Autoregressive greedy decoding for inference (translation)
        """
        device = src.device
        batch_size = src.size(0)

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        src_emb = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Start decoding with BOS
        ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            tgt_mask = self._generate_tgt_mask(ys.size(1)).to(device)
            tgt_emb = self.dropout(self.positional_encoding(self.decoder_embedding(ys)))

            dec_output = tgt_emb
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

            out = self.fc(dec_output[:, -1])  # lấy ra token cuối
            next_token = out.argmax(-1).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return ys[:, 1:]