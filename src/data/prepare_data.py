import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import sentencepiece as spm
from datasets import load_dataset

class TranslationDataset(Dataset):
    def __init__(self, source_ids, target_ids):
        self.source_ids = source_ids
        self.target_ids = target_ids

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx):
        x = torch.tensor(self.source_ids[idx], dtype=torch.long)
        y = torch.tensor(self.target_ids[idx], dtype=torch.long)
        return x, y

def prepare_data(output_dir='data', vocab_size=16000, max_seq_length=128):
    """
    Prepare dataset for Machine Translation using IWSLT2015 EN-VI
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load IWSLT2015 EN-VI dataset
    dataset = load_dataset("thainq107/iwslt2015-en-vi")
    with open(os.path.join(output_dir, "translate_en_vi.txt"), "w", encoding="utf-8") as f:
        for split in ['train', 'validation', 'test']:
            for example in dataset[split]:
                f.write(example['en'] + "\n")
                f.write(example['vi'] + "\n")
    
    # Train sentencepiece tokenizer
    spm.SentencePieceTrainer.Train(
        input=os.path.join(output_dir, "translate_en_vi.txt"),
        model_prefix=os.path.join(output_dir, "tokenizer"),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe"
    )

    # Load trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(output_dir, "tokenizer.model"))

    # Tokenize and save datasets
    for split in ['train', 'validation', 'test']:
        source_ids = []
        target_ids = []
        for example in dataset[split]:
            src = sp.encode(example['en'], out_type=int)
            tgt = sp.encode(example['vi'], out_type=int)

            # Truncation 
            src = src[:max_seq_length]
            tgt = tgt[:max_seq_length]

            source_ids.append(src)
            target_ids.append(tgt)
        
        np.save(os.path.join(output_dir, f"{split}_source.npy"), np.array(source_ids, dtype=object))
        np.save(os.path.join(output_dir, f"{split}_target.npy"), np.array(target_ids, dtype=object))

    # Save vocab size for later
    stats = {
        "vocab_size": vocab_size,
        "train_samples": len(dataset['train']),
        "val_samples": len(dataset['validation']),
        "test_samples": len(dataset['test'])
    }

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f)

    print("Data preparation completed.")
    return stats

def collect_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

def get_dataloaders(data_dir, batch_size):
    """
    Load prepared data and return dataloaders
    """
    # Load numpy arrays
    train_src = np.load(os.path.join(data_dir, "train_source.npy"), allow_pickle=True)
    train_tgt = np.load(os.path.join(data_dir, "train_target.npy"), allow_pickle=True)
    val_src = np.load(os.path.join(data_dir, "validation_source.npy"), allow_pickle=True)
    val_tgt = np.load(os.path.join(data_dir, "validation_target.npy"), allow_pickle=True)
    test_src = np.load(os.path.join(data_dir, "test_source.npy"), allow_pickle=True)
    test_tgt = np.load(os.path.join(data_dir, "test_target.npy"), allow_pickle=True)
    
    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt)
    val_dataset = TranslationDataset(val_src, val_tgt)
    test_dataset = TranslationDataset(test_src, test_tgt)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_batch)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare IWSLT2015 EN-VI dataset for training")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--vocab_size", type=int, default=16000, help="Vocabulary size for tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length after tokenization")

    args = parser.parse_args()
    
    stats = prepare_data(output_dir=args.output_dir, vocab_size=args.vocab_size, max_seq_length=args.max_seq_length)
    
    print(f"Data preparation completed. Stats: {stats}")
