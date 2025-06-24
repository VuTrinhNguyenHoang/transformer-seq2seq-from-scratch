import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
import json

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        # Get block_size tokens from idx
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        # Target is the same sequence shifted by one position
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

def prepare_data(data_path, tokenizer_name='gpt2', block_size=128, output_dir='data', train_split=0.8, dev_split=0.1):
    """
    Prepare dataset for training GPT model
    
    Args:
        data_path: Path to raw text file
        tokenizer_name: Name of the tokenizer to use (e.g., 'gpt2')
        block_size: Size of the context window
        output_dir: Directory to save processed data
        train_split: Proportion of data to use for training
    
    Returns:
        Dictionary with dataset statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load and tokenize data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    paragraphs = text.split('\n')
    
    # Tokenize entire text
    encoded_text = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        encoded_paragraph = tokenizer.encode(paragraph, add_special_tokens=False)
        encoded_text.extend(encoded_paragraph)
    
    # Train/dev/valid split
    total_len = len(encoded_text)
    train_end = int(total_len * train_split)
    dev_end = train_end + int(total_len * dev_split)

    train_data = encoded_text[:train_end]
    dev_data = encoded_text[train_end:dev_end]
    val_data = encoded_text[dev_end:]

    # Save processed data
    with open(os.path.join(output_dir, 'train.bin'), 'wb') as f:
        np.save(f, np.array(train_data, dtype=np.int32))
    with open(os.path.join(output_dir, 'dev.bin'), 'wb') as f:
        np.save(f, np.array(dev_data, dtype=np.int32))
    with open(os.path.join(output_dir, 'val.bin'), 'wb') as f:
        np.save(f, np.array(val_data, dtype=np.int32))
    
    tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))
    
    stats = {
        'src_vocab_size': len(tokenizer),
        'tgt_vocab_size': len(tokenizer),
        'train_tokens': len(train_data),
        'dev_tokens': len(dev_data),
        'val_tokens': len(val_data),
        'max_seq_length': block_size,
        'tokenizer': tokenizer_name
    }
    
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)
    
    return stats

def collect_batch(batch):
    x_list, y_list = [], []
    for x, y in batch:
        x_list.append(x)
        y_list.append(y)

    # Pad sequences in the batch
    x_padded = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True, padding_value=0)
    y_padded = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True, padding_value=0)

    return x_padded, y_padded

def get_dataloaders(data_dir, block_size, batch_size):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir: Directory with processed data
        block_size: Size of the context window
        batch_size: Batch size for training
    
    Returns:
        train_loader, val_loader
    """
    # Load data
    train_data = np.load(os.path.join(data_dir, 'train.bin'), allow_pickle=True)
    dev_data = np.load(os.path.join(data_dir, 'dev.bin'), allow_pickle=True)
    val_data = np.load(os.path.join(data_dir, 'val.bin'), allow_pickle=True)
    
    # Create datasets
    train_dataset = TextDataset(train_data, block_size)
    dev_dataset = TextDataset(dev_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collect_batch)
    
    return train_loader, dev_loader, val_loader

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for training GPT model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw text file")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Name of the tokenizer to use")
    parser.add_argument("--block_size", type=int, default=128, help="Size of the context window")
    
    args = parser.parse_args()
    
    stats = prepare_data(
        data_path=args.data_path,
        tokenizer_name=args.tokenizer,
        block_size=args.block_size,
        output_dir=args.output_dir
    )
    
    print(f"Data preparation completed. Stats: {stats}")
