import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_ff)  --> (batch_size, seq_length, d_model)
        return self.fc2(self.relu(self.fc1(x)))
