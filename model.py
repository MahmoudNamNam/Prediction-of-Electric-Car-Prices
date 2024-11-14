import torch
import torch.nn as nn
import numpy as np

class StockPriceLSTM_v1(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super(StockPriceLSTM_v1, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_layer_size).requires_grad_()
        c_0 = torch.zeros(1, x.size(0), self.hidden_layer_size).requires_grad_()
        out, _ = self.lstm(x, (h_0.detach(), c_0.detach()))
        return self.linear(out[:, -1, :])

    @staticmethod
    def create_sequences(data, seq_length):
        sequences, labels = [], []
        for i in range(seq_length, len(data)):
            sequence = data[i - seq_length:i, 0]
            label = data[i, 0]
            
            sequences.append(sequence.astype(np.float32))
            labels.append(np.float32(label))

        return np.array(sequences), np.array(labels)


class StockPriceLSTM_v2(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1, num_layers=2, dropout=0.2):
        super(StockPriceLSTM_v2, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        # Define a multi-layer LSTM with dropout regularization
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, 
                            dropout=dropout, batch_first=True)
        
        # Output layer to map LSTM output to desired output size
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.linear(out[:, -1, :])

        return out
    
    @staticmethod
    def create_sequences(data, seq_length):
        sequences, labels = [], []
        for i in range(seq_length, len(data)):
            sequence = data[i - seq_length:i, 0]
            label = data[i, 0]
            
            sequences.append(sequence.astype(np.float32))
            labels.append(np.float32(label))

        return np.array(sequences), np.array(labels)
