import torch
import torch.nn as nn
import numpy as np

class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super(StockPriceLSTM, self).__init__()
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
            
            # Ensure sequence and label are of numeric type
            sequences.append(sequence.astype(np.float32))
            labels.append(np.float32(label))

        return np.array(sequences), np.array(labels)
