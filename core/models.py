import torch.nn as nn

class ForexForecasting(nn.Module):
    def __init__(self, features, **kwargs):
        super(ForexForecasting, self).__init__()

        lstm_hidden_size = kwargs.get('lmst_hidden_size', 76)
        dropout = kwargs.get('dropout', 0.2)

        self.name = f'ForexForecasting_{lstm_hidden_size}_{dropout}'

        self.lstm = nn.LSTM(features, lstm_hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.output(x)

        return x