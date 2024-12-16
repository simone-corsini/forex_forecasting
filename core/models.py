import torch
import torch.nn as nn

import math

class ForexForecasting(nn.Module):
    def __init__(self, features, **kwargs):
        super(ForexForecasting, self).__init__()

        lstm_hidden_size = kwargs.get('lmst_hidden_size', 64)
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.2)

        self.name = f'ForexForecasting_h{lstm_hidden_size}_l{num_layers}_d{dropout}'

        self.lstm = nn.LSTM(features, lstm_hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.output(x)

        return x
    
class ForexForecastingSeq2Seq(nn.Module):
    def __init__(self, features, **kwargs):
        super(ForexForecastingSeq2Seq, self).__init__()

        lstm_hidden_size = kwargs.get('lmst_hidden_size', 64)
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.2)
        self.output_len = kwargs['output_len']

        self.name = f'ForexForecastingSeq2Seq_h{lstm_hidden_size}_l{num_layers}_{dropout}_{self.output_len}'

        self.encoder = nn.LSTM(features, lstm_hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.decoder = nn.LSTM(1, lstm_hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)

        decoder_input = torch.zeros((x.size(0), 1, 1), device=x.device)
        outputs = []

        for _ in range(self.output_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            decoder_output = self.dropout(decoder_output)
            decoder_input = self.output(decoder_output)
            outputs.append(decoder_input)

        outputs = torch.cat(outputs, dim=1)
        
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, features, **kwargs):
        super(TransformerModel, self).__init__()
        
        d_model = kwargs.get('d_model', 64)
        nhead = kwargs.get('nhead', 4)
        num_encoder_layers = kwargs.get('num_encoder_layers', 3)
        dropout = kwargs.get('dropout', 0.3)
        output_dim = kwargs.get('output_dim', 1)

        self.seq_len = kwargs['seq_len']
        self.output_len = kwargs['output_len']

        self.name = f'TransformerModel_dm{d_model}_nh{nhead}_nel{num_encoder_layers}_d{dropout}'

        self.input_projection = nn.Linear(features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=self.seq_len) # nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)  #x = x + self.positional_encoding[:, :self.seq_len, :]
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        x = x[:, -self.output_len:, :]
        return x.contiguous()
