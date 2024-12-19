import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class ForexForecasting(nn.Module):
    def __init__(self, features, seq_len, output_len, **kwargs):
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
    def __init__(self, features, seq_len, output_len, **kwargs):
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

class ForexForecastingTransformer(nn.Module):
    def __init__(self, features, seq_len, output_len, **kwargs):
        super(ForexForecastingTransformer, self).__init__()
        
        d_model = kwargs.get('d_model', 32)
        nhead = kwargs.get('nhead', 4)
        ff_dim = kwargs.get('ff_dim', None)
        
        if ff_dim is None:
            ff_dim = d_model * 4
            
        num_encoder_layers = kwargs.get('num_encoder_layers', 2)
        dropout = kwargs.get('dropout', 0.3)

        ff_final = kwargs.get('ff_final', [])

        stride = seq_len // output_len
        kernel = stride + 1
        padding = ((seq_len - kernel) // stride) + 1

        self.use_only_conv_padding = True
        conv_padding = 0
        if padding % 2 == 0:
            conv_padding = padding // 2
        else:
            self.use_only_conv_padding = False
            conv_padding = padding // 2 - 1

        self.name = f'ForexForecastingTransformer_m{d_model}_h{nhead}_l{num_encoder_layers}_f{ff_dim}_d{dropout}_k{kernel}_s{stride}_p{padding}'

        if len(ff_final) > 0:
            str_ff_final = '_'.join(map(str, ff_final)) 
            self.name += f'_ff{str_ff_final}'

        self.input_projection = nn.Linear(features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.downsample = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel, stride=stride, padding=conv_padding)
        
        if len(ff_final) > 0:
            last_size = d_model
            
            self.output_projection = nn.Sequential()

            for i, fc_size in enumerate(ff_final):
                self.output_projection.add_module(f'dense_{i}', nn.Linear(last_size, fc_size))
                self.output_projection.add_module(f'relu_{i}', nn.ReLU())
                last_size = fc_size

            self.output_projection.add_module('output', nn.Linear(last_size, 1))

        else:
            self.output_projection = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        x = x.permute(0, 2, 1)
        if not self.use_only_conv_padding:
            x = F.pad(x, (0, 1))
        
        x = self.downsample(x)
        
        x = x.permute(0, 2, 1)
        
        x = self.output_projection(x)
        
        return x
