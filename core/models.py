import torch
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
    
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))  # Addestrabile

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
class ConvLSTMBlock(nn.Module):
    def __init__(self, input_channels, conv1_out_channels, conv2_out_channels, kernel_size_1, kernel_size_2, lstm1_hidden_size, lstm1_bidirectional, lstm2_hidden_size, dropout):
        super(ConvLSTMBlock, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, conv1_out_channels, kernel_size_1, padding="same"),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(output_size=conv1_out_channels),
            nn.Conv1d(conv1_out_channels, conv2_out_channels, kernel_size_2, padding="same"),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(output_size=conv2_out_channels)
        )
        
        self.lstm1 = nn.LSTM(conv2_out_channels, lstm1_hidden_size, batch_first=True, bidirectional=lstm1_bidirectional)
        lstm_output_size = lstm1_hidden_size * (2 if lstm1_bidirectional else 1)

        self.norm1 = nn.LayerNorm(lstm_output_size)
        self.lstm2 = nn.LSTM(lstm_output_size, lstm2_hidden_size, batch_first=True, bidirectional=False)
        self.norm2 = nn.LayerNorm(lstm2_hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.norm2(x)
        
        return x

class ConvLSMTClassifier(nn.Module):
    def __init__(self, features, **kwargs):
        super(ConvLSMTClassifier, self).__init__()

        branches = kwargs.get('branches', [
            {
                "conv1_out_channels": 32,
                "kernel_size_1": 3,
                "conv2_out_channels": 64,
                "kernel_size_2": 2,
                "lstm1_hidden_size": 64,
                "lstm1_bidirectional": False,
                "lstm2_hidden_size": 64,
                "dropout": 0.3
            },
            {
                "conv1_out_channels": 32,
                "kernel_size_1": 5,
                "conv2_out_channels": 64,
                "kernel_size_2": 3,
                "lstm1_hidden_size": 64,
                "lstm1_bidirectional": False,
                "lstm2_hidden_size": 64,
                "dropout": 0.3
            },
            {
                "conv1_out_channels": 32,
                "kernel_size_1": 11,
                "conv2_out_channels": 64,
                "kernel_size_2": 7,
                "lstm1_hidden_size": 64,
                "lstm1_bidirectional": False,
                "lstm2_hidden_size": 64,
                "dropout": 0.3
            },
        ])

        hidden_layers = kwargs.get('hidden_layers', [256, 128, 64, 32])
        hidden_layers_dropout = kwargs.get('hidden_layers_dropout', 0.3)
        use_adderstrable_swish = kwargs.get('addestrable_swish', False)

        self.name = 'clm'
        branch_strings = '__'.join([f"{b['conv1_out_channels']}_{b['kernel_size_1']}_{b['conv2_out_channels']}_{b['kernel_size_2']}_{('bi_' if b["lstm1_bidirectional"] else '')}{b['lstm1_hidden_size']}_{b['lstm2_hidden_size']}_{b['dropout']}" for b in branches])
        self.name += f'_b{branch_strings}'
        hidden_layers_string = '_'.join([str(f) for f in hidden_layers])
        self.name += f'_fc{hidden_layers_string}'
        if use_adderstrable_swish:
            self.name += '_aswish'
        else:
            self.name += '_swish'

        concat_size = 0
        self.branches = nn.ModuleList()

        for branch_spec in branches:
            branch = ConvLSTMBlock(
                input_channels=features, 
                conv1_out_channels=branch_spec['conv1_out_channels'], 
                conv2_out_channels=branch_spec['conv2_out_channels'], 
                kernel_size_1=branch_spec['kernel_size_1'], 
                kernel_size_2=branch_spec['kernel_size_2'], 
                lstm1_hidden_size=branch_spec['lstm1_hidden_size'], 
                lstm1_bidirectional=branch_spec['lstm1_bidirectional'],
                lstm2_hidden_size=branch_spec['lstm2_hidden_size'], 
                dropout=branch_spec['dropout']
            )
            self.branches.append(branch)
            concat_size += branch_spec['lstm2_hidden_size']

        self.hidden_layers = self._create_hidden_layers(concat_size, hidden_layers, hidden_layers_dropout, use_adderstrable_swish)

        self.output = nn.Linear(hidden_layers[-1], 1)

    def _create_hidden_layers(self, input_size, hidden_layers, dropout, use_adderstrable_swish):
        layers = []
        for i, hidden_layer in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_layer))
            if use_adderstrable_swish:
                layers.append(Swish())
            else:
                layers.append(nn.SiLU())
            if i < len(hidden_layers) - 1 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_layer

        return nn.Sequential(*layers)

    def forward(self, input_1):
        branch_outputs = []

        for branch in self.branches:
            branch_output = branch(input_1)
            branch_outputs.append(branch_output[:, -1, :])

        branch_output_2d = torch.cat(branch_outputs, dim=1)

        x = self.hidden_layers(branch_output_2d)

        x = self.output(x)

        return x