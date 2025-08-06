import torch
import torch.nn as nn


class CNN_GRU(nn.Module):
    def __init__(self, input_size, cnn_out_channels, gru_hidden_size, num_layers, dropout, kernel_size, output_size=1):

        super(CNN_GRU, self).__init__()
        self.input_size = input_size
        self.cnn_out_channels = cnn_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.kernel_size = kernel_size


        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.conv_activation = nn.ReLU()

        self.pooling = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)


        if input_size != cnn_out_channels:
            self.residual_conv = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=1)
        else:
            self.residual_conv = None


        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=gru_hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)


        self.fc = nn.Linear(gru_hidden_size, output_size)

    def forward(self, x):

        x = x.permute(0, 2, 1)  

        identity = x
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)


        x = self.conv_activation(self.conv1(x))  
        x = self.pooling(x)  
        x = self.conv_activation(self.conv2(x))  
        x = self.pooling(x) 

        x = x + identity

        x = x.permute(0, 2, 1)  # [batch_size, seq_length, cnn_out_channels]

        h0 = torch.zeros(self.num_layers, x.size(0), self.gru_hidden_size).to(x.device)
        out, _ = self.gru(x, h0)  # [batch_size, seq_length, gru_hidden_size]


        out = out[:, -1, :]  # [batch_size, gru_hidden_size]


        out = self.fc(out)  # [batch_size, output_size]

        return out

