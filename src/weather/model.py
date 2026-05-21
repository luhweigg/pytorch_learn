import torch
import torch.nn as nn

class WeatherCNNLSTM(nn.Module):
    """
    A hybrid neural network architecture combining CNN and LSTM layers.
    The Conv1d layer extracts local, short-term patterns from the input features.
    The LSTM layer captures long-term temporal dependencies from the CNN outputs.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Note: PyTorch's Conv1d expects input in the shape (batch_size, channels, sequence_length).
        Since our data is (batch_size, sequence_length, features), we use .permute() to swap dimensions 
        before the convolution, and swap them back for the LSTM.
        Only the last time step of the LSTM output (out[:, -1, :]) is passed to the fully connected layer.
        """
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out