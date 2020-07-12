import torch
from torch import nn
import torch.nn.functional as F

class TransModeCNN(nn.Module):
    def __init__(self):
        super(TransModeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 1), stride=1)
        self.pool1 = nn.MaxPool2d((2, 1), (2, 1))

        #self.norm = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, (3, 1), stride=1)
        self.pool2 = nn.MaxPool2d((2, 1), (2, 1))

        self.fc1 = nn.Linear(64 * 9 * 17, 256)
        self.fc2 = nn.Linear(256, 128)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = self.pool1(x)
        #x = self.norm(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 64 * 9 * 17)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x

class CNNwithLSTM(nn.Module):
    def __init__(self):
        super(CNNwithLSTM, self).__init__()
        self.cnn = TransModeCNN()
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            dropout=0.2,
            batch_first=True)
        self.linear = nn.Linear(64, 9)

    def forward(self, x):
        batch_size, timesteps, H, W = x.size()
        c_in = x.view(batch_size * timesteps, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        #r_out2 = self.linear(r_out[:, -1, :])

        outs = []
        for point in r_out:
            outs.append(self.linear(point))
        return torch.stack(outs, dim=0)

