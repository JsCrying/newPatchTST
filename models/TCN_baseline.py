import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.input_size = configs.seq_len
        self.output_size = configs.pred_len
        self.num_channels = [configs.nhid]*configs.levels
        self.kernel_size = configs.kernel_size
        self.dropout = configs.dropout

        self.tcn = TemporalConvNet(self.input_size, self.num_channels, kernel_size=self.kernel_size, dropout=self.dropout)
        self.linear = nn.Linear(self.num_channels[-1], self.output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        if self.configs.debug:
            print('IN TCN.py')
            print(f'IN: {x.shape = }')
        y1 = self.tcn(x)
        if self.configs.debug:
            print(f'after tcn: {y1.shape = }')
        # y1 = self.linear(y1[:, :, -1])
        y1 = self.linear(y1.permute(0, 2, 1)).permute(0, 2, 1)
        if self.configs.debug:
            print(f'after linear: {y1.shape = }')
        return y1

# parser.add_argument('--batch_size', type=int, default=32, metavar='N',
#                     help='batch size (default: 32)')
# parser.add_argument('--cuda', action='store_false',
#                     help='use CUDA (default: True)')
# parser.add_argument('--dropout', type=float, default=0.0,
#                     help='dropout applied to layers (default: 0.0)')
# parser.add_argument('--clip', type=float, default=-1,
#                     help='gradient clip, -1 means no clip (default: -1)')
# parser.add_argument('--epochs', type=int, default=10,
#                     help='upper epoch limit (default: 10)')
# parser.add_argument('--ksize', type=int, default=7,
#                     help='kernel size (default: 7)')
# parser.add_argument('--levels', type=int, default=8,
#                     help='# of levels (default: 8)')
# parser.add_argument('--seq_len', type=int, default=400,
#                     help='sequence length (default: 400)')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                     help='report interval (default: 100')
# parser.add_argument('--lr', type=float, default=4e-3,
#                     help='initial learning rate (default: 4e-3)')
# parser.add_argument('--optim', type=str, default='Adam',
#                     help='optimizer to use (default: Adam)')
# parser.add_argument('--nhid', type=int, default=30,
#                     help='number of hidden units per layer (default: 30)')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed (default: 1111)')