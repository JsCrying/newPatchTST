
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        self.embed_size = 128 #embed_size
        self.hidden_size = 256 #hidden_size
        self.seq_length = configs.seq_len
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        if self.configs.debug:
            print(f'IN FreTS.py')
            print(f'IN: {x.shape = }')
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        if self.configs.debug:
            print(f'after tokenEmb: {x.shape = }')
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        if self.configs.debug:
            print(f'after MLP_temporal: {x.shape = }')
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        if self.configs.debug:
            print(f'after fc: {x.shape = }')
        return x



# # Formers
# parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
# parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# parser.add_argument('--factor', type=int, default=1, help='attn factor')
# parser.add_argument('--distil', action='store_false',
#                     help='whether to use distilling in encoder, using this argument means not using distilling',
#                     default=True)
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# parser.add_argument('--embed', type=str, default='timeF',
#                     help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# # optimization
# parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
# parser.add_argument('--itr', type=int, default=1, help='experiments times')
# parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
# parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
# parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
# parser.add_argument('--des', type=str, default='Exp', help='exp description')
# parser.add_argument('--loss', type=str, default='mse', help='loss function')
# parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
