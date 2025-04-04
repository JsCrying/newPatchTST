import torch
import torch.nn as nn
from models import DLinear, PatchTST_real, UMixer, TCN, FreTS
import copy
class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, input_size, configs):
        super(SimpleRNN, self).__init__()

        def choose_model(configs):
            rnn_base_model = configs.rnn_base_model
            model_dict = {
                'UMixer': UMixer,
                'DLinear': DLinear,
                'PatchTST_real': PatchTST_real,
                'TCN': TCN,
                'FreTS': FreTS
            }
            model = model_dict[rnn_base_model]
            return model

        def construct_configs(configs):
            if configs.rnn_base_model == 'RLinear':
                configs.channel = 1
                configs.drop = 0.1
                configs.rev = configs.revin

            return configs
        def construct_configs_x(configs):
            configs = copy.deepcopy(construct_configs(configs))
            configs.seq_len = input_size
            configs.pred_len = hidden_size


            return configs
        def construct_configs_h(configs):
            configs = copy.deepcopy(construct_configs(configs))
            configs.seq_len = hidden_size
            configs.pred_len = hidden_size

            return configs

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.configs = copy.deepcopy(configs)
        self.model = choose_model(self.configs)
        # self.Wx = nn.Parameter(torch.randn(input_size, hidden_size))
        # self.Wh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(hidden_size))
        self.Wx = self.model.Model(construct_configs_x(self.configs))
        self.Wh = self.model.Model(construct_configs_h(self.configs))
        # self.Wx = nn.Linear(input_size, hidden_size)
        # self.Wh = nn.Linear(hidden_size, hidden_size)
        # configs.seq_len = hidden_size
        # configs.pred_len = hidden_size
        # configs.enc_in = 1
        # configs.individual = 0
        # self.Wx = DLinear.Model(configs)
        # self.Wh = DLinear.Model(configs)
        # self.Wx = DLinear.Model(configs2={"seq_len":input_size, "pred_len":hidden_size, "individual":0, "enc_in":1})
        # self.Wh = DLinear.Model(configs2={"seq_len":hidden_size, "pred_len":hidden_size, "individual":0, "enc_in":1})
        # self.Wx = RLinear.Model(configs2={"seq_len":input_size, "pred_len":hidden_size, "individual":0, "channel":1, "drop":0.1, "rev":configs.revin})
        # self.Wh = RLinear.Model(configs2={"seq_len":hidden_size, "pred_len":hidden_size, "individual":0, "channel":1, "drop":0.1, "rev":configs.revin})

        # self.Wx = PatchTST_real.Model(configs, configs2={"seq_len":input_size, "pred_len":hidden_size, "individual":0, "enc_in":1})
        # self.Wh = PatchTST_real.Model(configs, configs2={"seq_len":hidden_size, "pred_len":hidden_size, "individual":0, "enc_in":1})
        # self.Wx = iTransformer.Model(configs, configs2={"seq_len":input_size, "pred_len":hidden_size, "individual":0, "enc_in":1})
        # self.Wh = iTransformer.Model(configs, configs2={"seq_len":hidden_size, "pred_len":hidden_size, "individual":0, "enc_in":1})

        # self.Wx = NLinear.Model(configs={"seq_len":input_size, "pred_len":hidden_size, "individual":0, "enc_in":1})
        # self.Wh = NLinear.Model(configs={"seq_len":hidden_size, "pred_len":hidden_size, "individual":0, "enc_in":1})


    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # x: [batch_size*channel, num_patches, input_size]
        if self.configs.debug:
            print('IN rnn.py')
        rnn_out = []
        if self.configs.rnn_base_model in ['DLinear', 'PatchTST_real', 'TCN', 'FreTS']:
            batch_size, seq_len, _ = batch_x.shape
            # h = torch.zeros(batch_size, self.hidden_size).to(x.device)
            h = torch.zeros(batch_size, 1, self.hidden_size).to(batch_x.device)
            all_h = []
            for t in range(seq_len):
                batch_x_t = batch_x[:, t, :].unsqueeze(1)
                # print("x_shape:",x_t.shape)
                # h = torch.tanh(x_t @ self.Wx + h @ self.Wh + self.b)
                if self.configs.debug:
                    print(f'[{t}]before tanh: {h.shape = }') # h.shape = torch.Size([896, 1, 512])
                h = torch.tanh(self.Wx(batch_x_t) + self.Wh(h) + self.bh) # output should be 3-dim!!!
                if self.configs.debug:
                    print(f'[{t}]after tanh: {h.shape = }') # h.shape = torch.Size([896, 1, 512])
                all_h.append(h)
                # all_h.append(h.unsqueeze(1)) [b, 1, 1, h]
                if self.configs.debug:
                    print(f'[{t}]after put in all_h: {h.shape = }') # h.shape = torch.Size([896, 1, 512])
            rnn_out = torch.cat(all_h, dim=1)
        elif self.configs.rnn_base_model in ['UMixer']:
            if self.configs.debug:
                print(f'{batch_x.shape[1] = }')
            x_h = torch.zeros(batch_x.shape[0], 1, self.hidden_size).to(batch_x.device)
            x_mark_h = torch.zeros(batch_x_mark.shape[0], 1, self.hidden_size).to(batch_x.device)

            all_h = []

            for t in range(batch_x.shape[1]):
                batch_x_t = batch_x[:, t, :].unsqueeze(1)
                batch_x_mark_t = batch_x_mark[:, t, :].unsqueeze(1)
                dec_inp_t = dec_inp[:, t, :].unsqueeze(1)
                batch_y_mark_t = batch_y_mark[:, t, :].unsqueeze(1)
                # print("x_shape:",x_t.shape)
                # h = torch.tanh(x_t @ self.Wx + h @ self.Wh + self.b)
                x_h = torch.tanh(self.Wx(batch_x_t, batch_x_mark_t, dec_inp_t, batch_y_mark_t) + \
                    self.Wh(x_h, x_mark_h, dec_inp_t, batch_y_mark_t) +
                    self.bh)
                all_h.append(x_h.unsqueeze(1))
                x_h = x_h.unsqueeze(1)
            rnn_out = torch.cat(all_h, dim=1)

        if self.configs.debug:
            print(f'{rnn_out.shape = }')
        return rnn_out