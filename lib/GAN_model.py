import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class G_net(nn.Module):
    def __init__(self, price_input_size=4, trend_input_size=4, n_hidden=32, n_layers=2, rnn_drop_prob=0.1,
                 fc_drop_prob=0.1, train_on_gpu=True, batch_first=True):
        super(G_net, self).__init__()
        self.price_input_size = price_input_size
        self.trend_input_size = trend_input_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.rnn_drop_prob = rnn_drop_prob
        self.fc_drop_prob = fc_drop_prob
        self.train_on_gpu = train_on_gpu
        self.batch_first = batch_first
        self.kernel_sizes = [2,4,8,16,32]

        self.combine_hidden_size = 64
        if self.train_on_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.batch_size = None

        self.price_lstm_G = nn.LSTM(self.price_input_size, self.n_hidden, self.n_layers, dropout=self.rnn_drop_prob,
                                    batch_first=self.batch_first).to(self.device)

        self.trend_lstm_G = nn.LSTM(self.trend_input_size, self.n_hidden, self.n_layers, dropout=self.rnn_drop_prob,
                                    batch_first=self.batch_first).to(self.device)

        self.combine_lstm = nn.LSTM((self.n_hidden*2), self.combine_hidden_size, self.n_layers, dropout=self.rnn_drop_prob,
                                    batch_first=self.batch_first).to(self.device)

        # conv 1
        self.convs_1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=h),
                          nn.BatchNorm1d(num_features=32))
            for h in self.kernel_sizes
        ])
        # nn.ReLU(),
        self.maxPool_1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        # conv 2
        self.convs_2 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=32,
                                    out_channels=16,
                                    kernel_size=h),
                          nn.BatchNorm1d(num_features=16))
            for h in self.kernel_sizes
        ])
        self.maxPool_2 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        # conv 3
        self.convs_3 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=16,
                                    out_channels=8,
                                    kernel_size=h),
                          nn.BatchNorm1d(num_features=8))
            for h in self.kernel_sizes
        ])
        self.maxPool_3 = nn.MaxPool1d(kernel_size=4, stride=4, padding=1)

        self.fc_1 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.fc_2 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.fc_3 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.fc_4 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.fc_5 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.fc_6 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.fc_7 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.fc_8 = nn.Sequential(
            nn.Linear(62, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
        ).to(self.device)

        self.hidden_p_G = None
        self.hidden_t_G = None
        self.hidden_c = None

    def preprocessor(self, X_v, K_v):
        X_v_tensor = X_v.to(self.device)
        K_v_tensor = K_v.to(self.device)
        return X_v_tensor, K_v_tensor

    def forward(self, X_v, K_v):
        X_v_tensor, K_v_tensor = self.preprocessor(X_v, K_v)

        # price lstm
        self.hidden_p_G = tuple([each.data for each in self.hidden_p_G])
        self.price_lstm_G.flatten_parameters()
        price_output, self.hidden_p_G = self.price_lstm_G(X_v_tensor, self.hidden_p_G)

        # trend lstm
        self.hidden_t_G = tuple([each.data for each in self.hidden_t_G])
        self.trend_lstm_G.flatten_parameters()
        trend_output, self.hidden_t_G = self.trend_lstm_G(K_v_tensor, self.hidden_t_G)

        # last output from lstms
        combine_input = torch.cat((price_output, trend_output), dim=2)
        # combine lstm
        self.hidden_c = tuple([each.data for each in self.hidden_c])
        self.combine_lstm.flatten_parameters()
        combine_output, self.hidden_c = self.combine_lstm(combine_input, self.hidden_c)

        # premute the dim
        combine_output_ = combine_output.permute(0,2,1)
        conv_outputs = [conv(combine_output_) for conv in self.convs_1]
        out_cat = None
        for c, conv_output in enumerate(conv_outputs):
            if c == 0:
                out_cat = conv_output
            else:
                out_cat = torch.cat((out_cat, conv_output), dim=2)
        out_cat_pool_1 = self.maxPool_1(out_cat)

        conv_outputs = [conv(out_cat_pool_1) for conv in self.convs_2]
        out_cat = None
        for c, conv_output in enumerate(conv_outputs):
            if c == 0:
                out_cat = conv_output
            else:
                out_cat = torch.cat((out_cat, conv_output), dim=2)
        out_cat_pool_2 = self.maxPool_2(out_cat)

        conv_outputs = [conv(out_cat_pool_2) for conv in self.convs_3]
        out_cat = None
        for c, conv_output in enumerate(conv_outputs):
            if c == 0:
                out_cat = conv_output
            else:
                out_cat = torch.cat((out_cat, conv_output), dim=2)
        out_cat_pool_3 = self.maxPool_3(out_cat)

        # into fc
        x_v_1 = self.fc_1(out_cat_pool_3[:, 0,:]).unsqueeze(1)
        x_v_2 = self.fc_2(out_cat_pool_3[:, 1,:]).unsqueeze(1)
        x_v_3 = self.fc_3(out_cat_pool_3[:, 2,:]).unsqueeze(1)
        x_v_4 = self.fc_4(out_cat_pool_3[:, 3,:]).unsqueeze(1)
        k_v_1 = self.fc_5(out_cat_pool_3[:, 4,:]).unsqueeze(1)
        k_v_2 = self.fc_6(out_cat_pool_3[:, 5,:]).unsqueeze(1)
        k_v_3 = self.fc_7(out_cat_pool_3[:, 6,:]).unsqueeze(1)
        k_v_4 = self.fc_8(out_cat_pool_3[:, 7,:]).unsqueeze(1)

        x_v_ = torch.cat((x_v_1, x_v_2, x_v_3, x_v_4), dim=2)
        k_v_ = torch.cat((k_v_1, k_v_2, k_v_3, k_v_4), dim=2)

        return x_v_, k_v_

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        if (self.train_on_gpu):
            weight = next(self.parameters()).data # first lstm
            self.hidden_p_G = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(),
                             weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda())
            weight = next(self.parameters()).data # second lstm
            self.hidden_t_G = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(),
                             weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda())
            weight = next(self.parameters()).data  # third lstm
            self.hidden_c = (weight.new_zeros(self.n_layers, batch_size, self.combine_hidden_size).cuda(),
                               weight.new_zeros(self.n_layers, batch_size, self.combine_hidden_size).cuda())
        else:
            weight = next(self.parameters()).data
            self.hidden_p_G = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
            weight = next(self.parameters()).data
            self.hidden_t_G = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                           weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
            weight = next(self.parameters()).data  # third lstm
            self.hidden_c = (weight.new_zeros(self.n_layers, batch_size, self.combine_hidden_size),
                               weight.new_zeros(self.n_layers, batch_size, self.combine_hidden_size))

class D_net(nn.Module):
    def __init__(self, price_input_size=4, trend_input_size=4, n_hidden=64, n_layers=1,
                 fc_drop_prob=0.1, bars_count=40, train_on_gpu=True, batch_first=True):
        super(D_net, self).__init__()
        self.input_size = price_input_size + trend_input_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.fc_drop_prob = fc_drop_prob
        self.bars_count = bars_count
        self.train_on_gpu = train_on_gpu
        self.batch_first = batch_first
        if self.train_on_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.batch_size = None

        self.lstm_D = nn.LSTM(self.input_size, self.n_hidden, num_layers=self.n_layers,
                              batch_first=self.batch_first).to(self.device)

        self.fc_D = nn.Sequential(
            nn.Linear((self.n_hidden * (self.bars_count + 1)), 2046),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(2046, 2046),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(2046, 1024),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.hidden_D = None

    def preprocessor(self, W_W):
        W_W_tesnor = W_W.to(self.device)
        return W_W_tesnor

    def forward(self, W_W):
        W_W_tesnor = self.preprocessor(W_W)

        # discriminator
        self.hidden_D = tuple([each.data for each in self.hidden_D])
        self.lstm_D.flatten_parameters()
        D_output, self.hidden_D = self.lstm_D(W_W_tesnor, self.hidden_D)

        # all output from lstm and cat them
        #D_output_ = D_output[:, -1, :]
        D_output_ = D_output.contiguous().view(self.batch_size, -1)

        # into fc
        D_W_W = self.fc_D(D_output_)

        return D_W_W

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        if (self.train_on_gpu):
            weight = next(self.parameters()).data # first lstm
            self.hidden_D = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(),
                             weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda())
        else:
            weight = next(self.parameters()).data
            self.hidden_D = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden))





