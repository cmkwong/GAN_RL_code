import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class G_net(nn.Module):
    def __init__(self, price_input_size=4, trend_input_size=4, n_hidden=256, n_layers=2, rnn_drop_prob=0.1,
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
        if self.train_on_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.batch_size = None

        self.price_lstm_G = nn.LSTM(self.price_input_size, self.n_hidden, self.n_layers, dropout=self.rnn_drop_prob,
                                    batch_first=self.batch_first).to(self.device)

        self.trend_lstm_G = nn.LSTM(self.trend_input_size, self.n_hidden, self.n_layers, dropout=self.rnn_drop_prob,
                                    batch_first=self.batch_first).to(self.device)

        self.fc_price_G = nn.Sequential(
            nn.Linear(self.n_hidden, 512),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 512),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 256),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(256, 128),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, self.price_input_size),
        ).to(self.device)

        self.fc_trend_G = nn.Sequential(
            nn.Linear(self.n_hidden, 512),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 512),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 256),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(256, 128),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, self.trend_input_size),
        ).to(self.device)

        self.hidden_p_G = None
        self.hidden_t_G = None

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
        price_output_ = price_output[:, -1, :]
        trend_output_ = trend_output[:, -1, :]
        price_output_ = price_output_.view(self.batch_size, -1)
        trend_output_ = trend_output_.view(self.batch_size, -1)

        # into fc
        x_v_ = self.fc_price_G(price_output_).unsqueeze(1)
        k_v_ = self.fc_trend_G(trend_output_).unsqueeze(1)

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
        else:
            weight = next(self.parameters()).data
            self.hidden_p_G = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
            weight = next(self.parameters()).data
            self.hidden_t_G = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                           weight.new_zeros(self.n_layers, batch_size, self.n_hidden))

class D_net(nn.Module):
    def __init__(self, price_input_size=4, trend_input_size=4, n_hidden=256, n_layers=1,
                 fc_drop_prob=0.1, train_on_gpu=True, batch_first=True):
        super(D_net, self).__init__()
        self.input_size = price_input_size + trend_input_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.fc_drop_prob = fc_drop_prob
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
            nn.Linear(self.n_hidden, 512),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.0001),
            nn.Linear(32, 1),
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

        # last output from lstms
        D_output_ = D_output[:, -1, :]
        D_output_ = D_output_.view(self.batch_size, -1)

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





