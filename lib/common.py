import sys
import time
import copy
import numpy as np
import re

import torch
import torch.nn as nn
import collections


class RewardTracker:
    def __init__(self, writer, stop_reward, group_rewards=1):
        self.writer = writer
        self.stop_reward = stop_reward
        self.reward_buf = []
        self.steps_buf = []
        self.group_rewards = group_rewards

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_steps = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward_steps, frame, epsilon=None):
        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        if len(self.reward_buf) < self.group_rewards:
            return False
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)
        self.reward_buf.clear()
        self.steps_buf.clear()
        self.total_rewards.append(reward)
        self.total_steps.append(steps)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:]).item()
        mean_steps = np.mean(self.total_steps[-100:]).item()
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards)*self.group_rewards, mean_reward, mean_steps, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("steps_100", mean_steps, frame)
        self.writer.add_scalar("steps", steps, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

class lossTracker:
    def __init__(self, writer, group_losses=1):
        self.writer = writer
        self.loss_buf = []
        self.total_loss = []
        self.steps_buf = []
        self.group_losses = group_losses
        self.capacity = group_losses*10

    def loss(self, loss, frame):
        assert (isinstance(loss, np.float))
        self.loss_buf.append(loss)
        if len(self.loss_buf) < self.group_losses:
            return False
        mean_loss = np.mean(self.loss_buf)
        self.loss_buf.clear()
        self.total_loss.append(mean_loss)
        movingAverage_loss = np.mean(self.total_loss[-100:])
        if len(self.total_loss) > self.capacity:
            self.total_loss = self.total_loss[1:]

        self.writer.add_scalar("loss_100", movingAverage_loss, frame)
        self.writer.add_scalar("loss", mean_loss, frame)

class gan_lossTracker:
    def __init__(self, writer, stop_loss=np.inf, mean_size=100):
        self.writer = writer
        self.stop_reward = stop_loss
        self.loss_buf = []
        self.steps_buf = []
        self.mean_size = mean_size

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        nameList = ['batch_loss_D_W', 'batch_loss_D_W_', 'batch_loss_D',
                    'g_loss_array', 'g_MSE_array', 'G_loss_array',
                    'val_batch_loss_D_W','val_batch_loss_D_W_','val_batch_loss_D',
                    'val_g_loss_array','val_g_MSE_array','val_G_loss_array']
        self.Buffer = {key: [] for key in nameList}

        return self

    def __exit__(self, *args):
        self.writer.close()

    def D_performance(self, D_W, D_W_, loss_D, frame):
        batch_loss_D_W = np.mean(D_W.cpu().detach().numpy(), axis=0).item()
        batch_loss_D_W_ = np.mean(D_W_.cpu().detach().numpy(), axis=0).item()
        batch_loss_D = loss_D.cpu().detach().numpy().item()

        # store the data
        self.Buffer['batch_loss_D_W'].append(batch_loss_D_W)
        self.Buffer['batch_loss_D_W_'].append(batch_loss_D_W_)
        self.Buffer['batch_loss_D'].append(batch_loss_D)

        # write the data
        self.writer.add_scalar("loss_D_W", batch_loss_D_W, frame)
        self.writer.add_scalar("loss_D_W_", batch_loss_D_W_, frame)
        self.writer.add_scalar("loss_D", batch_loss_D, frame)

    def G_performance(self, g_loss, g_MSE, G_loss, frame):
        g_loss_array = g_loss.cpu().detach().numpy().item()
        g_MSE_array = g_MSE.cpu().detach().numpy().item()
        G_loss_array = G_loss.cpu().detach().numpy().item()

        # store the data
        self.Buffer['g_loss_array'].append(g_loss_array)
        self.Buffer['g_MSE_array'].append(g_MSE_array)
        self.Buffer['G_loss_array'].append(G_loss_array)

        # write the data
        self.writer.add_scalar("g_loss", g_loss_array, frame)
        self.writer.add_scalar("g_MSE", g_MSE_array, frame)
        self.writer.add_scalar("loss_G", G_loss_array, frame)

    def D_val_performance(self, D_W, D_W_, loss_D, frame):
        val_batch_loss_D_W = np.mean(D_W.cpu().detach().numpy(), axis=0).item()
        val_batch_loss_D_W_ = np.mean(D_W_.cpu().detach().numpy(), axis=0).item()
        val_batch_loss_D = loss_D.cpu().detach().numpy().item()

        # store the data
        self.Buffer['val_batch_loss_D_W'].append(val_batch_loss_D_W)
        self.Buffer['val_batch_loss_D_W_'].append(val_batch_loss_D_W_)
        self.Buffer['val_batch_loss_D'].append(val_batch_loss_D)

        # write the data
        self.writer.add_scalar("val_loss_D_W", val_batch_loss_D_W, frame)
        self.writer.add_scalar("val_loss_D_W_", val_batch_loss_D_W_, frame)
        self.writer.add_scalar("val_loss_D", val_batch_loss_D, frame)

    def G_val_performance(self, g_loss, g_MSE, G_loss, frame):
        val_g_loss_array = g_loss.cpu().detach().numpy().item()
        val_g_MSE_array = g_MSE.cpu().detach().numpy().item()
        val_G_loss_array = G_loss.cpu().detach().numpy().item()

        # store the data
        self.Buffer['val_g_loss_array'].append(val_g_loss_array)
        self.Buffer['val_g_MSE_array'].append(val_g_MSE_array)
        self.Buffer['val_G_loss_array'].append(val_G_loss_array)

        # write the data
        self.writer.add_scalar("val_g_loss", val_g_loss_array, frame)
        self.writer.add_scalar("val_g_MSE", val_g_MSE_array, frame)
        self.writer.add_scalar("val_loss_G", val_G_loss_array, frame)

    def print_data(self, frame):
        if len(self.Buffer['batch_loss_D']) > self.mean_size:
            D_loss = np.mean(self.Buffer['batch_loss_D'][-self.mean_size:]).item()
            G_loss = np.mean(self.Buffer['G_loss_array'][-self.mean_size:]).item()
            speed = (frame - self.ts_frame) / (time.time() - self.ts)
            self.ts_frame = frame
            self.ts = time.time()
            print("%d: Train mode - D_loss: %.5f, G_loss: %.5f, speed: %.3f f/s"
                  % (frame, D_loss, G_loss, speed))
        else:
            print("%d: The number of data is not enough to shown yet." %(frame))


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = exp.state
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(exp.last_state)
    return states, np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), last_states


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = states
    next_states_v = next_states
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def calc_D_loss(D_W, D_W_, batch_size):
    loss_D = (-1/batch_size) * torch.sum(torch.log(D_W)) + (-1/batch_size) * torch.sum(torch.log(1-D_W_))
    return loss_D, D_W, D_W_

def calc_G_loss(D_W_, x_v_, k_v_, x_v, k_v, batch_size):
    w_weight_vector = torch.tensor(np.array([0.0333,0.0333,0.5000,0.3000,0.0333,0.0333,0.0333,0.0333]), dtype=torch.float32).to(torch.device("cuda"))
    G_weight_vector = torch.tensor(np.array([0.5, 0.5]), dtype=torch.float32).to(torch.device("cuda"))
    g_loss = (-1/batch_size) * torch.sum(torch.log(D_W_))
    w_ = torch.cat((x_v_, k_v_), axis=2)
    w = torch.cat((x_v, k_v), axis=2).to(torch.device("cuda"))
    g_MSE = (1/batch_size) * torch.sum(torch.sum((((w_ - w)**2) * (w_weight_vector)), axis=2), axis=0)
    G_loss = G_weight_vector[0] * g_loss + G_weight_vector[1] * g_MSE
    return G_loss, g_loss, g_MSE

def find_stepidx(text, open_str, end_str):
    regex_open = re.compile(open_str)
    regex_end = re.compile(end_str)
    matches_open = regex_open.search(text)
    matches_end = regex_end.search(text)
    return np.int(text[matches_open.span()[1]:matches_end.span()[0]])

class netPreprocessor:
    def __init__(self, net, tgt_net):
        self.net = net
        self.tgt_net = tgt_net

    def train_mode(self, batch_size):
        self.net.train()
        self.net.zero_grad()
        self.net.init_hidden(batch_size)

        self.tgt_net.eval()
        self.tgt_net.init_hidden(batch_size)

    def val_mode(self, batch_size):
        self.net.eval()
        self.net.init_hidden(batch_size)

    def populate_mode(self, batch_size):
        self.net.eval()
        self.net.init_hidden(batch_size)

class GANPreprocessor:
    def __init__(self, G_net, D_net, tgt_D_net):
        self.G = G_net
        self.D = D_net
        self.tgt_D = tgt_D_net

    def train_mode(self, batch_size):
        self.G.train()
        self.G.zero_grad()
        self.G.init_hidden(batch_size)
        self.D.train()
        self.D.zero_grad()
        self.D.init_hidden(batch_size)
        self.tgt_D.train()
        self.tgt_D.zero_grad()
        self.tgt_D.init_hidden(batch_size)

    def val_mode(self, batch_size):
        self.G.eval()
        self.G.init_hidden(batch_size)
        #self.D.eval()
        #self.D.init_hidden(batch_size)
        self.tgt_D.eval()
        self.tgt_D.init_hidden(batch_size)

def weight_visualize(net, writer):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param)

def valid_result_visualize(stats=None, writer=None, step_idx=None):
    # output the mean reward to the writer
    for key, vals in stats.items():
        if (len(stats[key]) > 0) and (np.mean(vals) != 0):
            mean_value = np.mean(vals)
            std_value = np.std(vals, ddof=1)
            writer.add_scalar(key + "_val", mean_value, step_idx)
            writer.add_scalar(key + "_std_val", std_value, step_idx)
            if (key == 'order_profits') or (key == 'episode_reward'):
                writer.add_histogram(key + "dist_val", np.array(vals))
        else:
            writer.add_scalar(key + "_val", 0, step_idx)
            writer.add_scalar(key + "_std_val", 0, step_idx)
            if (key == 'order_profits') or (key == 'episode_reward'):
                writer.add_histogram(key + "_val", 0)

    # output the reward distribution to the writer

def inputShape_check(train_set, extra_set, bars_count, required_volume=False):

    instrument = np.random.choice(list(train_set.keys()))
    extra_set_ = extra_set[instrument]
    extra_trend_size = 0
    extra_status_size = 0
    if len(extra_set_['trend']) is not 0:
        for trend_name in list(extra_set_['trend'].keys()):
            extra_trend_size += extra_set_['trend'][trend_name].encoded_size
    if len(extra_set_['status']) is not 0:
        for status_name in list(extra_set_['status'].keys()):
            extra_status_size += extra_set_['status'][status_name].encoded_size

    base_status_size = 2
    if required_volume:
        base_trend_size = 5
        # return price_shape, trend_shape, status_shape
        return (bars_count, base_trend_size ), (bars_count, extra_trend_size), (1, base_status_size + extra_status_size)
    else:
        base_trend_size = 4
        # return price_shape, trend_shape, status_shape
        return (bars_count, base_trend_size ), (bars_count, extra_trend_size), (1, base_status_size + extra_status_size)

class TargetNet:
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(self.model)

    def sync(self, D_net):
        self.target_model.load_state_dict(D_net.state_dict())