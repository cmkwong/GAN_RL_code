import os
from datetime import datetime
from gym import wrappers
import ptan
import numpy as np
import torch
import torch.optim as optim
from lib import environ, data, models, common, validation, GAN_model
from torch.utils.tensorboard import SummaryWriter

G_lr = 0.00001
D_lr = 0.0000001
BATCH_SIZE = 32
VAL_STEPS = 10000
CHECKPOINT_EVERY_STEP = 20000
TARGET_NET_SYNC = 1000 # 1000
BARS_COUNT = 40
PRINT_STEP = 200

load_fileName = "checkpoint_GAN-200000.data"
saves_path = "../checkpoint/16"

LOAD_NET = False
TRAIN_ON_GPU = True
if TRAIN_ON_GPU:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# get the now time
now = datetime.now()
dt_string = now.strftime("%y%m%d_%H%M%S")

# read raw data
train_set, val_set, extra_set = data.read_bundle_csv(
        path="../data/16",
        sep='\t', filter_data=True, fix_open_price=False, percentage=0.8, extra_indicator=True,
        trend_names=['bollinger_bands', 'MACD', 'RSI'], status_names=[])

# define the shape producer for convenience
price_shape, trend_shape, status_shape = common.inputShape_check(train_set, extra_set, BARS_COUNT, required_volume=False)
shapeList_ = [price_shape, trend_shape, status_shape]

# define the network
G_net = GAN_model.G_net(price_input_size=price_shape[1], trend_input_size=trend_shape[1], n_hidden=32, n_layers=2,
                        rnn_drop_prob=0.1, fc_drop_prob=0.1, train_on_gpu=TRAIN_ON_GPU, batch_first=True).to(device)

D_net = GAN_model.D_net(price_input_size=price_shape[1], trend_input_size=trend_shape[1], n_hidden=64, n_layers=1,
                        fc_drop_prob=0.1, bars_count=BARS_COUNT, train_on_gpu=TRAIN_ON_GPU, batch_first=True).to(device)

# define the optimizers
optimizerG = optim.Adam(params=G_net.parameters(), lr=G_lr, betas=(0.9,0.999))
optimizerD = optim.Adam(params=D_net.parameters(), lr=D_lr, betas=(0.9,0.999))

# define the data container for both training set and validation set
train_container = data.gan_data_container(train_set, extra_set, shapeList_, train_mode=True, required_volume=False)
val_container = data.gan_data_container(val_set, extra_set, shapeList_, train_mode=False, required_volume=False)

# load the network
if LOAD_NET is True:
    with open(os.path.join(saves_path, load_fileName), "rb") as f:
        checkpoint = torch.load(f)
    G_net = GAN_model.G_net(price_input_size=price_shape[1], trend_input_size=trend_shape[1], n_hidden=32, n_layers=2,
                            rnn_drop_prob=0.1, fc_drop_prob=0.1, train_on_gpu=TRAIN_ON_GPU, batch_first=True).to(device)

    D_net = GAN_model.D_net(price_input_size=price_shape[1], trend_input_size=trend_shape[1], n_hidden=64, n_layers=1,
                            fc_drop_prob=0.1, bars_count=BARS_COUNT, train_on_gpu=TRAIN_ON_GPU, batch_first=True).to(device)
    G_net.load_state_dict(checkpoint['G_state_dict'])
    D_net.load_state_dict(checkpoint['D_state_dict'])

# update the step_idx
if LOAD_NET:
    step_idx = common.find_stepidx(load_fileName, "-", "\.")
else:
    step_idx = 0

# create the target net (stable)
tgt_D_net = common.TargetNet(D_net)

# define the net_processor
net_processor = common.GANPreprocessor(G_net, D_net, tgt_D_net.target_model)

# define the writer
writer = SummaryWriter(log_dir="../runs/GAN/" + dt_string, comment="GAN_stock_trading")

with common.gan_lossTracker(writer, stop_loss=np.inf, mean_size=1000) as loss_tracker:
    while True:
        step_idx += 1
        net_processor.train_mode(batch_size=BATCH_SIZE)
        # generate the training set
        X_v, K_v, x_v, k_v = train_container.generate_batch(BATCH_SIZE)
        input_real = data.D_preprocess(X_v, K_v, x_v, k_v)

        # train D by input_real
        D_W = D_net(input_real)

        # train D for input_fake
        optimizerD.zero_grad()
        x_v_, k_v_ = G_net(X_v, K_v)
        input_fake = data.D_preprocess(X_v, K_v, x_v_, k_v_)
        D_W_ = D_net(input_fake.detach())
        loss_D, Loss_D_W, Loss_D_W_ = common.calc_D_loss(D_W, D_W_, BATCH_SIZE)
        loss_tracker.D_performance(Loss_D_W, Loss_D_W_, loss_D, step_idx)
        #D_W_ = tgt_D_net.target_model(input_fake)
        loss_D.backward()
        optimizerD.step()

        # train G
        optimizerG.zero_grad()
        #D_W_ = tgt_D_net.target_model(input_fake)
        D_W_ = D_net(input_fake)
        lossG, g_loss, g_MSE = common.calc_G_loss(D_W_, x_v_, k_v_, x_v, k_v, BATCH_SIZE)
        loss_tracker.G_performance(g_loss, g_MSE, lossG, step_idx)
        lossG.backward()
        optimizerG.step()

        #if step_idx % TARGET_NET_SYNC == 0:
        #    tgt_D_net.sync(D_net)

        if step_idx % PRINT_STEP == 0:
            loss_tracker.print_data(step_idx)

        if step_idx % CHECKPOINT_EVERY_STEP == 0:
            # idx = step_idx // CHECKPOINT_EVERY_STEP
            checkpoint = {
                "G_state_dict": G_net.state_dict(),
                "D_state_dict": D_net.state_dict()
            }
            with open(os.path.join(saves_path, "checkpoint_GAN-%d.data" % step_idx), "wb") as f:
                torch.save(checkpoint, f)

        if step_idx % VAL_STEPS == 0:
            net_processor.val_mode(batch_size=BATCH_SIZE)
            X_v, K_v, x_v, k_v = val_container.generate_batch(BATCH_SIZE)
            input_real = data.D_preprocess(X_v, K_v, x_v, k_v)
            #D_W = tgt_D_net.target_model(input_real)
            D_W = D_net(input_real)

            # gen fake input
            x_v_, k_v_ = G_net(X_v, K_v)
            input_fake = data.D_preprocess(X_v, K_v, x_v_, k_v_)
            #D_W_ = tgt_D_net.target_model(input_fake)
            D_W_ = D_net(input_fake)

            # calculate the validation loss of G and D
            loss_D_val, Loss_D_W_val, Loss_D_W__val = common.calc_D_loss(D_W, D_W_, BATCH_SIZE)
            loss_tracker.D_val_performance(Loss_D_W_val, Loss_D_W__val, loss_D_val, step_idx)
            loss_G_val, g_loss_val, g_MSE_val= common.calc_G_loss(D_W_, x_v_, k_v_, x_v, k_v, BATCH_SIZE)
            loss_tracker.G_val_performance(g_loss_val, g_MSE_val, loss_G_val, step_idx)


