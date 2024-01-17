import json
import os
import sys
import time
import random
import argparse
from os import path as osp
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/test')

from utils import *
from network_model import TFBlockTranformer
from data_load import *
from criterion import *
from config import *

def get_model(mode, **kwargs):
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')

    global_input_channel = kwargs.get('input_channel')
    global_window_size = kwargs.get('window_size')
    global_device = kwargs.get('device')
    global_model_type = kwargs.get('model_type', None)

    if global_model_type == None:
        raise ValueError("Model type required")

    if mode == 'train':
        if global_model_type == "tf-brt":
            print("TF Block Recurrent Transformer model")
            network = TFBlockTranformer(out_dim = 2, input_dim = global_input_channel, encoder_dim = 128, state_dim = 128,
                                       head_dim = 128, state_length = global_window_size, num_attention_heads = 8,
                                       conv_kernel_size = 21, num_encoder_layers = 1, time_length = global_window_size).to(global_device)
    elif mode == 'test':
        if global_model_type == "tf-brt":
            print("TF Block Recurrent Transformer model")
            network = TFBlockTranformer(out_dim = 2, input_dim = global_input_channel, encoder_dim = 128, state_dim = 128,
                                       head_dim = 128, state_length = global_window_size, num_attention_heads = 8,
                                       conv_kernel_size = 21, num_encoder_layers = 1, time_length = global_window_size).to(global_device)

    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    return network

def train(**kwargs):
    # load config
    global_data_dir = kwargs.get('data_dir')
    global_val_data_dir = kwargs.get('val_data_dir')
    global_batch_size = kwargs.get('batch_size')
    global_window_size = kwargs.get('window_size')
    global_sliding_size = kwargs.get('sliding_size')
    global_epochs = kwargs.get('epochs')
    global_num_workers = kwargs.get('num_workers')
    global_device = kwargs.get('device')
    global_out_dir = kwargs.get('out_dir', None)
    global_learning_rate = kwargs.get('learning_rate')
    global_save_interval = kwargs.get('save_interval')
    global_log = kwargs.get('log')
    # Loading data
    start_t = time.time()
    train_dataset = get_train_dataset(global_data_dir, **kwargs)
    val_dataset = get_valid_dataset(global_val_data_dir, **kwargs)
    train_loader = DataLoader(train_dataset, batch_size = global_batch_size, num_workers = global_num_workers, shuffle = True,
                              drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size = global_batch_size, shuffle = True, drop_last = True)
    end_t = time.time()

    global_log('Training and validation set loaded. Time usage: {:.3f}s'.format(end_t - start_t))

    global device
    device = torch.device(global_device if torch.cuda.is_available() else 'cpu')
    global_log("Device: {}".format(device))

    if global_out_dir:
        if not osp.isdir(global_out_dir):
            os.makedirs(global_out_dir)
        if not osp.isdir(osp.join(global_out_dir, 'checkpoints')):
            os.makedirs(osp.join(global_out_dir, 'checkpoints'))

    global_log('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    if val_dataset:
        global_log('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_model('train', **kwargs).to(device)
    criterion = CovWeighting()

    optimizer = torch.optim.Adam(network.parameters(), global_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 10, factor = 0.75, verbose = True, eps = 1e-12)
    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', True)

    start_epoch = 0
    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(global_epochs)

    global_log("Starting from epoch {}".format(start_epoch))
    try:
        for epoch in range(start_epoch, global_epochs):
            log_line = ''
            network.train()
            train_vel = MSEAverage()
            train_loss = 0
            start_t = time.time()

            for bid, batch in enumerate(tqdm(train_loader)):
                feats, targs, _, _ = batch
                state = None
                for feat, targ in split_sequence(feats, targs, **kwargs):
                    feat, targ = feat.to(device), targ.to(device)
                    optimizer.zero_grad()
                    predicted, state, _, _ = network(feat, state)
                    train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                    loss = criterion.get_loss(predicted, targ, 'train')
                    loss.backward()
                    train_loss += loss.cpu().detach().numpy()
                    optimizer.step()
                    predicted, state = predicted.detach(), state.detach()
                    step += 1

            train_errs[epoch] = train_loss / (train_mini_batches * global_sliding_size / global_window_size)
            end_t = time.time()
            if not quiet_mode:
                global_log('-' * 25)
                global_log('Epoch {}, time usage: {:.3f}s, loss: {}, vec_loss {}/{:.6f}'.format(
                    epoch, end_t - start_t, train_errs[epoch], train_vel.get_channel_avg(), train_vel.get_total_avg()))

            saved_model = False
            if val_loader:
                network.eval()
                val_vel = MSEAverage()
                val_loss = 0
                for bid, batch in enumerate(val_loader):
                    feats, targs, _, _ = batch
                    state = None
                    for feat, targ in split_sequence(feats, targs, **kwargs):
                        feat, targ = feat.to(device), targ.to(device)
                        optimizer.zero_grad()
                        pred, state, _, _ = network(feat, state)
                        val_vel.add(pred.cpu().detach().numpy(), targ.cpu().detach().numpy())
                        val_loss += criterion.get_loss(pred, targ, 'val').cpu().detach().numpy()
                        pred, state = pred.detach(), state.detach()
                val_loss = val_loss / (val_mini_batches * global_sliding_size / global_window_size)

                if not quiet_mode:
                    global_log('Validation loss: {} vec_loss: {}/{:.6f}'.format(val_loss, val_vel.get_channel_avg(),
                                                                                val_vel.get_total_avg()))

                writer.add_scalars('Loss', {'train': train_errs[epoch], 'val': val_loss}, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if global_out_dir:
                        model_path = osp.join(global_out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_errs[epoch],
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        global_log('Best Validation Model saved to ' + model_path)
                if use_scheduler:
                    scheduler.step(val_loss)

            if global_out_dir and not saved_model and (epoch + 1) % global_save_interval == 0:  # save even with validation
                model_path = osp.join(global_out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                global_log('Model saved to ' + model_path)

            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break
    except KeyboardInterrupt:
        global_log('-' * 60)
        global_log('Early terminate')

    global_log('Training completed')
    if global_out_dir:
        model_path = osp.join(global_out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)

def test_seq_to_seq(**kwargs):
    global_num_workers = kwargs.get('num_workers')
    global_device = kwargs.get('device')
    global_out_dir = kwargs.get('out_dir', None)
    global_test_dir = kwargs.get('test_dir', None)
    global_out_dir = kwargs.get('out_dir', None)
    global_model_path = kwargs.get('model_path', None)
    global_log = kwargs.get('log')
    global_window_size = kwargs.get('window_size', None)
    global_projection_width = kwargs.get('projection_width', None)
    global_sampling_rate = kwargs.get('sampling_rate', None)
    global_enable_complementary = kwargs.get('enable_complementary', None)
    global device
    device = torch.device(global_device if torch.cuda.is_available() else 'cpu')

    if global_test_dir is None:
        raise ValueError('Test_path is needed.')

    test_dirs = read_dir(global_test_dir)
    if global_out_dir and not osp.exists(global_out_dir):
        os.makedirs(global_out_dir)
    if global_model_path is None:
        raise ValueError('Model path is needed.')

    checkpoint = torch.load(global_model_path)

    network = get_model('test', **kwargs)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(device)
    global_log('Model {} loaded to device {}.'.format(global_model_path, device))

    losses_vec = MSEAverage()
    ate_all, rte_all, pde_all = [], [], []
    aye_all = []
    vec_loss = []
    pred_per_min = int(global_sampling_rate / global_projection_width) * 60 # 2 + 0.5*(x - 1) = 60

    # Test for every sequence
    for i in range(len(test_dirs)):
        seq_dir = [test_dirs[i]]
        seq_dataset = get_test_dataset(global_test_dir, seq_dir, **kwargs)
        seq_loader = DataLoader(seq_dataset, batch_size = 1, shuffle = False, num_workers = global_num_workers)
        magnetometer = seq_dataset.get_magn()
        magnetometer = magnetometer.reshape(-1, 3)

        preds = []
        targets = []
        merges = []
        encoders = []
        state = None
        for _, data in enumerate(seq_loader):
            feat, targ, _, _ = data
            feat, targ = feat.to(device), targ.to(device)
            pred, state, merged_outputs, encoder_outputs  = network(feat, state)
            pred = pred.cpu().detach().numpy()
            preds.append(pred)
            targets.append(targ.cpu().detach().numpy())
            merges.append(merged_outputs.cpu().detach().numpy())
            encoders.append(encoder_outputs.cpu().detach().numpy())
            state = state.detach()

        preds = np.array(preds)
        preds = np.squeeze(preds, axis = 1)
        targets = np.array(targets)
        targets = np.squeeze(targets, axis = 1)
        merges = np.array(merges)
        merges = np.squeeze(merges, axis = 1)
        encoders = np.array(encoders)
        encoders = np.squeeze(encoders, axis = 1)

        preds = preds.reshape(int(len(preds) * global_window_size / global_projection_width), 2)
        targets = targets.reshape(int(len(targets) * global_window_size / global_projection_width), 2)
        magnetometer = magnetometer[:len(preds)]

        gt_v = np.linalg.norm(targets, axis = 1)
        error_v = np.linalg.norm(targets - preds, axis = 1)
        vec_loss.append(np.concatenate((gt_v.reshape(-1, 1), error_v.reshape(-1, 1)), axis = 1))

        # Apply complementary filter
        if global_enable_complementary == True:
            preds = complementary_filter(preds, magnetometer)

        # Compute the loss
        losses_vec.add(preds, targets)
        vec_losses = losses_vec.average[-1]

        # Reconstruct the trajectory
        global_log("Reconstructing trajectory")
        pos_pred = reconstruct_traj(preds, **kwargs)
        pos_gt = reconstruct_traj(targets, **kwargs)
        # save the trajectory
        save_trajectory(pos_pred, pos_gt, merges, encoders, seq_dir[0], **kwargs)

        ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
        rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta = pred_per_min)
        pde = compute_position_drift_error(pos_pred, pos_gt)
        heading_error = compute_heading_error(preds, targets)
        ate_all.append(ate)
        if rte >= 0:
            rte_all.append(rte)
        pde_all.append(pde)
        if heading_error >= 0:
            aye_all.append(heading_error)

        print(seq_dir[0])
        draw_trajectory(pos_pred, pos_gt, seq_dir[0], ate, rte, **kwargs)
        global_log('Sequence {}, Vector loss {} / {}, ATE: {}, RTE:{}, PDE:{}, AYE:{}'.format(test_dirs[i], vec_losses, np.mean(vec_losses), ate,
                                                                                              rte, pde, heading_error))
    ate_all = np.array(ate_all)
    rte_all = np.array(rte_all)
    pde_all = np.array(pde_all)
    aye_all = np.array(aye_all)
    vec_loss = np.asarray(vec_loss, dtype="object")

    measure = format_string('ATE', 'RTE', 'PDE', 'AYE',  sep = '\t')
    values = format_string(np.mean(ate_all), np.mean(rte_all), np.mean(pde_all), np.mean(aye_all), sep = '\t')
    global_log(measure + '\n' + values)

    # save ate_all, rte_all, pde_all, orientation_error_cdf
    # np.save(global_out_dir + '/' + global_dataset + "_" + global_model_type + '_ate.npy', ate_all)
    # np.save(global_out_dir + '/' + global_dataset + "_" + global_model_type + '_rte.npy', rte_all)
    # np.save(global_out_dir + '/' + global_dataset + "_" + global_model_type + '_pde.npy', pde_all)
    # np.save(global_out_dir + '/' + global_dataset + "_" + global_model_type + '_aye.npy', aye_all)
    # np.save(global_out_dir + '/' + global_dataset + "_" + global_model_type + '_vec_loss.npy', vec_loss)

def test(**kwargs):
    test_seq_to_seq(**kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run seq2seq model in train/test mode [required]. Optional "
                                                 "configurations can be specified as --key [value..] pairs",
                                     add_help=True)
    parser.add_argument('--mode', type = str, help = 'Slect to train the model or test the model. [train/test]')
    args = parser.parse_args()

    kwargs = load_config()
    log = set_up_logging(**kwargs)
    kwargs['log'] = log
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    print(kwargs)
    if args.mode == 'train':
        train(**kwargs)
    elif args.mode == 'test':
        if MODEL_PATH is None:
            raise ValueError("Model path required")
        test(**kwargs)
