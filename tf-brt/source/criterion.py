import json
import os
import sys
import time
import random
import argparse
import scipy.ndimage
import math
from os import path as osp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

class GlobalPosLoss(torch.nn.Module):
    def __init__(self):
        """
        Calculate position loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalPosLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction = 'none')

    def forward(self, pred, targ):
        gt_pos = torch.cumsum(targ[:, 1:, ], 1)
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)
        loss = self.mse_loss(pred_pos, gt_pos)
        # calculate the sum of absolute trajectory error
        return torch.mean(loss)

class GlobalVelLoss(nn.Module):
    def __init__(self):
        """
        Calculate velocity loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalVelLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction = 'none')

    def forward(self, pred, targ):
        loss = self.mse_loss(pred, targ)

        return torch.mean(loss)

class GlobalOriLoss(nn.Module):
    def __init__(self):
        """
        Calculate orientation loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalOriLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction = 'none')

    def forward(self, pred, targ):
        pred_v = torch.norm(pred, dim=2)
        targ_v = torch.norm(targ, dim=2)

        pred_o = pred / pred_v.unsqueeze(2)
        targ_o = targ / targ_v.unsqueeze(2)

        loss = self.mse_loss(pred_o, targ_o)

        return torch.mean(loss)

class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        self.criteria = nn.ModuleList([GlobalVelLoss(), GlobalPosLoss(), GlobalOriLoss()])

    def forward(self, predicts, targets):
        """
        Params:
            predicts: velocities of shape (batch_size, length, 2)
            targets: velocities of shape (batch_size, length, 2)
        """
        loss = torch.ones((len(self.criteria),))
        for i, criterion in enumerate(self.criteria):
            loss[i] = criterion(predicts, targets)

        return torch.mean(loss)

class CoVWeightingLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CoVWeightingLoss, self).__init__()
        self.device = kwargs.get('device')
        self.num_losses = 3
        self.posloss = GlobalPosLoss()
        self.oriloss = GlobalOriLoss()
        self.velloss = GlobalVelLoss()

    def forward(self, predicts, targets):
        """
        Params:
            predicts: velocities of shape (batch_size, length, 2)
            targets: velocities of shape (batch_size, length, 2)
        """
        # global velocity loss
        vel_loss = self.velloss(predicts, targets)

        # global position loss
        pos_loss = self.posloss(predicts, targets)

        # global orientation loss
        ori_loss = self.oriloss(predicts, targets)

        loss = [vel_loss, pos_loss, ori_loss]
        # no vl
        # loss = [pos_loss, ori_loss]
        # no pl
        # loss = [vel_loss, ori_loss]
        # no ol
        # loss = [vel_loss, pos_loss]

        return loss

class CovWeighting():
    def __init__(self, **kwargs):
        super(CovWeighting, self).__init__()
        self.device = kwargs.get('device')
        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = False
        self.mean_decay_param = 1.0

        self.criterion = CoVWeightingLoss()
        self.current_iter = -1
        self.num_losses = 3
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None

    def get_loss(self, predicts, targets, option):
        # Retrieve the unweighted losses.
        unweighted_losses = self.criterion.forward(predicts, targets)
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if option == 'val':
            return torch.sum(L) / self.num_losses

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        # elif self.current_iter > 0 and self.mean_decay:
        #     mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)

        return loss

    def get_val_loss(self, predicts, targets):
        # Retrieve the unweighted losses.
        unweighted_losses = self.criterion.forward(predicts, targets)
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        return torch.sum(L)

class MSEAverage():
    def __init__(self):
        self.count = 0
        self.targets = []
        self.predictions = []
        self.average = []

    def add(self, pred, targ):
        self.targets.append(targ)
        self.predictions.append(pred)
        self.average.append(np.average((pred - targ) ** 2, axis=(0, 1)))
        # print("The shape of average is: ", np.array(self.average).shape)
        # print("THe shape of np.average(np.array(self.average), axis=0) is: ", np.average(np.array(self.average), axis=0).shape)
        self.count += 1

    def get_channel_avg(self):
        average = np.average(np.array(self.average), axis=0)
        return average

    def get_total_avg(self):
        average = np.average(np.array(self.average), axis=0)
        return np.average(average)

    def get_elements(self, axis):
        return np.concatenate(self.predictions, axis=axis), np.concatenate(self.targets, axis=axis)

def reconstruct_traj(vector, **kwargs):
    global_projection_width = kwargs.get('projection_width')
    global_sampling_rate = kwargs.get('sampling_rate')
    # reconstruct the vector to one sequence
    # velocity_sequence = vector.reshape(len(vector) * global_window_size, global_output_channel)
    vector = vector / global_sampling_rate * global_projection_width
    glob_pos = np.cumsum(vector, axis = 0)

    return glob_pos

def compute_absolute_trajectory_error(pred, gt):
    """
    The Absolute Trajectory Error (ATE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.

    Return:
        Absolution trajectory error, which is the Root Mean Squared Error between
        two trajectories.
    """
    return np.sqrt(np.mean((pred - gt) ** 2))


def compute_relative_trajectory_error(est, gt, delta, max_delta=-1):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    if max_delta == -1:
        max_delta = est.shape[0]
    print("delta: ", delta)
    deltas = np.array([min(delta, max_delta - 1)])
    # deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]:] + gt[:-deltas[i]] - est[:-deltas[i]] - gt[deltas[i]:]
        rtes[i] = np.sqrt(np.mean(err ** 2))

    # The average of RTE of all window sized is returned.
    rtes = rtes[~np.isnan(rtes)]
    return np.mean(rtes)

def compute_position_drift_error(pos_pred, pos_gt):
    """
    Params:
        pos_pred: predicted position [seq_len, 2]
        pos_gt: ground truth position [seq_len, 2]
    """
    position_drift = np.linalg.norm((pos_gt[-1] - pos_pred[-1]))
    delta_position = pos_gt[1:] - pos_gt[:-1]
    delta_length = np.linalg.norm(delta_position, axis=1)
    moving_len = np.sum(delta_length)

    return position_drift / moving_len

def compute_distance_error(pos_pred, pos_gt):
    """
    Params:
        pos_pred: predicted position [seq_len, 2]
        pos_gt: ground truth position [seq_len, 2]
    """
    distance_error = np.linalg.norm((pos_gt - pos_pred), axis=1)

    return distance_error

def compute_heading_error(preds, targets):
    """
    Params:
        pos_pred: predicted position [seq_len, 2]
        pos_gt: ground truth position [seq_len, 2]
    """
    pred_v = np.linalg.norm(preds, axis=1)
    targ_v = np.linalg.norm(targets, axis=1)

    pred_o = preds / pred_v[:, np.newaxis]
    targ_o = targets / targ_v[:, np.newaxis]

    # calculate the heading angle of the predicted and target vectors
    pred_heading = np.arctan2(pred_o[:, 1], pred_o[:, 0])
    targ_heading = np.arctan2(targ_o[:, 1], targ_o[:, 0])

    # calculate the heading error
    heading_error = np.mean(np.abs(pred_heading - targ_heading))
    # convert to degrees
    heading_error = heading_error * 180 / np.pi

    return heading_error

import numpy as np

def calc_angle_2(v1, v2):
    '''
    calculate the angle between two vectors
    :param v1:
    :param v2:
    :return:
    '''
    r = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)))
    deg = r * 180 / np.pi

    a1 = np.array([*v1, 0])
    a2 = np.array([*v2, 0])

    a3 = np.cross(a1, a2)

    if np.sign(a3[2]) > 0:
        deg = 360 - deg

    return deg

def complementary_filter(preds, magn):
    """
    Params:
        preds: shape (seq_len, 2)
        magn: shape (seq_len, 3)
    """
    # copy preds
    new_preds = preds.copy()
    # Only use magnetometers on floor plane
    magnetometer = magn[:, 0:2]
    # apply low_pass filter to magnetometer
    magnetometer = scipy.ndimage.filters.gaussian_filter1d(magnetometer, sigma=50, axis=0)
    magnetometer = np.concatenate((magnetometer[:50], magnetometer), axis=0)

    for i in range(1, new_preds.shape[0]):
        if np.linalg.norm(new_preds[i]) == 0:
            continue
        # Initilization
        average_magnetometer = np.mean(magnetometer[i:i+50], axis=0)
        alpha_pre = calc_angle_2(new_preds[i-1], average_magnetometer)
        alpha_now = calc_angle_2(new_preds[i], magnetometer[i])
        theta_pre_now = calc_angle_2(new_preds[i-1], new_preds[i])
        if math.isnan(alpha_pre) or math.isnan(alpha_now) or math.isnan(theta_pre_now):
            continue
        hat_alpha_now = alpha_pre - theta_pre_now
        delta_alpha_pre_now = alpha_now - hat_alpha_now
        delta_alpha_pre_now = -0.01 * delta_alpha_pre_now * np.pi / 180
        delta_alpha_pre_now_magn = -0.01 * delta_alpha_pre_now * np.pi / 180
        rotation_matrix = np.asarray([[np.cos(delta_alpha_pre_now),-np.sin(delta_alpha_pre_now)],[np.sin(delta_alpha_pre_now),np.cos(delta_alpha_pre_now)]])
        rotation_matrix_magn = np.asarray([[np.cos(delta_alpha_pre_now_magn),-np.sin(delta_alpha_pre_now_magn)],[np.sin(delta_alpha_pre_now_magn),np.cos(delta_alpha_pre_now_magn)]])
        new_preds[i] = np.dot(new_preds[i], rotation_matrix)
        magnetometer[i] = np.dot(magnetometer[i], rotation_matrix_magn)

    return new_preds
