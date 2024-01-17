import json
import random
import os
from os import path as osp
from typing import List, Tuple
import h5py
import torch
import numpy as np
import quaternion
import math
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

class GlobSequence():
    """
    Datasets: RIDI, RONIN
    Property: global coordinate frame
    """
    # add 3-axis magnetometer
    feature_dim = 9
    # feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__()
        self.ts, self.features, self.targets, self.gt_pos = None, None, None, None
        # self.info = {}
        self.w = kwargs.get('interval', 1)
        self.magn_status = kwargs.get('use_magnetometer')
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        # raw data: gyroscope (3D), accelerometer (3D), gravity (1D), magnetometer (3D), game vector (4D),
        print("the data_path is:", data_path)
        data = np.load(osp.join(data_path, 'rawdata.npy'))
        ground_truth = np.load(osp.join(data_path, 'groundtruth.npy'))
        # already in global coordinate frame and start from start frame
        # timestamp (1D) gyroscope (3D), accelerometer (3D), magnetometer (3D), rotation vector (4D)
        gyro = data[:, 1:4]
        acce = data[:, 4:7]
        magn = data[:, 7:10]
        magn_diff = np.diff(magn, axis=0)
        magn_diff = np.concatenate([magn_diff, magn_diff[-1, :].reshape(1,3)], axis=0)
        ts = ground_truth[:, 0]
        # tango position
        tango_pos = ground_truth[:, 1:4]
        # tango orientation
        tango_ori = ground_truth[:, 4:8]
        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt
        # start_frame = self.info.get('start_frame', 0)
        self.ts = ts

        if self.magn_status == True:
            self.features = np.concatenate([gyro, acce, magn_diff], axis = 1)
        elif self.magn_status == False:
            self.features = np.concatenate([gyro, acce], axis = 1)
        self.magn = magn
        self.targets = glob_v[:, :2]
        self.orientations = tango_ori
        self.gt_pos = tango_pos

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_magn(self):
        return self.magn

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis = 1)

def load_sequences(seq_type, root_dir, data_list, **kwargs):
    features_all, targets_all, magn_all, aux_all = [], [], [], []

    for i in range(len(data_list)):
        seq = seq_type(osp.join(root_dir, data_list[i]), **kwargs)
        feat, targ, magn, aux = seq.get_feature(), seq.get_target(), seq.get_magn(), seq.get_aux()
        # add feat, targ, aux to list
        features_all.append(feat)
        targets_all.append(targ)
        magn_all.append(magn)
        aux_all.append(aux)

    return features_all, targets_all, magn_all, aux_all

class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, data_set, root_dir, data_list, stepsize = 50, slidingsize = 100,
                 random_shift = 0, shuffle = True, transform = None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.data_set = data_set
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.sliding_size = slidingsize
        self.step_size = stepsize
        self.random_shift = random_shift
        self.shuffle = shuffle
        self.transform = transform
        self.projection_width = kwargs.get('projection_width')
        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, self.magn, aux = load_sequences(seq_type, root_dir, data_list, **kwargs)
        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma', -1)
        targ_sigma = kwargs.get('target_sigma', -1)
        if feat_sigma > 0:
            # print("Smoothing the features with sigma = ", feat_sigma)
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            # print("Smoothing the targets with sigma = ", targ_sigma)
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        if self.data_set == "neurit":
            max_norm = 10 #
            self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
            for i in range(len(self.features)):
                self.features[i] = self.features[i][:-1]
                self.targets[i] = self.targets[i]
                self.ts.append(aux[i%len(data_list)][:-1, :1])
                self.orientations.append(aux[i%len(data_list)][:-1, 1:5])
                self.gt_pos.append(aux[i%len(data_list)][:-1, 5:8])

                velocity = np.linalg.norm(self.targets[i], axis=1) # Remove outlier ground truth data
                bad_data = velocity > max_norm
                for j in range(self.sliding_size + random_shift, self.targets[i].shape[0], self.step_size):
                    if not bad_data[j - self.sliding_size - random_shift:j + random_shift].any():
                        self.index_map.append([i, j])

        if self.shuffle == True:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        if self.data_set == "neurit":
            feat = np.copy(self.features[seq_id][frame_id - self.sliding_size:frame_id])
            targ = np.copy(self.targets[seq_id][frame_id - self.sliding_size:frame_id])
            # random rotate the sequence in the horizontal plane
            if self.transform is not None:
                feat, targ = self.transform(feat, targ)

            projection_targ = np.zeros((int(len(targ) / self.projection_width), 2))
            for i in range(int(len(targ) / self.projection_width)):
                projection_targ[i] = np.sum(targ[i * self.projection_width:(i + 1) * self.projection_width], axis=0) / self.projection_width

            return feat.astype(np.float32), projection_targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_lstm_test_seq(self):
        return np.array(self.features).astype(np.float32), np.array(self.targets).astype(np.float32)

    def get_magn(self):
        return np.array(self.magn).astype(np.float32)

def change_cf(ori, vectors):
    """
    Euler-Rodrigous formula v'=v+2s(rxv)+2rx(rxv)
    :param ori: quaternion [n]x4
    :param vectors: vector nx3
    :return: rotated vector nx3
    """
    assert ori.shape[-1] == 4
    assert vectors.shape[-1] == 3

    if len(ori.shape) == 1:
        ori = ori.reshape(1, -1)

    q_s = ori[:, :1]
    q_r = ori[:, 1:]

    tmp = np.cross(q_r, vectors)
    vectors = np.add(np.add(vectors, np.multiply(2, np.multiply(q_s, tmp))), np.multiply(2, np.cross(q_r, tmp)))
    return vectors

class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, feat, targ, **kwargs):
        for t in self.transforms:
            feat, targ = t(feat, targ)
        return feat, targ

class RandomHoriRotateSeq:
    def __init__(self, input_format, output_format=None):
        """
        Rotate global input, global output by a random angle
        @:param input format - input feature vector(x,3) boundaries as array (E.g [0,3,6] or [0,3,6,9])
        @:param output format - output feature vector(x,2/3) boundaries as array (E.g [0,2,5])
                                if 2, 0 is appended as z.
        """
        self.i_f = input_format
        self.o_f = output_format

    def __call__(self, feature, target):
        a = np.random.random() * 2 * np.math.pi
        t = np.array([np.cos(a), 0, 0, np.sin(a)])

        for i in range(len(self.i_f) - 1):
            feature[:, self.i_f[i]: self.i_f[i + 1]] = \
                change_cf(t, feature[:, self.i_f[i]: self.i_f[i + 1]])

        for i in range(len(self.o_f) - 1):
            if self.o_f[i + 1] - self.o_f[i] == 3:
                vector = target[self.o_f[i]: self.o_f[i + 1]]
                target[:, self.o_f[i]: self.o_f[i + 1]] = change_cf(t, vector)
            elif self.o_f[i + 1] - self.o_f[i] == 2:
                vector = np.concatenate([target[:, self.o_f[i]: self.o_f[i + 1]], np.zeros([target.shape[0], 1])], axis=1)
                target[:, self.o_f[i]: self.o_f[i + 1]] = change_cf(t, vector)[:, :2]

        return feature.astype(np.float32), target.astype(np.float32)

class RandomHoriRotateSeqTensor:
    def __init__(self):
        """
        Rotate global input, global output by a random angle
        @:param input format - input feature vector(x,3) boundaries as array (E.g [0,3,6])
        @:param output format - output feature vector(x,2/3) boundaries as array (E.g [0,2,5])
                                if 2, 0 is appended as z.
        """

    def __call__(self, feature, target):
        # Tensor random rotation matrix
        a = torch.rand(1) * 2 * np.math.pi
        rotation_matrix_feat = torch.tensor([[torch.cos(a), torch.sin(a), 0, 0, 0, 0],
                                            [-torch.sin(a), torch.cos(a), 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, torch.cos(a), torch.sin(a), 0],
                                            [0, 0, 0, -torch.sin(a), torch.cos(a), 0],
                                            [0, 0, 0, 0, 0, 1]], dtype=torch.float32)

        rotation_matrix_targ = torch.tensor([[torch.cos(a), torch.sin(a)],
                                            [-torch.sin(a), torch.cos(a)]], dtype=torch.float32)

        # Matrix multiplication
        feature = torch.matmul(feature, rotation_matrix_feat)
        target = torch.matmul(target, rotation_matrix_targ)

        return feature, target

def get_dataset(root_dir, data_list, mode, **kwargs):
    # load config
    global_step_size = 0
    global_dataset = kwargs.get('dataset')
    if kwargs['use_magnetometer'] == True:
        # input data includes: accelemeters, gyroscopes, magnetometers
        input_format = [0, 3, 6, 9]
    elif kwargs['use_magnetometer'] == False:
        input_format = [0, 3, 6]
    # output data is the moving distance and its direction
    output_format = [0, 2]

    random_shift, shuffle, transforms = 0, False, []

    if mode == 'train':
        random_shift = global_step_size // 2
        shuffle = True
        if kwargs['use_angmentation'] == True:
            print("Enable data augmentation: random horizontal rotation")
            transforms.append(RandomHoriRotateSeq(input_format, output_format))
        else:
            print("Disable data augmentation")
        global_step_size = kwargs.get('step_size')
        global_sliding_size = kwargs.get('sliding_size')
    elif mode == 'val':
        shuffle = True
        global_step_size = kwargs.get('step_size')
        global_sliding_size = kwargs.get('sliding_size')
    elif mode == 'test':
        shuffle = False
        global_step_size = kwargs.get('test_step_size')
        global_sliding_size = kwargs.get('test_sliding_size')
    transforms = ComposeTransform(transforms)

    if global_dataset == 'neurit':
        seq_type = GlobSequence

    dataset = SequenceToSequenceDataset(seq_type, global_dataset, root_dir, data_list, stepsize = global_step_size, slidingsize = global_sliding_size,
                                        random_shift = random_shift, shuffle = shuffle, transform = transforms, **kwargs)

    return dataset

def read_dir(dir_path):
    # read dirs from dir_path
    for _, dirs, _ in os.walk(dir_path):
        return dirs

def get_train_val_dataset(root_dir, **kwargs):
    total_dirs = read_dir(root_dir)
    random.shuffle(total_dirs)
    length = len(total_dirs)
    train_list = total_dirs[:int(length * 0.95)]
    val_list = total_dirs[int(length * 0.95):]
    return get_dataset(root_dir, train_list, mode = 'train', **kwargs), get_dataset(root_dir, val_list, mode = 'val', **kwargs)

def get_train_dataset(root_dir, **kwargs):
    trainlist = read_dir(root_dir)
    return get_dataset(root_dir, trainlist, mode = 'train', **kwargs)

def get_valid_dataset(root_dir, **kwargs):
    validlist = read_dir(root_dir)
    return get_dataset(root_dir, validlist, mode = 'val', **kwargs)

def get_test_dataset(root_dir, dir, **kwargs):
    return get_dataset(root_dir, dir, mode = 'test', **kwargs)

def get_seq_indices(seq_len: int, window_len: int) -> List[Tuple[int, int]]:
    prev = 0
    inds = []
    for curr in range(window_len, seq_len + 1, window_len):
        inds.append((prev, curr))
        prev += window_len
    return inds

def split_sequence(feat, targ, **kwargs):
    """
    Params:
        feat: input features: [batch_size, seq_len, feat_dim]
        targ: output features: [batch_size, seq_len, targ_dim]
    """
    seq_len = feat.shape[1]
    global_window_size = kwargs.get('window_size')
    for i, j in get_seq_indices(seq_len, global_window_size):
        yield feat[:, i:j], targ[:, i:j]
