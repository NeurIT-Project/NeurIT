import os
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')
    return write_log

def set_up_logging(**kwargs):
    if kwargs['out_dir'] is None:
        raise ValueError('out_dir is needed')
    if kwargs['model_type'] is None:
        raise ValueError('model_type is needed')
    if kwargs['dataset'] is None:
        raise ValueError('dataset is needed')
    log = logging(os.path.join(kwargs["out_dir"], kwargs["model_type"] + "_" + kwargs["dataset"] + '.txt'))

    log("%s:\t%s\n" % (str(kwargs["model_type"]), str(kwargs["dataset"])))
    return log

def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]

def draw_trajectory(pos_pred, pos_gt, dir_name, ate, rte, **kwargs):
    """
    :param data:
    :pos_pred: (N, 2)
    :pos_gt: (N, 2)
    :dir_name: test directory
    :ate: average trajectory error
    :rte: relative trajectory error
    """
    global_out_dir = kwargs.get('out_dir', None)
    if global_out_dir is None:
        raise ValueError('out_dir is needed')

    plt.figure(figsize=(8, 5), dpi = 400)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1], label = 'Predicted')
    plt.plot(pos_gt[:, 0], pos_gt[:, 1], label = 'Ground truth')
    plt.title(dir_name)
    print("make title success")
    plt.xlabel('$m$')
    plt.ylabel('$m$')
    plt.axis('equal')
    plt.legend()
    plt.title('ATE:{:.3f}, RTE:{:.3f}'.format(ate, rte), y = 0, loc = 'right')

    plt.savefig(osp.join(global_out_dir, '{}.png'.format(dir_name)))

def save_trajectory(pos_pred, pos_gt, merges, encoders, dir_name, **kwargs):

    global_out_dir = kwargs.get('out_dir', None)
    if global_out_dir is None:
        raise ValueError('out_dir is needed')
    global_enable_complementary = kwargs.get('enable_complementary', "")
    if global_enable_complementary:
        tag = "complementary"
    else:
        tag = "no_complementary"

    np.save(osp.join(global_out_dir, tag + '{}_pred.npy'.format(dir_name)), pos_pred)
    np.save(osp.join(global_out_dir, tag + '{}_gt.npy'.format(dir_name)), pos_gt)
    np.save(osp.join(global_out_dir, tag + '{}_merges.npy'.format(dir_name)), merges)
    np.save(osp.join(global_out_dir, tag + '{}_encoders.npy'.format(dir_name)), encoders)
