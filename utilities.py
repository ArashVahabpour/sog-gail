import pickle
import os
from typing import Set
import torch
import torch.nn
import numpy as np
# from numpy import random
import random
import scipy.signal
from collections import deque
import matplotlib.pyplot as plt
#from running_state import ZFilter
import math
import logging


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators
    :param seed: (int)
    :param using_cuda: (bool)
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        torch.cuda.manual_seed(seed)
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff:
        pickle.dump(variable, ff)


def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn, " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        # print(fn)
    return lookup

# InfoGail related:


def discount(x, gamma):
    assert x.ndim >= 1
    #print("discount filter:", x)
    #print("::::::::", scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1])
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
# ZFilter


def gauss_prob_np(mu, logstd, x):
    std = np.exp(logstd)
    var = np.square(std)
    gp = np.exp(-np.square(x - mu)/(2*var)) / ((2*np.pi)**.5 * std)
    return np.prod(gp, axis=1)


def gauss_prob(mu, logstd, x):
    std = torch.exp(logstd)
    var = torch.square(std)
    gp = torch.exp(-torch.square(x - mu)/(2*var)) / ((2*np.pi)**.5 * std)
    return torch.reduce_prod(gp, [1])


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    # pylint: disable=not-callable
    var = std.pow(2)
    torch_pi = torch.asin(torch.tensor(1.))
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * torch.log(2 * torch_pi) - log_std
    return log_density.sum(1, keepdim=True)

# def normal_log_density(x, mean, log_std, std):
#     var = std.pow(2)
#     log_density = -(x - mean).pow(2) / (
#         2 * var) - 0.5 * math.log(2 * math.pi) - log_std
#     return log_density.sum(1, keepdim=True)


def normal_log_density_fixedstd(x, mean):
    std = torch.from_numpy(np.array([2, 2])).clone().float()
    var = std.pow(2)
    log_std = torch.log(std)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = torch.exp(2*logstd1)
    var2 = torch.exp(2*logstd2)
    kl = torch.sum(logstd2 - logstd1 +
                   (var1 + torch.square(mu1 - mu2))/(2*var2) - 0.5)
    return kl


def gauss_ent(mu, logstd):
    h = torch.sum(logstd + torch.constant(0.5 *
                                          np.log(2*np.pi*np.e), torch.float32))
    return h


def gauss_sample(mu, logstd):
    return mu + torch.exp(logstd)*torch.random_normal(torch.shape(logstd))


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is a file")


def save_checkpoint(state, save_path='models/checkpoint.pth.tar'):
    create_dir(os.path.dirname(save_path))
    torch.save(state, save_path)


# TODO: use torch.nn.functional.one_hot
def onehot(data, dim: int):
    # return torch.zeros(*data.shape[:-1], dim).scatter_(-1, data, 1)
    fake_z = np.zeros((data.shape[0], dim))
    row = np.arange(data.shape[0])
    fake_z[row, data] = 1
    return fake_z


def generate_random_code(count: int, latent_size: int):
    # return torch.nn.functional.one_hot(torch.randint(0, latent_size, (count,))).float()
    return onehot(np.random.randint(latent_size, size=(count,)), latent_size)


def visualize_pts_tb(writer, locations, latent_code, fig_key, iter=0):
    """
    Visualize pts in the tensorboard
    """
    fig = plt.figure()
    if latent_code is not None:
        latent_code_num = np.argmax(latent_code, axis=1)
    else:
        latent_code_num = np.zeros(locations.shape[0])  # default k color
    col_list = np.where(latent_code_num == 0, 'k',
                        np.where(latent_code_num == 1, 'b', 'r'))
    plt.scatter(locations[:, 0], locations[:, 1], c=col_list, s=5)
    plt.plot(locations[:, 0], locations[:, 1], "-", alpha=0.5)
    plt.title(f"iter:{iter}")
    writer.add_figure(fig_key, fig)


def step(state, action, mode="flat"):
    if mode == "nested":
        cur_loc = state[:, -1, :]
        next_loc = cur_loc + action
        new_state = torch.cat(
            [state[:, 1:, :], next_loc.reshape(-1, 1, 2)], axis=1)
    elif mode == "flat":
        cur_loc = state[:, -2:]
        next_loc = cur_loc + action
        new_state = torch.cat([state[:, 2:], next_loc.reshape(-1, 2)], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return new_state


def to_tensor(target, device):
    # pylint: disable=not-callable
    if target is None:
        return None
    try:
        target = torch.as_tensor(target, device=device).float()
    except:
        target = torch.tensor(target).float().to(device)
    return target


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def load_data(data_f):
    """For Qiujing's format
    """
    data_dict = load_pickle(data_f)
    keys = list(data_dict.keys())
    X_all = []
    y_all = []
    c_all = []
    # three one hot encoder

    for key_i, val in enumerate(data_dict.values()):
        num_traj = val['state'].shape[0]
        traj_len = val['state'].shape[1]
        num_data = num_traj * traj_len
        c_all.append((key_i * np.ones(num_data)).astype(int))
        y_all.append(val['action'])
        X_all.append(val['state'])

    c_all = np.concatenate(c_all)
    fake_z0 = np.random.randint(3, size=num_traj * 3)
    fake_z0 = np.repeat(fake_z0, traj_len)
    print(fake_z0.shape)
    fake_z = onehot(fake_z0, 3)
    print(fake_z.shape, fake_z[0], fake_z0[0])

    y_all = np.concatenate(y_all).squeeze().reshape(-1, 2)

    X_all = np.concatenate(X_all).transpose(
        (0, 1, 3, 2)).flatten().reshape(-1, 10)
    print(c_all.shape, y_all.shape, X_all.shape)
    return X_all, y_all, c_all, fake_z


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_paths = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        if self.num_paths < sample_size:
            return random.sample(self.buffer, self.num_paths)
        else:
            return random.sample(self.buffer, sample_size)

    def size(self):
        return self.buffer_size

    def add(self, path):
        if self.num_paths < self.buffer_size:
            self.buffer.append(path)
            self.num_paths += 1
        else:
            self.buffer.popleft()
            self.buffer.append(path)

    def count(self):
        return self.num_paths

    def erase(self):
        self.buffer = deque()
        self.num_paths = 0


def get_module_device(module):
    return next(module.parameters()).device


def get_unique_devices_(module: torch.nn.Module) -> Set[torch.device]:
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}
