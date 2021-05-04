import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from a2c_ppo_acktr.envs import VecNormalize
from itertools import product

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True) #TODO remove these lines if not really needed


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)


    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def generate_latent_codes(args, count=None, vae_data=None, eval=False):
    """
    Args:
        eval: indicates whether the returned values are suitable for training or evaluation purposes
        count: number of samples
        vae_data: tuple of vae_modes, vae_cluster_centers. the former is of size `dataset_len x latent_dim` and the latter `k x latent_dim`
    Returns:
        a random tensor of shape `count x latent_dim`, which is row-wise-one-hot if latent codes are discrete, or Gaussian if continuous
    """
    n = args.latent_dim
    if args.vae_gail:
        vae_mus, vae_log_vars, vae_cluster_centers = vae_data
    if eval:
        if args.vae_gail:
            if args.vae_kmeans_clusters > 0:
                return vae_cluster_centers
            else:
                raise NotImplementedError('please implement below') #TODO (calculate cluster centers)
                trajs = torch.load()
                np.unique(torch.load('./trajs_halfcheetahvel.pt')['modes'])
            # else:  # if not args.vae_kmeans_cluster > 0
            #     perm = torch.randperm(vae_mus.size(0))
            #     idx = perm[:count]
            #     return vae_mus[idx]
        elif args.continuous:
            # make an n-dimensional grid
            if count is None:
                count = 5
            count_per_dim = int(np.ceil(count ** (1 / args.latent_dim)))
            cdf = np.linspace(.1, .9, count_per_dim)
            m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            bins = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(args.device)
            return torch.stack([torch.stack(t) for t in list(product(*[bins]*args.latent_dim))])
        else:  # all one-hot codes
            return torch.eye(args.latent_dim, device=args.device)

    else:
        if count is None:
            count = 1
        if args.vae_gail:
            perm = torch.randperm(vae_mus.size(0))
            idx = perm[:count]
            return vae_mus[idx]
        elif args.continuous:
            return torch.randn((count, n), device=args.device)
        else:
            return torch.eye(n, device=args.device)[torch.randint(n, (count,))]
