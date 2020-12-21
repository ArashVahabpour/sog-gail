import glob
import os
from functools import partial

import torch
import torch.nn as nn
import gym

from a2c_ppo_acktr.envs import VecNormalize

import matplotlib.pyplot as plt
import numpy as np


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


def visualize_env(args, actor_critic, epoch, num_steps=1000):
    # action_func: a function with input obs and output action
    action_func = partial(actor_critic.act, rnn_hxs=None, masks=None)
    device = next(actor_critic.parameters()).device

    if args.env_name == 'Circles-v0':
        import gym_sog
        env = gym.make(args.env_name, args=args)
    else:
        env = gym.make(args.env_name)

    obs = env.reset()

    filename = os.path.join(args.results_dir, f'{epoch}.png')

    for i in range(num_steps):
        with torch.no_grad():
            # we have to consider an extra dimension 0 because the actor critic object works with environment vectors (see the training code)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[None]
            _, actions_tensor, _, _ = action_func(obs_tensor)
            actions = actions_tensor[0].cpu().numpy()

        obs, _, done, _ = env.step(actions)

        if i == num_steps - 1:
            plt.figure(figsize=(10, 20))
            plt.plot(env.loc_history[:, 0], env.loc_history[:, 1])
            for r in args.radii:
                t = np.linspace(0, 2 * np.pi, 200)
                plt.plot(r * np.cos(t), r * np.sin(t) + r, color='#d0d0d0')
            max_r = np.max(np.abs(args.radii))
            plt.axis('equal')
            plt.axis('off')
            plt.xlim([-max_r, max_r])
            plt.ylim([-2 * max_r, 2 * max_r])
            plt.savefig(filename)
            plt.close()

    env.close()


def generate_latent_codes(args, count):
    """
    Returns:
        a row-wise one-hot tensor of shape `count x latent_size`
    """
    n = args.latent_size
    return torch.eye(n, device=args.device)[torch.randint(n, (count,))]


criterion = nn.MSELoss(reduction='none')


def resolve_latent_code(actor_critic, state, action, latent_size):
    batch_size = len(state)
    latent_batch_size = latent_size

    device = state.device

    # batch_size x latent_batch_size x variable_dim
    all_z = torch.eye(latent_size, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    all_state = state.unsqueeze(1).expand(-1, latent_batch_size, -1)
    all_action = action.unsqueeze(1).expand(-1, latent_batch_size, -1)

    with torch.no_grad():
        policy_action_all = actor_critic.act(all_state.reshape(batch_size * latent_batch_size, -1),
                                             all_z.reshape(batch_size * latent_batch_size, -1),
                                             None, deterministic=True)[1].reshape(batch_size, latent_batch_size, -1)

    # batch_size x latent_batch_size x action_dim
    loss = criterion(policy_action_all, all_action)

    # batch_size x latent_batch_size
    loss = loss.mean(dim=2)

    # batch_size
    _, argmin = loss.min(dim=1)

    # new_z: batch_size x latent_batch_size x n_latent
    # best_idx: batch_size x 1 x n_latent
    best_idx = argmin[:, None, None].repeat(1, 1, latent_size)

    # batch_size x 1 x n_latent
    best_z = torch.gather(all_z, 1, best_idx)

    # batch_size x n_latent
    best_z = best_z.squeeze(1)

    return best_z
