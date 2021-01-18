import glob
import os
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


def generate_latent_codes(args, count=1):
    """
    Returns:
        a random row-wise-one-hot tensor of shape `count x latent_dim`
    """
    n = args.latent_dim
    return torch.eye(n, device=args.device)[torch.randint(n, (count,))]


def visualize_env(args, actor_critic, epoch, num_steps=1000):
    plt.figure(figsize=(10, 20))
    plt.set_cmap('gist_rainbow')

    # plotting the actual circles
    for r in args.radii:
        t = np.linspace(0, 2 * np.pi, 200)
        plt.plot(r * np.cos(t), r * np.sin(t) + r, color='#d0d0d0')
    max_r = np.max(np.abs(args.radii))
    plt.axis('equal')
    plt.axis('off')
    plt.xlim([-max_r, max_r])
    plt.ylim([-2 * max_r, 2 * max_r])

    # preparing the environment
    device = next(actor_critic.parameters()).device
    if args.env_name in {'Circles-v0', 'Ellipses-v0'}:
        import gym_sog
        env = gym.make(args.env_name, args=args)
    else:
        env = gym.make(args.env_name)
    obs = env.reset()

    # generate rollouts and plot them
    for j, latent_code in enumerate(torch.eye(args.latent_dim, device=device)):
        latent_code = latent_code.unsqueeze(0)

        for i in range(num_steps):
            # randomize latent code at each step in case of vanilla gail
            if args.vanilla:
                latent_code = generate_latent_codes(args)
            # interacting with env
            with torch.no_grad():
                # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[None]
                _, actions_tensor, _ = actor_critic.act(obs_tensor, latent_code, deterministic=True)
                action = actions_tensor[0].cpu().numpy()
            obs, _, _, _ = env.step(action)

        # plotting the trajectory
        plt.plot(env.loc_history[:, 0], env.loc_history[:, 1], color=plt.cm.Dark2.colors[j])
        if args.vanilla:
            break  # one trajectory in vanilla mode is enough. if not, then rollout for each separate latent code
        else:
            obs = env.reset()

    env.close()

    filename = os.path.join(args.results_dir, f'{epoch}.png')
    plt.savefig(filename)
    plt.close()

