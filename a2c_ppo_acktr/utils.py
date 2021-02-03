import glob
import os
import torch
import torch.nn as nn
from torch.distributions import Normal
import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import product

from a2c_ppo_acktr.envs import VecNormalize

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


def generate_latent_codes(args, count=1):
    """
    Returns:
        a random row-wise-one-hot tensor of shape `count x latent_dim`
    """
    n = args.latent_dim
    return torch.eye(n, device=args.device)[torch.randint(n, (count,))]


class MujocoPlay:
    def __init__(self, args, env, actor_critic, filename, obsfilt, max_episode_time=10):
        self.env = env
        self.max_episode_steps = int(max_episode_time / env.dt)
        self.actor_critic = actor_critic
        self.args = args
        self.obsfilt = obsfilt

        # self.agent.set_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_size = (250, 250)
        self.VideoWriter = cv2.VideoWriter(f'{filename}.avi', self.fourcc, 1/env.dt, self.video_size)

    def evaluate_continuous(self, max_episode=10):
        args = self.args

        max_episode_per_dim = int(np.ceil(max_episode ** (1/args.latent_dim)))
        cdf = np.linspace(.1, .9, max_episode_per_dim)
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        latent_codes = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(args.device)

        for latent_code_ in product(*[latent_codes] * args.latent_dim):
            episode_reward = 0
            s = self.env.reset()
            for step in range(self.max_episode_steps):
                latent_code = torch.stack(list(latent_code_))[None]
                s = self.obsfilt(s, update=False)
                s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
                with torch.no_grad():
                    _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                action = actions_tensor[0].cpu().numpy()
                s_, r, done, _ = self.env.step(action)
                episode_reward += r
                if done:
                    break
                s = s_
                I = self.env.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, self.video_size)
                self.VideoWriter.write(I)
            self.VideoWriter.write(np.zeros([*self.video_size, 3], dtype=np.uint8))
            print(f"episode reward:{episode_reward:3.3f}")
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()

    def evaluate_discrete(self, max_episode=1):
        args = self.args
        for _ in range(max_episode):
            episode_reward = 0
            for j, latent_code in enumerate(torch.eye(args.latent_dim, device=args.device)):
                s = self.env.reset()
                latent_code = latent_code.unsqueeze(0)
                for step in range(self.max_episode_steps):
                    if args.vanilla:
                        latent_code = generate_latent_codes(args)
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s_, r, done, _ = self.env.step(action)
                    episode_reward += r
                    if done:
                        break
                    s = s_
                    I = self.env.render(mode='rgb_array')
                    I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                    I = cv2.resize(I, self.video_size)
                    self.VideoWriter.write(I)
                self.VideoWriter.write(np.zeros([*self.video_size, 3], dtype=np.uint8))
            print(f"episode reward:{episode_reward:3.3f}")
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()


def visualize_env(args, actor_critic, obsfilt, epoch, num_steps=1000):
    plt.figure(figsize=(10, 20))
    plt.set_cmap('gist_rainbow')

    # plotting the actual circles
    if args.env_name == 'Circles-v0':
        for r in args.radii:
            t = np.linspace(0, 2 * np.pi, 200)
            plt.plot(r * np.cos(t), r * np.sin(t) + r, color='#d0d0d0')
            max_r = np.max(np.abs(args.radii))
            plt.axis('equal')
            plt.axis('off')
            plt.xlim([-1.5 * max_r, 1.5 * max_r])
            plt.ylim([-3 * max_r, 3 * max_r])

    elif not args.mujoco:
        raise NotImplementedError

    # preparing the environment
    device = next(actor_critic.parameters()).device
    if args.env_name == 'Circles-v0':
        import gym_sog
        env = gym.make(args.env_name, args=args)
    elif args.mujoco:
        import rlkit
        env = gym.make(args.env_name)
    obs = env.reset()

    filename = os.path.join(args.results_dir, str(epoch))

    if args.mujoco:
        mujoco_play = MujocoPlay(args, env, actor_critic, filename, obsfilt)
        if args.sog_gail and args.latent_optimizer == 'bcs':
            mujoco_play.evaluate_continuous()
        else:
            mujoco_play.evaluate_discrete()

    else:
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
                    obs = obsfilt(obs, update=False)
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

        # from itertools import product
        # n_grid = 10
        # for i1, i2 in product(range(n_grid), range(n_grid)):
        #     init_loc = np.array([[np.linspace(-max_r, max_r, n_grid)[i1], np.linspace(2*max_r, -2*max_r, n_grid)[i2]]]*5)
        #     # generate rollouts and plot them
        #     for j, latent_code in enumerate(torch.eye(args.latent_dim, device=device)):
        #         env.manual_init(init_loc)
        #         latent_code = latent_code.unsqueeze(0)
        #
        #         for i in range(num_steps):
        #             # randomize latent code at each step in case of vanilla gail
        #             if args.vanilla:
        #                 latent_code = generate_latent_codes(args)
        #             # interacting with env
        #             with torch.no_grad():
        #                 # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
        #                 obs = obsfilt(obs, update=False)
        #                 obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[None]
        #                 _, actions_tensor, _ = actor_critic.act(obs_tensor, latent_code, deterministic=True)
        #                 action = actions_tensor[0].cpu().numpy()
        #             obs, _, _, _ = env.step(action)
        #
        #         # plotting the trajectory
        #         ax = plt.subplot(n_grid, n_grid, 1+i2*n_grid+i1)
        #         ax.set_xlim([-1.5 * max_r, 1.5 * max_r])
        #         ax.set_ylim([-3 * max_r, 3 * max_r])
        #         ax.set_axis_off()
        #         plt.plot(env.loc_history[:, 0], env.loc_history[:, 1], color=plt.cm.Dark2.colors[j])
        #         if args.vanilla:
        #             break  # one trajectory in vanilla mode is enough. if not, then rollout for each separate latent code
        #         else:
        #             obs = env.reset()

        env.close()

        plt.savefig(filename + '.png')
        plt.close()


def single_traj_loader(expert_filename, batch_size):
    """
    a generator that returns (s, a) samples of a shared sample trajectory

    Args:
        expert_filename: expert file name which contains the following tensors
            states: tensor of size `num_trajs x num_steps x state_dim`
            actions: tensor of size `num_trajs x num_steps x action_dim`
        batch_size: size of batches to return
    """

    expert = torch.load(expert_filename)
    states, actions = expert['states'], expert['actions']

    while True:
        traj_idx = np.random.randint(states.shape[0])
        step_idx = np.random.randint(low=0, high=states.shape[1], size=batch_size)
        yield states[traj_idx, step_idx], actions[traj_idx, step_idx]
