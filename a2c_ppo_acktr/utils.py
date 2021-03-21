import glob
import os
import torch
import torch.nn as nn
from torch.distributions import Normal
import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import product, permutations
import h5py

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


def generate_latent_codes(args, count=1, vae_modes=None):
    """
    Returns:
        a random tensor of shape `count x latent_dim`, which is row-wise-one-hot if latent codes are discrete, or Gaussian if continuous
    """
    n = args.latent_dim
    if args.vae_gail:
        perm = torch.randperm(vae_modes.size(0))
        idx = perm[:count]
        return vae_modes[idx]
    elif args.continuous:
        return torch.randn((count, n), device=args.device)
    else:
        return torch.eye(n, device=args.device)[torch.randint(n, (count,))]


class MujocoBase:
    def __init__(self, args, env, actor_critic, filename, obsfilt, max_episode_time=10):
        self.env = env
        self.max_episode_steps = int(max_episode_time / env.dt)
        self.actor_critic = actor_critic
        self.args = args
        self.obsfilt = obsfilt
        self.filename = filename


class MujocoPlay(MujocoBase):
    def evaluate_continuous(self, max_episode=10):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_size = (250, 250)
        VideoWriter = cv2.VideoWriter(f'{self.filename}.avi', fourcc, 1 / self.env.dt, video_size)

        args = self.args

        max_episode_per_dim = int(np.ceil(max_episode ** (1/args.latent_dim)))
        cdf = np.linspace(.1, .9, max_episode_per_dim)
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        latent_codes = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(args.device)

        for j, latent_code_ in enumerate(product(*[latent_codes] * args.latent_dim)):
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
                I = cv2.resize(I, video_size)
                VideoWriter.write(I)
            VideoWriter.write(np.zeros([*video_size, 3], dtype=np.uint8))
            print(f"episode reward:{episode_reward:3.3f}")
        self.env.close()
        VideoWriter.release()
        cv2.destroyAllWindows()

    def evaluate_discrete(self, max_episode=1):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_size = (250, 250)
        VideoWriter = cv2.VideoWriter(f'{self.filename}.avi', fourcc, 1 / self.env.dt, video_size)

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
                    I = cv2.resize(I, video_size)
                    VideoWriter.write(I)
                VideoWriter.write(np.zeros([*video_size, 3], dtype=np.uint8))
            print(f"episode reward:{episode_reward:3.3f}")
        self.env.close()
        VideoWriter.release()
        cv2.destroyAllWindows()


class MujocoBenchmark(MujocoBase):
    def halfcheetahvel_plot(self, num_codes, num_repeats, epoch):
        args = self.args

        cdf = np.linspace(.1, .9, num_codes)
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        latent_codes = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(args.device)

        vel_mean = []
        vel_std = []

        for j, latent_code in enumerate(latent_codes):
            vels = []
            latent_code = latent_code[None, None]
            for _ in range(num_repeats):
                s = self.env.reset()
                for step in range(self.max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s, r, done, infos = self.env.step(action)
                    vels.append(infos['forward_vel'])
            vel_mean.append(np.mean(vels))
            vel_std.append(np.std(vels))
        self.env.close()

        vel_mean, vel_std = np.array(vel_mean), np.array(vel_std)
        plt.figure()
        plt.plot(cdf, vel_mean, marker='o', color='r')
        plt.fill_between(cdf, vel_mean-vel_std, vel_mean+vel_std, alpha=0.2)
        plt.savefig(os.path.join(args.results_dir, f'{epoch}.png'))
        plt.close()

        plt.figure()
        plt.hist(vel_mean, bins=np.linspace(1.5, 3, 10))
        plt.savefig(os.path.join(args.results_dir, f'{epoch}_hist.png'))
        plt.close()

    def ant_plot(self, num_repeats, epoch):
        args = self.args
        plt.figure()

        if args.continuous:
            assert args.latent_dim == 1, 'higher latent dim not implemented'
            num_codes = 30
            num_repeats = 1
            cdf = np.linspace(.1, .9, num_codes)
            m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            all_codes = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(args.device)[:, None]
        else:
            all_codes = torch.eye(args.latent_dim, device=args.device)

        for j, latent_code in enumerate(all_codes):
            latent_code = latent_code[None]
            for _ in range(num_repeats):
                s = self.env.reset()
                xpos = []
                for step in range(self.max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s, r, done, infos = self.env.step(action)
                    xpos.append(infos['xpos'])
                xpos = np.array(xpos)
                plt.plot(xpos[:, 0], xpos[:, 1], color=('b' if continuous else plt.cm.Dark2.colors[j]))
        plt.plot([0], [0], marker='o', markersize=3, color='k')
        plt.axis('off')
        plt.axis('equal')
        plt.savefig(os.path.join(args.results_dir, f'{epoch}.png'))
        plt.close()
        self.env.close()

    def ant_robustness_test(self, num_cycles, num_steps, epoch):
        args = self.args
        plt.figure()
        s = self.env.reset()
        for c in range(num_cycles):
            for j, latent_code in enumerate(torch.eye(args.latent_dim, device=args.device)):
                xpos = []
                latent_code = latent_code[None]
                for step in range(num_steps):
                    print(c,j, step)
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s, r, done, infos = self.env.step(action)
                    xpos.append(infos['xpos'])
                xpos = np.array(xpos)
                plt.plot(xpos[:, 0], xpos[:, 1], color=plt.cm.Dark2.colors[j])
        plt.plot([0], [0], marker='o', markersize=3, color='k')
        plt.axis('off')
        plt.axis('equal')
        plt.savefig(os.path.join(args.results_dir, f'{epoch}_robustness.png'))
        plt.close()
        self.env.close()

    def collect_rewards(self):
        """
        Creates matrix of rewards of latent codes vs radii

        Returns:
            all_mode_rewards_mean: numpy array of shape [latent_dim x latent_dim] containing mean reward collected in a trajectory
            all_mode_rewards_std: numpy array of shape [latent_dim x latent_dim] containing std of rewards collected among trajectories
        """

        args = self.args
        device = args.device
        latent_dim = args.latent_dim

        trajs_per_mode = 10
        max_steps = 200

        all_mode_rewards_mean, all_mode_rewards_std = [], []
        for i, latent_code in enumerate(torch.eye(latent_dim, device=device)):
            latent_code = latent_code[None]
            all_traj_rewards = []
            for _ in range(trajs_per_mode):
                obs = self.env.reset()
                traj_rewards = np.zeros(latent_dim)
                for step in range(max_steps):
                    with torch.no_grad():
                        # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                        obs = self.obsfilt(obs, update=False)
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[None]
                        _, actions_tensor, _ = self.actor_critic.act(obs_tensor, latent_code, deterministic=True)
                        action = actions_tensor[0].cpu().numpy()

                    obs, _, _, infos = self.env.step(action)
                    traj_rewards += np.array(infos['rewards'])
                all_traj_rewards.append(traj_rewards)

            all_traj_rewards = np.stack(all_traj_rewards)
            all_mode_rewards_mean.append(all_traj_rewards.mean(axis=0))
            all_mode_rewards_std.append(all_traj_rewards.std(axis=0))

        return np.stack(all_mode_rewards_mean), np.stack(all_mode_rewards_std)


class CirclesBenchmark:
    def __init__(self, args, env, actor_critic, filename, obsfilt):
        self.env = env
        self.max_steps = 1000
        self.trajs_per_mode = 10
        self.actor_critic = actor_critic
        self.args = args
        self.latent_dim = args.latent_dim
        self.obsfilt = obsfilt
        self.filename = filename

    def collect_rewards(self, use_expert=False):
        """
        Creates matrix of rewards of latent codes vs radii

        Args:
            use_expert: determines whether the trajectories should be generated by expert or agent policy

        Returns:
            all_mode_rewards_mean: numpy array of shape [latent_dim x latent_dim] containing mean reward collected in a trajectory
            all_mode_rewards_std: numpy array of shape [latent_dim x latent_dim] containing std of rewards collected among trajectories
        """

        args = self.args
        device = args.device
        latent_dim = args.latent_dim
        radii = args.radii

        if use_expert:
            from gym_sog.envs.circles_expert import CirclesExpert
            circles_expert = CirclesExpert(self.args)

        all_mode_rewards_mean, all_mode_rewards_std = [], []
        for i, latent_code in enumerate(torch.eye(latent_dim, device=device)):
            latent_code = latent_code[None]
            all_traj_rewards = []
            for _ in range(self.trajs_per_mode):
                obs = self.env.reset()
                traj_rewards = np.zeros(latent_dim)
                for step in range(self.max_steps):
                    if use_expert:
                        action = circles_expert.policy(obs, radii[i])
                    else:
                        with torch.no_grad():
                            # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                            obs = self.obsfilt(obs, update=False)
                            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[None]
                            _, actions_tensor, _ = self.actor_critic.act(obs_tensor, latent_code, deterministic=True)
                            action = actions_tensor[0].cpu().numpy()

                    obs, _, _, infos = self.env.step(action)
                    traj_rewards += np.array(infos['rewards'])
                all_traj_rewards.append(traj_rewards)

            all_traj_rewards = np.stack(all_traj_rewards)
            all_mode_rewards_mean.append(all_traj_rewards.mean(axis=0))
            all_mode_rewards_std.append(all_traj_rewards.std(axis=0))

        return np.stack(all_mode_rewards_mean), np.stack(all_mode_rewards_std)


def visualize_env(args, actor_critic, obsfilt, epoch, num_steps=1000):
    filename = os.path.join(args.results_dir, str(epoch))

    if args.env_name in {'Circles-v0', 'Ellipses-v0'}:
        plt.figure(figsize=(10, 20))
        plt.set_cmap('gist_rainbow')
        # plotting the actual circles
        for r in args.radii:
            t = np.linspace(0, 2 * np.pi, 200)
            plt.plot(r * np.cos(t), r * np.sin(t) + r, color='#d0d0d0')
            max_r = np.max(np.abs(args.radii))
            plt.axis('equal')
            plt.axis('off')
            plt.xlim([-1.5 * max_r, 1.5 * max_r])
            plt.ylim([-3 * max_r, 3 * max_r])

        import gym_sog
        env = gym.make(args.env_name, args=args)
        obs = env.reset()

        device = next(actor_critic.parameters()).device
        if args.vae_gail:
            vae_modes = torch.load(args.save_filename.format('vae_modes'), map_location='cpu').numpy()
            from sklearn.cluster import KMeans
            latent_codes = KMeans(n_clusters=3).fit(vae_modes).cluster_centers_
            latent_codes = torch.tensor(latent_codes, dtype=torch.float32, device=args.device)
        elif args.continuous:
            latent_codes = torch.randn(5, args.latent_dim, device=device)
        else:
            latent_codes = torch.eye(args.latent_dim, device=device)
        # generate rollouts and plot them
        for j, latent_code in enumerate(latent_codes):
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

    elif args.mujoco:
        import rlkit
        env = gym.make(args.env_name)
        mujoco_play = MujocoPlay(args, env, actor_critic, filename, obsfilt)
        if args.sog_gail and args.latent_optimizer == 'bcs':
            mujoco_play.evaluate_continuous()
        else:
            mujoco_play.evaluate_discrete()

    else:
        raise NotImplementedError


# def plot_ant_expert(states, args):
#     plt.figure(figsize=(20, 20))
#     for i, traj in enumerate(states):
#         if i < 100:
#             plt.plot(traj[:, 0], traj[:, 1], color='blue')
#     plt.plot([0], [0], marker='o', markersize=3, color='k')
#     plt.savefig(os.path.join(args.results_dir, 'expert.png'))
#     plt.close()


def record_rewards(h5, args, group, rew_mean, rew_std):
    """Records rewards according to best correspondence of latent codes and actual modes."""
    max_reward, best_mean, best_std = -np.inf, None, None
    for perm_mean, perm_std in zip(permutations(rew_mean), permutations(rew_std)):
        tmp = np.array(perm_mean).trace()
        if tmp > max_reward:
            max_reward = tmp
            best_mean = perm_mean
            best_std = perm_std
    if group in h5:
        del h5[group]
    h5.create_group(group)
    h5[group]['mean'], h5[group]['std'] = np.diag(np.array(best_mean)), np.diag(np.array(best_std))


def benchmark_env(args, actor_critic, obsfilt, epoch):
    filename = os.path.join(args.results_dir, str(epoch))

    if args.mujoco:
        import rlkit
        kwargs = {}
        if args.env_name == 'AntDir-v0' and not (args.sog_gail and args.latent_optimizer == 'bcs'):
            kwargs['n_tasks'] = args.latent_dim
        env = gym.make(args.env_name, **kwargs)
        mujoco_bench = MujocoBenchmark(args, env, actor_critic, filename, obsfilt)
        if args.env_name == 'HalfCheetahVel-v0':
            mujoco_bench.halfcheetahvel_plot(50, 1, epoch)
        elif args.env_name == 'AntDir-v0':
            mujoco_bench.ant_plot(3, epoch)
            # rew_mean, rew_std = mujoco_bench.collect_rewards()
            # print(f"""
            #         policy:
            #         {rew_mean}
            #         +/-
            #         {rew_std}
            #         """)
            # mujoco_bench.ant_robustness_test(10, 50, epoch)
        else:
            raise NotImplementedError
    elif args.env_name == 'Circles-v0':
        import gym_sog
        env = gym.make(args.env_name, args=args)
        h5 = h5py.File(os.path.join(args.results_dir, 'rewards.h5'), 'a')
        circles_bench = CirclesBenchmark(args, env, actor_critic, filename, obsfilt)
        assert args.latent_dim <= 5, 'all permutations of too many latent codes is prohibitively costly'

        record_rewards(h5, args, str(epoch), *circles_bench.collect_rewards(False))
        if 'expert' not in h5:
            record_rewards(h5, args, 'expert', *circles_bench.collect_rewards(True))

    else:
        raise NotImplementedError
