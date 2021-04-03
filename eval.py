import torch
import os
from torch.distributions import Normal
import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import permutations
import h5py
from sklearn.feature_selection import mutual_info_regression

from a2c_ppo_acktr.utils import generate_latent_codes


class Base:
    def __init__(self, args, env, actor_critic, filename, obsfilt, vae_data):
        self.env = env
        self.actor_critic = actor_critic
        self.args = args
        self.obsfilt = obsfilt
        self.filename = filename
        self.vae_mus, self.vae_log_vars, self.vae_cluster_centers = self.vae_data = vae_data
        assert not args.vanilla, 'Vanilla GAIL benchmarking not implemented'


class Play(Base):  # TODO fix faulty save of files
    def __init__(self, args, env, actor_critic, filename, obsfilt, vae_data):
        super(Play, self).__init__(args, env, actor_critic, filename, obsfilt, vae_data)
        max_episode_time = 10
        self.max_episode_steps = int(max_episode_time / env.dt)

    def play(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_size = (250, 250)
        video_writer = cv2.VideoWriter(f'{self.filename}.avi', fourcc, 1 / self.env.dt, video_size)

        args = self.args

        count = None
        if args.vae_gail and args.env_name == 'HalfCheetahVel-v0':
            count = 30
        latent_codes = generate_latent_codes(args, count=count, vae_data=self.vae_data, eval=True)

        for j, latent_code in enumerate(latent_codes):
            episode_reward = 0
            s = self.env.reset()
            latent_code = latent_code[None]
            for step in range(self.max_episode_steps):
                s = self.obsfilt(s, update=False)
                s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
                with torch.no_grad():
                    _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                action = actions_tensor[0].cpu().numpy()
                s, r, done, _ = self.env.step(action)
                episode_reward += r
                if done:
                    break
                I = self.env.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, video_size)
                video_writer.write(I)
            video_writer.write(np.zeros([*video_size, 3], dtype=np.uint8))
            print(f"episode reward:{episode_reward:3.3f}")
        self.env.close()
        video_writer.release()
        cv2.destroyAllWindows()


class Plot(Base):
    def __init__(self, args, env, actor_critic, filename, obsfilt, vae_data):
        super(Plot, self).__init__(args, env, actor_critic, filename, obsfilt, vae_data)
        self.max_episode_steps = 1000 if args.env_name == 'Circles-v0' else 200

    def plot(self):
        {'Circles-v0': self._circles,
        'HalfCheetahVel-v0': self._halfcheetahvel,
        'AntDir-v0': self._ant,
         }.get(self.args.env_name, lambda: None)()

    def _circles(self):
        args, actor_critic, filename = self.args, self.actor_critic, self.filename

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

        count = None
        if args.vae_gail and args.vae_kmeans_clusters == -1:
            if args.env_name == 'Circles-v0':
                count = 5
            elif args.env_name == 'HalfCheetahVel-v0':
                count = 30
        latent_codes = generate_latent_codes(args, count=count, vae_data=self.vae_data, eval=True)

        # generate rollouts and plot them
        for j, latent_code in enumerate(latent_codes):
            latent_code = latent_code.unsqueeze(0)

            for i in range(self.max_episode_steps):
                # randomize latent code at each step in case of vanilla gail
                if args.vanilla:
                    latent_code = generate_latent_codes(args)
                # interacting with env
                with torch.no_grad():
                    # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                    obs = self.obsfilt(obs, update=False)
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

        plt.savefig(filename + '.png')
        plt.close()
        
    def _halfcheetahvel(self):
        device = self.args.device
        filename = self.filename

        num_codes, num_repeats = 50, 1

        if self.args.vae_gail:
            # 2100 x 1   or   2100 x 20
            latent_codes = self.vae_mus
            # 30 x 70 x 1 x 1   or   30 x 70 x 1 x 20
            latent_codes = latent_codes.reshape(30, 70, 1, -1)

            # latent_codes = latent_codes[:, :num_repeats]
            x = np.arange(30)
        else:
            cdf = np.linspace(.1, .9, num_codes)
            m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            # num_codes
            latent_codes = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(device)
            # num_codes x num_repeats x 1 x 1
            latent_codes = latent_codes[:, None, None, None].expand(-1, num_repeats, -1, -1)
            x = cdf

        vel_mean = []
        vel_std = []

        for j, latent_code_group in enumerate(latent_codes):
            vels = []
            for latent_code in latent_code_group:
                s = self.env.reset()
                for step in range(self.max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=device)[None]
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
        plt.plot(x, vel_mean, marker='o', color='r')
        plt.fill_between(x, vel_mean-vel_std, vel_mean+vel_std, alpha=0.2)
        plt.savefig(f'{filename}.png')
        plt.close()

        plt.figure()
        plt.hist(vel_mean, bins=np.linspace(1.5, 3, 10))
        plt.savefig(f'{filename}_hist.png')
        plt.close()

    def _ant(self):
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
            num_repeats = 3
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
                plt.plot(xpos[:, 0], xpos[:, 1], color=('b' if args.continuous else plt.cm.Dark2.colors[j]))
        plt.plot([0], [0], marker='o', markersize=3, color='k')
        plt.axis('off')
        plt.axis('equal')
        plt.savefig(f'{args.filename}.png')
        plt.close()
        self.env.close()


class Benchmark(Base):
    def __init__(self, args, env, actor_critic, filename, obsfilt, vae_data):
        super(Benchmark, self).__init__(args, env, actor_critic, filename, obsfilt, vae_data)
        self.h5 = h5py.File(os.path.join(args.results_dir, 'rewards.h5'), 'a')

    def collect_rewards(self, group):
        """
        Creates matrix of rewards of latent codes vs radii, find the best correspondence between the codes and the radii, and store the results.

        Returns:
            all_mode_rewards_mean: numpy array of shape [latent_dim x latent_dim] containing mean reward collected in a trajectory
            all_mode_rewards_std: numpy array of shape [latent_dim x latent_dim] containing std of rewards collected among trajectories
        """

        args = self.args
        device = args.device

        num_rewards = args.vae_kmeans_clusters if args.vae_gail else args.latent_dim
        assert num_rewards <= 6, 'all permutations of too many dimensions of latent codes is prohibitively costly'

        trajs_per_mode = 10
        max_episode_steps = 1000 if args.env_name == 'Circles-v0' else 200  # circles --> 1000 // mujoco --> 200

        if group == 'expert':
            if args.env_name != 'Circles-v0' or 'expert' in self.h5:
                return
            from gym_sog.envs.circles_expert import CirclesExpert
            circles_expert = CirclesExpert(self.args)

        all_mode_rewards_mean, all_mode_rewards_std = [], []
        for i, latent_code in enumerate(generate_latent_codes(args, vae_data=self.vae_data, eval=True)):
            latent_code = latent_code[None]
            all_traj_rewards = []
            for _ in range(trajs_per_mode):
                obs = self.env.reset()
                traj_rewards = np.zeros(num_rewards)
                for step in range(max_episode_steps):
                    if group == 'expert':
                        action = circles_expert.policy(obs, args.radii[i])
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

            rew_mean, rew_std = np.stack(all_mode_rewards_mean), np.stack(all_mode_rewards_std)

            # Record rewards according to best correspondence of latent codes and actual modes
            max_reward, best_mean, best_std = -np.inf, None, None
            for perm_mean, perm_std in zip(permutations(rew_mean), permutations(rew_std)):
                tmp = np.array(perm_mean).trace()
                if tmp > max_reward:
                    max_reward = tmp
                    best_mean = perm_mean
                    best_std = perm_std

            d = {'mean': np.diag(np.array(best_mean)), 'std': np.diag(np.array(best_std))}
            self.store(group, d)

    def collect_mutual_info(self, group):
        args = self.args
        device = args.device

        num_codes, num_repeats = 50, 1
        max_episode_steps = 200

        if args.vae_gail:
            # 2100 x 1   or   2100 x 20
            latent_codes = self.vae_mus
            # 30 x 70 x 1 x 1   or   30 x 70 x 1 x 20
            latent_codes = latent_codes.reshape(30, 70, 1, -1)

            # latent_codes = latent_codes[:, :num_repeats]
            x = np.arange(30)[:, None]
        else:
            cdf = np.linspace(.1, .9, num_codes)
            m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

            # num_codes
            latent_codes = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(device)
            # num_codes x num_repeats x 1 x 1
            latent_codes = latent_codes[:, None, None, None].expand(-1, num_repeats, -1, -1)
            x = cdf[:, None]

        if args.env_name != 'HalfCheetahVel-v0':
            raise NotImplementedError

        vel_mean = []
        vel_std = []

        for j, latent_code_group in enumerate(latent_codes):
            vels = []
            for k,latent_code in enumerate(latent_code_group):
                print(j,k )
                s = self.env.reset()
                for step in range(max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s, r, done, infos = self.env.step(action)
                    vels.append(infos['forward_vel'])
            vel_mean.append(np.mean(vels))
            vel_std.append(np.std(vels))
        self.env.close()

        vel_mean, vel_std = np.array(vel_mean), np.array(vel_std)
        self.store(group, {'mutual_info': mutual_info_regression(x, vel_mean)})

    def store(self, group, d):
        if group in self.h5:
            del self.h5[group]
        self.h5.create_group(group)
        for k, v in d.items():
            self.h5[group][k] = v


def plot_env(args, actor_critic, obsfilt, epoch, vae_data=None):
    filename = os.path.join(args.results_dir, str(epoch))

    if args.env_name == 'Circles-v0':
        env = None
    elif args.mujoco:
        import rlkit
        kwargs = {}
        if args.env_name == 'AntDir-v0':
            if args.vae_gail and args.vae_kmeans_clusters > 0:
                kwargs['n_tasks'] = args.vae_kmeans_clusters
            elif not args.continuous:
                kwargs['n_tasks'] = args.latent_dim
            else:
                raise NotImplementedError
        env = gym.make(args.env_name, **kwargs)
    else:
        raise NotImplementedError
    Plot(args, env, actor_critic, filename, obsfilt, vae_data).plot()


def play_env(args, actor_critic, obsfilt, epoch, vae_data=None):
    filename = os.path.join(args.results_dir, str(epoch))
    if args.mujoco:
        import rlkit
        env = gym.make(args.env_name)
        Play(args, env, actor_critic, filename, obsfilt, vae_data).play()
    else:
        raise NotImplementedError


def benchmark_env(args, actor_critic, obsfilt, epoch, vae_data=None):
    filename = os.path.join(args.results_dir, str(epoch))

    if args.env_name == 'Circles-v0':
        import gym_sog
        env = gym.make(args.env_name, args=args)
    elif args.mujoco:
        import rlkit
        kwargs = {}
        if args.env_name == 'AntDir-v0':
            if args.vae_gail and args.vae_kmeans_clusters > 0:
                kwargs['n_tasks'] = args.vae_kmeans_clusters
            elif not args.continuous:
                kwargs['n_tasks'] = args.latent_dim
            else:
                raise NotImplementedError
        env = gym.make(args.env_name, **kwargs)
    else:
        raise NotImplementedError

    benchmark = Benchmark(args, env, actor_critic, filename, obsfilt, vae_data)
    if args.env_name == 'HalfCheetahVel-v0':
        benchmark.collect_mutual_info(str(epoch))
    else:
        benchmark.collect_rewards(str(epoch))
