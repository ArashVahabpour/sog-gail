import torch
import os
from torch.distributions import Normal
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import permutations
import h5py
from sklearn.feature_selection import mutual_info_regression
import matplotlib.ticker as ticker
from a2c_ppo_acktr.envs import FetchWrapper  #TODO  remove fetch and add meta-world instead
from a2c_ppo_acktr.utils import load_expert


#TODO remove any 'fetch' related thing from the repo

from a2c_ppo_acktr.utils import generate_latent_codes

class Base:
    def __init__(self, args, env, actor_critic, filename, obsfilt, vae_data):
        if args.fetch_env:
            self.env = FetchWrapper(env)
        else:
            self.env = env
        self.actor_critic = actor_critic
        self.args = args
        self.obsfilt = obsfilt
        self.filename = filename
        self.vae_data = vae_data
        self.vae_mus = vae_data[0]
        assert not args.vanilla, 'Vanilla GAIL benchmarking not implemented'
        self.max_episode_steps = self._get_max_episode_steps()


    def resolve_latent_code(self, states, actions, i):
        def _get_sog(args, actor_critic):
            from a2c_ppo_acktr.algo.sog import OneHotSearch, BlockCoordinateSearch

            if args.latent_optimizer == 'bcs':
                SOG = BlockCoordinateSearch
            elif args.latent_optimizer == 'ohs':
                SOG = OneHotSearch
            else:
                raise NotImplementedError
            return SOG(actor_critic, args)

        device = self.args.device
        if self.args.sog_gail:
            sog = _get_sog(self.args, self.actor_critic)
            return sog.resolve_latent_code(torch.from_numpy(self.obsfilt(states[i].cpu().numpy(), update=False)).float().to(device), actions[i].to(device))[:1]
        elif self.args.vae_gail:
            return self.vae_mus[i]
        elif self.args.infogail:
            return generate_latent_codes(self.args, count=1, eval=True)
        else:
            raise NotImplementedError

    def _get_max_episode_steps(self):
        return {
            'Circles-v0': 1000,
            'AntDir-v0': 200,
            'HalfCheetahVel-v0': 200,
            'FetchReach-v0': 50,
            'HopperVel-v0': 1000,
            'Walker-v0': 1000,
            'HumanoidDir-v0': 1000,
        }.get(self.args.env_name, 200)

class Play(Base):
    def __init__(self, **kwargs):
        super(Play, self).__init__(**kwargs)
        if self.args.fetch_env:
            self.max_episode_steps = self.env.env._max_episode_steps
        else:
            max_episode_time = 10
            dt = kwargs['env'].dt
            self.max_episode_steps = int(max_episode_time / dt)

    def play(self):
        args = self.args

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if args.fetch_env:
            video_size = (500, 500)
        else:
            video_size = (250, 250)
        video_writer = cv2.VideoWriter(f'{self.filename}.avi', fourcc, 1 / self.env.dt, video_size)

        s = self.env.reset()

        if (args.fetch_env or args.mujoco) and args.continuous and args.env_name != 'HalfCheetahVel-v0':
            expert = torch.load(args.expert_filename, map_location=args.device)

            count = 30 if args.fetch_env else 5  # for fetch, try 30 of the goals from the expert trajectories

            ####### recover expert embeddings #######
            expert_len = len(expert['states'])
            sample_idx = torch.randint(low=0, high=expert_len, size=(count,))
            states, actions, desired_goals = [expert.get(key, [None]*expert_len)[sample_idx] for key in ('states', 'actions', 'desired_goal')]  # only keep the trajectories specified by `sample_idx`
            latent_codes = [self.resolve_latent_code(states, actions, i) for i in range(len(states))]

        else:
            # if 'Humanoid' in args.env_name and args.continuous:
                # expert = load_expert(args.expert_filename, device=args.device)
                # ####### recover expert embeddings #######
                # count = 5
                # sample_idx = torch.randint(low=0, high=len(expert['states']), size=(count,))
                # states, actions = [expert[key][sample_idx] for key in ('states', 'actions')]  # only keep the trajectories specified by `sample_idx`
                # latent_codes = [self.resolve_latent_code(torch.from_numpy(self.obsfilt(states.cpu().numpy(), update=False)).float().to(args.device), actions, i) for i in len(states)]

                # expert = load_expert(args.expert_filename, device=args.device)
                # ####### recover expert embeddings #######
                # latent_codes = list()
                # possible_angles = torch.unique(expert['angles'])
                # sog = self._get_sog()
                # for angle in possible_angles:
                #     sample_idx = expert['angles'] == angle
                #     states, actions = [expert[key][sample_idx] for key in ('states', 'actions')]  # only keep the trajectories specified by `sample_idx`
                #     from tqdm import tqdm
                #     mode_list = []
                #     for (traj_states, traj_actions) in tqdm(zip(states, actions)):
                #         mode_list.append(sog.resolve_latent_code(torch.from_numpy(self.obsfilt(traj_states.cpu().numpy(), update=False)).float().to(args.device), traj_actions)[0])
                #     latent_codes.append(torch.stack(mode_list).mean(0))
            count = None
            if args.vae_gail and args.env_name == 'HalfCheetahVel-v0':
                count = 30
            latent_codes = generate_latent_codes(args, count=count, vae_data=self.vae_data, eval=True)

        for j, latent_code in enumerate(latent_codes):
            latent_code = latent_code
            episode_reward = 0
            if args.fetch_env:
                self.env.set_desired_goal(desired_goals[j].cpu().numpy())
                self.env._max_env_steps = 100
                # print(desired_goals[j])
            print(f'traj #{j+1}/{len(latent_codes)}')
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
            if args.fetch_env:
                pass
                # achieved_goal = self.env.unwrapped.sim.data.get_site_xpos("robot0:grip").copy()
                # success = self.env.unwrapped._is_success(achieved_goal, self.env.unwrapped.goal)
                # print('success' if success else 'failed')
            else:
                print(f"episode reward:{episode_reward:3.3f}")
            s = self.env.reset()
            video_writer.write(np.zeros([*video_size, 3], dtype=np.uint8))
        self.env.close()
        video_writer.release()
        cv2.destroyAllWindows()


class Plot(Base):
    def plot(self):
        return {'Circles-v0': self._circles_ellipses,
        'Ellipses-v0': self._circles_ellipses,
        'HalfCheetahVel-v0': self._halfcheetahvel,
        'AntDir-v0': self._ant,
        'FetchReach-v1': self._fetch,
        'Walker2dVel-v0': self._walker_hopper,
        'HopperVel-v0': self._walker_hopper,
        'HumanoidDir-v0': self._humanoid,
         }.get(self.args.env_name, lambda: None)()
        
    def _circles_ellipses(self):
        args, actor_critic, filename = self.args, self.actor_critic, self.filename

        fig = plt.figure(figsize=(2, 3), dpi=300)
        plt.set_cmap('gist_rainbow')
        # plotting the actual circles/ellipses
        if args.env_name == 'Circles-v0':
            for r in args.radii:
                t = np.linspace(0, 2 * np.pi, 200)
                plt.plot(r * np.cos(t), r * np.sin(t) + r, color='#d0d0d0')
        elif args.env_name == 'Ellipses-v0':
            for rx, ry in np.array(args.radii).reshape(-1, 2):
                t = np.linspace(0, 2 * np.pi, 200)
                plt.plot(rx * np.cos(t), ry * np.sin(t) + ry, color='#d0d0d0')
        max_r = np.max(np.abs(args.radii))
        plt.axis('equal')
        # plt.axis('off')
        # Turn off tick labels
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.xlim([-1 * max_r, 1 * max_r])
        plt.ylim([-1.5 * max_r, 2.5 * max_r])

        import gym_sog
        env = gym.make(args.env_name, args=args)
        obs = env.reset()

        device = next(actor_critic.parameters()).device

        count = 3
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
    
    def _fetch(self):
        args = self.args
        actor_critic = self.actor_critic
        obsfilt = self.obsfilt
        filename = self.filename

        # TODO   
        expert = load_expert(args.expert_filename)
        count = 100  # how many number of expert trajectories
        sample_idx = np.random.randint(low=0, high=len(expert['states']), size=(count,))
        # sample_idx = np.arange(len(expert['states']))
        states, actions, desired_goals = [expert[key][sample_idx] for key in ('states', 'actions', 'desired_goal')]  # only keep the trajectories specified by `sample_idx`
        
        # init env
        env = gym.make(args.env_name)    

        # init plots
        matplotlib.rcParams['legend.fontsize'] = 10
        fig = plt.figure(figsize=(3,3), dpi=300)
        ax = fig.gca(projection='3d')


        normalize = lambda x: (x - np.array([1.34183226, 0.74910038, 0.53472284]))/.15  # map the motion range to the unit cube
        s = self.env.reset()
        for i in range(len(states)):
            env.unwrapped.goal = desired_goals[i].cpu().numpy()
            latent_code = self.resolve_latent_code(states, actions, i)
            
            achieved_goals = np.zeros([env._max_episode_steps, 3])

            s = env.reset()['observation']
            
            for step in range(env._max_episode_steps):
                s = obsfilt(s, update=False)
                s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]

                with torch.no_grad():
                    _, actions_tensor, _ = actor_critic.act(s_tensor, latent_code, deterministic=True)
                action = actions_tensor[0].cpu().numpy()
                obs, _, _, _ = env.step(action)
                s = obs['observation']
                achieved_goals[step] = normalize(obs['achieved_goal'])
            ax.plot(*achieved_goals.T)

        if not args.infogail:
            ax.scatter(*normalize(desired_goals).T)
            
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        plt.savefig(f'{filename}.png')


    def _halfcheetahvel(self):
        device = self.args.device
        filename = self.filename

        if self.args.vae_gail:
            # 2100 x 1   or   2100 x 20
            latent_codes = self.vae_mus
            # 30 x 70 x 1 x 1   or   30 x 70 x 1 x 20
            latent_codes = latent_codes.reshape(30, 70, 1, -1)

            # latent_codes = latent_codes[:, :num_repeats]
            x = np.linspace(1.5, 3, 30)
        else:
            num_codes, num_repeats = 100, 30
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
            print(f'{j+1} / {len(latent_codes)}')
            vels = []
            for k, latent_code in enumerate(latent_code_group):
                print(f'\t - {k+1} / {len(latent_code_group)}')
                s = self.env.reset()
                for step in range(self.max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s, r, done, infos = self.env.step(action)
                    vels.append(infos['forward_vel'])

            # fix for the dataset slight offset of the dataset that begins from 1.539 instead of accurate 1.5 #TODO modify the dataset  instead
            rescale = lambda input, input_low, input_high, output_low, output_high: ((input - input_low) / (input_high - input_low)) * (output_high - output_low) + output_low
            vels = rescale(np.array(vels), 1.539, 3., 1.5, 3)

            vel_mean.append(np.mean(vels))
            vel_std.append(np.std(vels))
        self.env.close()

        vel_mean, vel_std = np.array(vel_mean), np.array(vel_std)
        plt.figure(figsize=(3.5, 7/5*1.85), dpi=300)
        plt.plot(x, vel_mean)#, marker='o', color='r')
        plt.fill_between(x, vel_mean-vel_std, vel_mean+vel_std, alpha=0.2)
        for bound in (1.5, 3):
            plt.axhline(bound, linestyle='--', c='0.5')
        plt.ylim([0,5])
        plt.savefig(f'{filename}.png')
        plt.close()

        plt.figure()
        plt.hist(vel_mean, bins=np.linspace(1.5, 3, 10))
        plt.savefig(f'{filename}_hist.png')
        plt.close()

    def _ant(self):
        args = self.args
        fig = plt.figure(figsize=(3,3), dpi=300)

        num_repeats = 3

        if args.vae_gail:
            all_codes = generate_latent_codes(args, vae_data=self.vae_data, eval=True)
        else:
            all_codes = torch.eye(args.latent_dim, device=args.device)

        # Turn off tick labels
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlim([-70, 70])
        ax.set_ylim([-70, 70])

        m = []
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
                m.append(np.max(np.abs(xpos)))
                ax.plot(xpos[:, 0], xpos[:, 1], color=(plt.cm.Dark2.colors[j]))
        ax.plot([0], [0], marker='o', markersize=3, color='k')
        if args.vae_gail or args.infogail:
            m = max(m) * (np.random.rand() * 3 + 1)
            ax.set_xlim([-m,m])
            ax.set_ylim([-m,m])
        plt.savefig(f'{self.filename}.png')
        plt.close()
        self.env.close()

    def _humanoid(self):
        args = self.args
        fig = plt.figure(figsize=(3,3), dpi=300)

        num_repeats = 3

        if args.vae_gail:
            all_codes = generate_latent_codes(args, vae_data=self.vae_data, eval=True)
        elif args.continuous:
            if args.sog_gail:
                latent_codes = []
                count = 3

                expert = load_expert(args.expert_filename, device=args.device)
                modes = expert['modes'].cpu().numpy()
                unique_modes = np.unique(modes)
                for i in range(args.num_clusters):
                    sample_idx = np.random.permutation(np.argwhere(modes==unique_modes[i]).squeeze())[:count]
                    states, actions = [expert[key][sample_idx] for key in ('states', 'actions')]  # only keep the trajectories specified by `sample_idx`
                    latent_codes.extend([self.resolve_latent_code(states, actions, i) for i in range(len(states))])
            
                all_codes = torch.cat(latent_codes)

            else:
                raise NotImplementedError
        else:
            all_codes = torch.eye(args.latent_dim, device=args.device)

        # Turn off tick labels
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlim([-300, 300])
        ax.set_ylim([-300, 300])

        m = []
        for j, latent_code in enumerate(all_codes):
            latent_code = latent_code[None]
            for _ in range(num_repeats):
                s = self.env.reset()
                xvel = []
                for step in range(self.max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s, r, done, infos = self.env.step(action)
                    xvel.append(s[22:24])
                xpos = np.cumsum(np.array(xvel), axis=0)
                m.append(np.max(np.abs(xpos)))
                ax.plot(xpos[:, 0], xpos[:, 1], color=(plt.cm.Dark2.colors[j//count if (args.sog_gail and args.continuous) else j]))
        ax.plot([0], [0], marker='o', markersize=3, color='k')
        if args.vae_gail or args.infogail:
            # TODO
            m = max(m) * (np.random.rand() * 3 + 1)
            ax.set_xlim([-m,m])
            ax.set_ylim([-m,m])
        plt.savefig(f'{self.filename}.png')
        plt.close()
        self.env.close()


    def _walker_hopper(self):
        args = self.args

        device = args.device
        filename = self.filename

        count_per_mode = 3

        if args.continuous:
            expert = load_expert(args.expert_filename)
            
        if self.args.vae_gail:
            modes = load_expert(args.expert_filename)['modes']
            latent_codes = self.vae_mus
            # group latent codes into groups
            latent_codes = [latent_codes[modes==m][:count_per_mode] for m in np.unique(modes)]
        elif args.continuous:
            if args.sog_gail:
                latent_codes = []

                expert = load_expert(args.expert_filename, device=args.device)
                modes = expert['modes'].cpu().numpy()
                unique_modes = np.unique(modes)
                for i in range(args.num_clusters):
                    sample_idx = np.random.permutation(np.argwhere(modes==unique_modes[i]).squeeze())[:count_per_mode]
                    states, actions = [expert[key][sample_idx] for key in ('states', 'actions')]  # only keep the trajectories specified by `sample_idx`
                    latent_codes.append(torch.cat([self.resolve_latent_code(states, actions, i) for i in range(len(states))]))
        else:
            latent_codes = torch.eye(args.latent_dim, device=args.device).unsqueeze(1).expand(-1, count_per_mode, -1)


        #>>>>>>                                                  <<<<<<<
        #>>>>>>>>>>>>>> find x, latent codes in all cases <<<<<<<<<<<<<<
        #>>>>>>                                                  <<<<<<<

        vels_all = []

        for j, latent_code_group in enumerate(latent_codes):
            vels_mode = []
            for k, latent_code in enumerate(latent_code_group):
                vels = []
                latent_code = latent_code[None]
                s = self.env.reset()
                for step in range(self.max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    action = actions_tensor[0].cpu().numpy()
                    s, r, done, infos = self.env.step(action)
                    vels.append(infos['forward_vel'])
                vels_mode.append(vels)
            vels_all.append(vels_mode)
        self.env.close()

        plt.figure(figsize=(3, 2), dpi=300)
        for i, vels_mode in enumerate(vels_all):
            for j, vels in enumerate(vels_mode):
                plt.plot(np.arange(len(vels)), vels, color=plt.cm.Dark2.colors[i])
        plt.xlim([0, self.max_episode_steps - 1])
        plt.ylim([-3,3])
        plt.savefig(f'{filename}.png')
        plt.close()


class Benchmark(Base):
    def collect_rewards(self, group):
        """
        Creates matrix of rewards of latent codes vs radii, find the best correspondence between the codes and the radii, and store the results.

        Returns:
            all_mode_rewards_mean: numpy array of shape [latent_dim x latent_dim] containing mean reward collected in a trajectory
            all_mode_rewards_std: numpy array of shape [latent_dim x latent_dim] containing std of rewards collected among trajectories
        """

        args = self.args
        device = args.device

        num_modes = args.vae_num_modes if args.vae_gail else args.latent_dim
        assert num_modes <= 6, 'all permutations of too many dimensions of latent codes is prohibitively costly, try implementing hungarian method'

        trajs_per_mode = 10

        if group == 'expert':
            return
            # if args.env_name not in {'Circles-v0' or 'Ellipses-v0'} or 'expert' in self.h5:
            #     return
            # from gym_sog.envs.circles_expert import CirclesExpert
            # circles_expert = CirclesExpert(self.args)

        all_mode_rewards_mean, all_mode_rewards_std = [], []
        all_codes = generate_latent_codes(args, vae_data=self.vae_data, eval=True)
        for i, latent_code in enumerate(all_codes):
            print(group, i)
            latent_code = latent_code[None]
            all_traj_rewards = []
            for _ in range(trajs_per_mode):
                print(_)
                obs = self.env.reset()
                traj_rewards = np.zeros(num_modes)
                for step in range(self.max_episode_steps):
                    # if group == 'expert':
                    #     action = circles_expert.policy(obs, args.radii[i])
                    # else:
                    if True:
                        with torch.no_grad():
                            # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                            obs = self.obsfilt(obs, update=False)
                            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[None]
                            _, actions_tensor, _ = self.actor_critic.act(obs_tensor, latent_code, deterministic=True)
                            if np.random.rand() > args.test_perturb_amount:
                                action = actions_tensor[0].cpu().numpy()
                            else:
                                action = self.env.action_space.sample()

                    obs, _, _, infos = self.env.step(action)
                    traj_rewards += np.array(infos['rewards'])
                # each element: [num_modes,]
                all_traj_rewards.append(traj_rewards)

            # [trajs_per_mode, num_modes]
            all_traj_rewards = np.stack(all_traj_rewards)

            # each element: [num_modes,]
            all_mode_rewards_mean.append(all_traj_rewards.mean(axis=0))
            all_mode_rewards_std.append(all_traj_rewards.std(axis=0))

        # [num_modes, num_modes]
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
        print(d)
        self.store(group, d)

    def collect_mutual_info(self, group):
        args = self.args
        device = args.device

        num_codes, num_repeats = 30, 1#70

        if args.vae_gail:
            # 2100 x 1   or   2100 x 20
            latent_codes = self.vae_mus
            # 30 x 70 x 1 x 1   or   30 x 70 x 1 x 20
            latent_codes = latent_codes.reshape(30, 70, 1, -1)

            x = np.arange(30)

        else:
            cdf = np.linspace(.1, .9, num_codes)
            m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

            # num_codes
            latent_codes = m.icdf(torch.tensor(cdf, dtype=torch.float32)).to(device)
            # num_codes x num_repeats x 1 x 1
            latent_codes = latent_codes[:, None, None, None].expand(-1, num_repeats, -1, -1)
            x = cdf

        if args.env_name != 'HalfCheetahVel-v0':
            raise NotImplementedError

        vel_mean = []
        vel_std = []

        all_vels = []
        for j, latent_code_group in enumerate(latent_codes):
            vels = []
            for k,latent_code in enumerate(latent_code_group):
                print(j, k)
                s = self.env.reset()
                for step in range(self.max_episode_steps):
                    s = self.obsfilt(s, update=False)
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=device)[None]
                    with torch.no_grad():
                        _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
                    if np.random.rand() > args.test_perturb_amount:
                        action = actions_tensor[0].cpu().numpy()
                    else:
                        action = self.env.action_space.sample()
                    s, r, done, infos = self.env.step(action)
                    vels.append(infos['forward_vel'])
            vel_mean.append(np.mean(vels))
            vel_std.append(np.std(vels))
            all_vels.append(vels)
        self.env.close()

        # [num_codes, max_episode_steps * num_repeats]
        all_vels = np.array(all_vels)
        # [num_codes, max_episode_steps * num_repeats]
        all_x = np.tile(x[:, None], (1, all_vels.shape[1]))

        # [num_codes * max_episode_steps, ]
        all_vels = all_vels.ravel()
        # [num_codes * max_episode_steps, ]
        all_x = all_x.reshape(len(all_vels), -1)
        mutual_info = mutual_info_regression(all_x, all_vels)

        vel_mean, vel_std = np.array(vel_mean), np.array(vel_std)
        self.store(group, {'vel_mean': vel_mean,
                           'vel_std': vel_std,
                           'all_x': all_x,
                           'all_vels': all_vels,
                           'mutual_info': mutual_info})
        print(f'--- {mutual_info} ---')


    def store(self, group, d):
        with h5py.File(os.path.join(self.args.results_dir, f'rewards_{self.args.test_perturb_amount}.h5'), 'a') as h5:
            if group in h5:
                del h5[group]
            h5.create_group(group)
            for k, v in d.items():
                h5[group][k] = v


class RobustnessTest(Benchmark):
    def robustness_test(self, epoch):
        args = self.args
        if args.env_name == 'Circles-v0':
            import gym_sog
            self.env = gym.make(args.env_name, args=args)
        elif args.mujoco:
            import rlkit
            kwargs = {}
            if args.env_name == 'AntDir-v0':
                if args.vae_gail and args.vae_num_modes > 0:
                    kwargs['n_tasks'] = args.vae_num_modes
                elif not args.continuous:
                    kwargs['n_tasks'] = args.latent_dim
                else:
                    raise NotImplementedError
            self.env = gym.make(args.env_name, **kwargs)
        else:
            raise NotImplementedError

        # if args.env_name == 'HalfCheetahVel-v0':
        #     self.collect_mutual_info(str(epoch))
        # else:
        #     self.collect_rewards(str(epoch))
        self._ant()

    def _ant(self):
        # """
        # There is no direct reference to this function anywhere in the repository, should be manually called
        # The purpose of this function is to plot the ant going to several directions
        # """
        args, filename = self.args, self.filename
        # plt.figure()

        if args.vae_gail:
            all_codes = generate_latent_codes(args, vae_data=self.vae_data, eval=True)
        else:
            all_codes = torch.eye(args.latent_dim, device=args.device)

        max_episode_steps = self.max_episode_steps * len(all_codes)
        #
        # # all_traj_rewards = []
        # # num_repeats = 10
        # # for i in range(num_repeats):
        # #     xpos = []
        # #     traj_rewards = np.zeros([6, 6])
        # #     s = self.env.reset()
        # #     for step in range(max_episode_steps):
        # #         code_idx = step // (max_episode_steps // len(all_codes))
        # #         latent_code = all_codes[code_idx]
        # #         latent_code = latent_code[None]
        # #         s = self.obsfilt(s, update=False)
        # #         s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
        # #         with torch.no_grad():
        # #             _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
        # #         action = actions_tensor[0].cpu().numpy()
        # #         s, r, done, infos = self.env.step(action)
        # #         xpos.append(infos['xpos'])
        # #         traj_rewards[code_idx] += np.array(infos['rewards'])
        # #     all_traj_rewards.append(traj_rewards)
        # #
        # # # [trajs_per_mode, num_modes, num_modes]
        # # all_traj_rewards = np.stack(all_traj_rewards)
        # #
        # # # each element: [num_modes, num_modes]
        # # rew_mean = all_traj_rewards.mean(axis=0)
        # # rew_std = all_traj_rewards.std(axis=0)
        # #
        # # # Record rewards according to best correspondence of latent codes and actual modes
        # # max_reward, best_mean, best_std = -np.inf, None, None
        # # for perm_mean, perm_std in zip(permutations(rew_mean), permutations(rew_std)):
        # #     tmp = np.array(perm_mean).trace()
        # #     if tmp > max_reward:
        # #         max_reward = tmp
        # #         best_mean = perm_mean
        # #         best_std = perm_std
        # #
        # # d = {'mean': np.diag(np.array(best_mean)).sum(), 'std': np.diag(np.array(best_std)).mean()}
        # # print(d)
        #
        # # xpos = np.array(xpos)
        # #     plt.plot(xpos[:, 0], xpos[:, 1], color=plt.cm.Dark2.colors[i])
        # # plt.plot([0], [0], marker='o', markersize=3, color='k')
        # # plt.axis('off')
        # # plt.axis('equal')
        # # plt.savefig(f'{filename}_robustness.png')
        # # plt.close()
        # # self.env.close()
        #
        #
        # xpos_all = []
        # num_repeats = 100
        # for i in range(num_repeats):
        #     print(i)
        #     xpos = []
        #     s = self.env.reset()
        #     for step in range(max_episode_steps):
        #         latent_code = all_codes[step // (max_episode_steps // len(all_codes))]
        #         latent_code = latent_code[None]
        #         s = self.obsfilt(s, update=False)
        #         s_tensor = torch.tensor(s, dtype=torch.float32, device=args.device)[None]
        #         with torch.no_grad():
        #             _, actions_tensor, _ = self.actor_critic.act(s_tensor, latent_code, deterministic=True)
        #         action = actions_tensor[0].cpu().numpy()
        #         s, r, done, infos = self.env.step(action)
        #         xpos.append(infos['xpos'])
        #     xpos = np.array(xpos)
        #     plt.plot(xpos[:, 0], xpos[:, 1])#, color=plt.cm.Dark2.colors[i])
        #     plt.plot([0], [0], marker='o', markersize=3, color='k')
        #     plt.axis('off')
        #     plt.axis('equal')
        #     plt.savefig(f'{filename}_robustness_{i}.png')
        #     plt.close()
        #     self.env.close()
        #
        #     xpos_all.append(xpos)
        #
        # np.save('b.pkl', np.stack(xpos_all))

        # assembling the best

        fig = plt.figure(figsize=(2, 2), dpi=500)

        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(18.00))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(18.00))
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlim([-20, 70*1.5*.9])
        ax.set_ylim([-20, 70*np.sqrt(3)*.9])


        xpos_all = np.load(f'a.pkl.npy')
        for i, fig_id in enumerate([53,54,67]):#32,86, 63,])::
            xpos = xpos_all[fig_id]
            plt.plot(xpos[:, 0], xpos[:, 1], color=plt.cm.Dark2.colors[i])
        plt.plot([0], [0], marker='o', markersize=3, color='k')
        # plt.axis('off')
        # plt.axis('equal')
        plt.savefig(f'robustness.pdf')
        plt.legend()

        plt.close()

def plot_env(args, actor_critic, obsfilt, epoch, vae_data=None):
    filename = os.path.join(args.results_dir, str(epoch))

    if args.env_name in {'Circles-v0', 'Ellipses-v0'}:
        env = None
    elif args.mujoco:
        import rlkit
        kwargs = {}
        if args.env_name == 'AntDir-v0':
            if args.vae_gail and args.vae_num_modes > 0:
                kwargs['n_tasks'] = args.vae_num_modes
            elif not args.continuous:
                kwargs['n_tasks'] = args.latent_dim
            else:
                raise NotImplementedError
        env = gym.make(args.env_name, **kwargs)
    else:
        raise NotImplementedError
    Plot(args=args, env=env, actor_critic=actor_critic, filename=filename, obsfilt=obsfilt, vae_data=vae_data).plot()


def play_env(args, actor_critic, obsfilt, epoch, vae_data=None):
    filename = os.path.join(args.results_dir, str(epoch))
    if args.mujoco:
        import rlkit
        env = gym.make(args.env_name)
        Play(args=args, env=env, actor_critic=actor_critic, filename=filename, obsfilt=obsfilt, vae_data=vae_data).play()
    else:
        raise NotImplementedError


def benchmark_env(args, actor_critic, obsfilt, epoch, vae_data=None):
    filename = os.path.join(args.results_dir, str(epoch))

    if args.env_name in {'Circles-v0', 'Ellipses-v0'}:
        import gym_sog
        env = gym.make(args.env_name, args=args)
    elif args.mujoco:
        import rlkit
        kwargs = {}
        if args.env_name == 'AntDir-v0':
            if args.vae_gail and args.num_clusters > 0:
                kwargs['n_tasks'] = args.num_clusters
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
        # benchmark.collect_rewards('expert')


def robustness_test(args, actor_critic, obsfilt, epoch, vae_data=None):
    filename = os.path.join(args.results_dir, str(epoch))
    if args.env_name in {'Circles-v0', 'Ellipses-v0'}:
        import gym_sog
        env = gym.make(args.env_name, args=args)
    elif args.mujoco:
        import rlkit
        kwargs = {}
        if args.env_name == 'AntDir-v0':
            if args.vae_gail and args.num_clusters > 0:
                kwargs['n_tasks'] = args.num_clusters
            elif not args.continuous:
                kwargs['n_tasks'] = args.latent_dim
            else:
                raise NotImplementedError
        env = gym.make(args.env_name, **kwargs)
    else:
        raise NotImplementedError
    RobustnessTest(args, env, actor_critic, filename, obsfilt, vae_data).robustness_test(epoch)
