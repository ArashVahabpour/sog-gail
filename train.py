import os
import time
from collections import deque
from tqdm import tqdm
from itertools import cycle

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.algo.bc import BC
from a2c_ppo_acktr.algo.vae import VAE
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from eval import plot_env, benchmark_env


def main():
    args = get_args(is_train=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # managing dirs
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    expert_filename = args.expert_filename

    torch.set_num_threads(1)
    device = args.device

    envs = make_vec_envs(args.env_name, args.seed, 1,
                         args.gamma, args.log_dir, device, False, args)
    obsfilt = utils.get_vec_normalize(envs)._obfilt

    if len(envs.observation_space.shape) != 1:
        raise NotImplementedError

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args)
    actor_critic.to(device)
    gail_input_dim = envs.observation_space.shape[0] + envs.action_space.shape[0]
    discr = gail.Discriminator(gail_input_dim, 128, args)
    if args.infogail:
        posterior = gail.Posterior(gail_input_dim, 128, args)
    else:
        posterior = None

    agent = algo.PPO(
        actor_critic,
        args,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    bc_filename, vae_filename = [args.save_filename.format(s) for s in ('pretrain', 'vae_modes')]

    if args.vae_gail:
        if os.path.exists(vae_filename):
            vae_data = torch.load(vae_filename, map_location=device)
        else:
            vae = VAE(args, expert_filename).to(device)
            vae_data = vae.recover_modes()
            torch.save(vae_data, vae_filename)
    else:
        vae_data = [None] * 4
    vae_mus, _, _, vae_codes_all = vae_data

    # raise ValueError
    if not args.no_pretrain:
        BC(agent, bc_filename, expert_filename, args, obsfilt, vae_codes_all).pretrain(envs)
        # plot_env(args, actor_critic, obsfilt, 'pretrain', vae_data=vae_data)

    if len(envs.observation_space.shape) != 1:
        raise NotImplementedError

    expert_dataset = gail.ExpertDataset(expert_filename, num_traj=None, subsample_frequency=20, vae_modes=vae_codes_all)
    drop_last = len(expert_dataset) > args.gail_batch_size
    gail_train_loader = torch.utils.data.DataLoader(
        dataset=expert_dataset,
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=drop_last)

    sog_expert_dataset = gail.ExpertDataset(expert_filename, num_traj=None, subsample_frequency=20,
                                            sog_expert=True, args=args)
    sog_train_loader = torch.utils.data.DataLoader(
        dataset=sog_expert_dataset,
        batch_size=args.gail_batch_size,
        shuffle=(not args.shared),
        drop_last=True)
    sog_train_loader = cycle(sog_train_loader)

    rollouts = RolloutStorage(args.num_steps, 1,
                              envs.observation_space.shape, envs.action_space,
                              args.latent_dim)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    done = [True]
    latent_code = None

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps
    for j in tqdm(range(num_updates)):
        # decrease learning rate linearly
        utils.update_linear_schedule(
            agent.optimizer, j, num_updates,
            args.lr)

        ### sample trajectories
        for step in range(args.num_steps):
            # Update latent code
            if args.vanilla or done[0]:
                latent_code = utils.generate_latent_codes(args, 1, vae_data)

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(rollouts.obs[step], latent_code)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, latent_code, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.latent_codes[-1]).detach()

        if j >= 10:
            envs.venv.eval()

        ### update discriminator
        gail_epoch = args.gail_epoch
        if j < 10:
            gail_epoch = 100  # Warm up
        for _ in range(gail_epoch):
            discr.update(gail_train_loader, rollouts, obsfilt)

        ### update posterior
        if args.infogail:
            posterior.update(rollouts)

        ### update agent
        for step in range(args.num_steps):
            # discriminator reward
            rollouts.rewards[step] = discr.predict_reward(
                rollouts.obs[step],
                rollouts.actions[step],
                rollouts.latent_codes[step] if args.vae_gail else None,
                args.gamma,
                rollouts.masks[step])

            # infogail reward
            if args.infogail:
                rollouts.rewards[step] += args.infogail_coef * posterior.predict_reward(
                    rollouts.obs[step], rollouts.latent_codes[step], rollouts.actions[step],
                    args.gamma, rollouts.masks[step])

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        value_loss, action_loss, dist_entropy, sog_loss = agent.update(rollouts, sog_train_loader, obsfilt)

        rollouts.after_update()

        ### save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            torch.save([
                actor_critic,
                discr,
                posterior if args.infogail else None,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], args.save_filename.format(j))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, "
                "dist entropy loss {:.1f}, value loss {:.1f}, action loss {:.1f}{}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss,
                        ", sog loss {:.5f}".format(sog_loss) if args.sog_gail else ""))

        if j % args.result_interval == 0:
            ## visualize a sample trajectory
            plot_env(args, actor_critic, obsfilt, j, vae_data=vae_data)
            # from eval import benchmark_env
            # benchmark_env(args, actor_critic, obsfilt, j, vae_data=vae_data)
            # print('PLOT DISABLED')
            pass

if __name__ == "__main__":
    main()
