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
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


def main():
    args = get_args()
    args.is_train = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # managing dirs
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    save_dir = os.path.join(args.save_dir, args.name)  # directory to store network weights
    save_filename = os.path.join(save_dir, '{}_{}.pt')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    expert_filename = os.path.join(args.gail_experts_dir, 'trajs_{}.pt'.format(args.env_name.split('-')[0].lower()))

    torch.set_num_threads(1)
    device = args.device

    envs = make_vec_envs(args.env_name, args.seed, 1,
                         args.gamma, args.log_dir, device, False, args)
    if len(envs.observation_space.shape) != 1:
        raise NotImplementedError

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args)
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    if args.bc_pretrain:
        bc_save_filename = save_filename.format(args.env_name, 'pretrain')
        BC(agent, bc_save_filename, expert_filename, args).pretrain()
        utils.visualize_env(args, actor_critic, 'pretrain')


if __name__ == "__main__":
    main()
