import torch
import os

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import visualize_env


# initialize
args = get_args()
args.is_train = False

load_path = os.path.join(args.save_dir, args.algo)
actor_critic, _ = torch.load(os.path.join(load_path, args.env_name + ".pt"))

# generate a rollout and visualize
visualize_env(args, actor_critic, 'last')
