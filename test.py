import torch
import os

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import visualize_env
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize


# initialize
args = get_args()
args.is_train = False

load_path = os.path.join(args.save_dir, args.name)
envs = make_vec_envs(args.env_name, args.seed, 1,
                     args.gamma, args.log_dir, args.device, False, args)
ob_rms = get_vec_normalize(envs)
actor_critic, _, _, saved_ob_rms = torch.load(os.path.join(load_path, f'{args.env_name}_{args.which_epoch}.pt'))
ob_rms.mean, ob_rms.var, ob_rms.count = saved_ob_rms.mean, saved_ob_rms.var, saved_ob_rms.count

os.makedirs(args.results_dir, exist_ok=True)

# generate a rollout and visualize
visualize_env(args, actor_critic, ob_rms._obfilt, args.which_epoch)
