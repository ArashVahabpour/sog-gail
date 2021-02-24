import torch
import os

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import visualize_env
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize


# initialize
args = get_args(is_train=False)

envs = make_vec_envs(args.env_name, args.seed, 1,
                     args.gamma, args.log_dir, args.device, False, args)
ob_rms = get_vec_normalize(envs).ob_rms
obfilt = get_vec_normalize(envs)._obfilt
actor_critic, _, _, saved_ob_rms = torch.load(os.path.join(args.save_filename.format(args.env_name, args.which_epoch)),
                                              map_location=args.device)
args.device = list(actor_critic.base.parameters())[0].device
ob_rms.mean, ob_rms.var, ob_rms.count = saved_ob_rms.mean, saved_ob_rms.var, saved_ob_rms.count

os.makedirs(args.results_dir, exist_ok=True)

# generate a rollout and visualize
visualize_env(args, actor_critic, obfilt, args.which_epoch)
