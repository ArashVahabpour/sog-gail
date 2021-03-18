import torch
import os

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import benchmark_env#, plot_ant_expert
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize


# initialize
args = get_args(is_train=False)

for epoch in range(0, int(args.num_env_steps) // args.num_steps, args.save_interval):
    print(epoch)
    load_path = os.path.join(args.save_filename.format(epoch))
    envs = make_vec_envs(args.env_name, args.seed, 1,
                         args.gamma, args.log_dir, args.device, False, args)
    ob_rms = get_vec_normalize(envs).ob_rms
    obfilt = get_vec_normalize(envs)._obfilt
    actor_critic, _, _, saved_ob_rms = torch.load(load_path, map_location=args.device)
    args.device = list(actor_critic.base.parameters())[0].device
    ob_rms.mean, ob_rms.var, ob_rms.count = saved_ob_rms.mean, saved_ob_rms.var, saved_ob_rms.count

    os.makedirs(args.results_dir, exist_ok=True)

    # generate a rollout and visualize
    benchmark_env(args, actor_critic, obfilt, epoch)
