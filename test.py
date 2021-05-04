import torch
import os

from a2c_ppo_acktr.arguments import get_args
from eval import plot_env, play_env, benchmark_env
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize


# initialize
args = get_args(is_train=False)
device = args.device

for epoch in range(0, int(args.num_env_steps) // args.num_steps, args.save_interval):
    print(epoch)

    if args.vae_gail:
        vae_filename = args.save_filename.format('vae_modes')
        vae_data = torch.load(vae_filename, map_location=device)
    else:
        vae_data = None, None, None

    envs = make_vec_envs(args.env_name, args.seed, 1,
                         args.gamma, args.log_dir, device, False, args)
    ob_rms = get_vec_normalize(envs).ob_rms
    obfilt = get_vec_normalize(envs)._obfilt

    load_path = os.path.join(args.save_filename.format(epoch))
    actor_critic, _, _, saved_ob_rms = torch.load(load_path, map_location=args.device)

    args.device = list(actor_critic.base.parameters())[0].device
    ob_rms.mean, ob_rms.var, ob_rms.count = saved_ob_rms.mean, saved_ob_rms.var, saved_ob_rms.count

    os.makedirs(args.results_dir, exist_ok=True)

    # generate a rollout and visualize
    test_task = {'benchmark': benchmark_env, 'plot': plot_env, 'play': play_env}.get(args.test_task, None)
    if test_task is None:
        raise NotImplementedError(f'test task {args.test_task} is invalid')
    test_task(args, actor_critic, obfilt, epoch, vae_data=vae_data)
