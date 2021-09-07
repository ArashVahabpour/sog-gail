##### DRAWS ACTION ARROWS
import matplotlib.pyplot as plt
from numpy.linalg import norm
import torch
from a2c_ppo_acktr import utils
import os

from a2c_ppo_acktr.arguments import get_args
from eval import plot_env, play_env, benchmark_env
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize

args = get_args(is_train=False)
log_dir = os.path.expanduser(args.log_dir)
eval_log_dir = log_dir + "_eval"
utils.cleanup_log_dir(log_dir)
utils.cleanup_log_dir(eval_log_dir)
expert_filename = args.expert_filename if args.expert_filename else 'trajs_{}.pt'.format(
    args.env_name.split('-')[0].lower())
expert_filename = os.path.join(args.gail_experts_dir, expert_filename)


envs = make_vec_envs(args.env_name, args.seed, 1,
                     args.gamma, args.log_dir, args.device, False, args)
ob_rms = get_vec_normalize(envs).ob_rms
obfilt = get_vec_normalize(envs)._obfilt
load_path = '/mnt/SSD3/arash/sog-gail/trained_models/circles/sog-pretrain-10x-stronger/Circles-v0_pretrain.pt'
actor_critic, _, _, saved_ob_rms = torch.load(load_path, map_location=args.device)

args.device = list(actor_critic.base.parameters())[0].device
ob_rms.mean, ob_rms.var, ob_rms.count = saved_ob_rms.mean, saved_ob_rms.var, saved_ob_rms.count

plt.figure(figsize=(10, 30))
for traj_mode in range(3):
    states = torch.load('./gail_experts/trajs_circles.pt')['states'][traj_mode].numpy()
    plt.subplot(3, 1, traj_mode + 1)
    for j, latent_code in enumerate(torch.eye(3, device=args.device)[:, None, :]):
        for i in range(0, 100, 8):
            s = states[i]
            with torch.no_grad():
                # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                obs = obfilt(s, update=False)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)[None]
                _, actions_tensor, _ = actor_critic.act(obs_tensor, latent_code, deterministic=True)
                action = actions_tensor[0].cpu().numpy()
                dx, dy = action/norm(action) * 10
                x, y = s[-2:]
                head_width = 1
                plt.arrow(x, y, dx, dy, color=plt.cm.Dark2.colors[j], head_width=head_width, head_length=head_width/2)

plt.savefig('/home/arash/Desktop/a.png')
