import glob
import torch
import torch.nn as nn
import gym

import sys, os, inspect
import argparse

################################## Temporary ##################################
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
# sys.path.insert(0, os.path.dirname(parentdir))
# from a2c_ppo_acktr.algo.vae_model_clean import EncoderRNN
###############################################################################

from a2c_ppo_acktr.envs import VecNormalize
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import visualize_env
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize, MujocoPlay, generate_latent_codes

import matplotlib.pyplot as plt

import numpy as np

from utilities import create_dir

# import cv2
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)

## from https://github.com/ArashVahabpour/sog-gail/blob/master/a2c_ppo_acktr/utils.py#L127
def visualize_env(args, actor_critic, epoch, obsfilt, num_steps=1000):
    plt.figure(figsize=(10, 20))
    plt.set_cmap("gist_rainbow")

    # plotting the actual circles
    if args.env_name == "Circles-v0":
        for r in args.radii:
            t = np.linspace(0, 2 * np.pi, 200)
            plt.plot(r * np.cos(t), r * np.sin(t) + r, color="#d0d0d0")
    else:
        raise NotImplementedError

    max_r = np.max(np.abs(args.radii))
    plt.axis("equal")
    plt.axis("off")
    plt.xlim([-1.5 * max_r, 1.5 * max_r])
    plt.ylim([-3 * max_r, 3 * max_r])

    # preparing the environment
    device = next(actor_critic.parameters()).device
    if args.env_name == "Circles-v0":
        import gym_sog

        env = gym.make(args.env_name, args=args)
    else:
        env = gym.make(args.env_name)
    obs = env.reset()

    create_dir(args.results_dir)
    filename = os.path.join(args.results_dir, str(epoch))

    if args.mujoco:
        MujocoPlay(args, env, actor_critic, filename, obsfilt).evaluate()
    else:
        # generate rollouts and plot them
        for j, latent_code in enumerate(torch.eye(args.latent_dim, device=device)):
            latent_code = latent_code.unsqueeze(0)

            for i in range(num_steps):
                # randomize latent code at each step in case of vanilla gail
                if args.vanilla:
                    latent_code = generate_latent_codes(args)
                # interacting with env
                with torch.no_grad():
                    # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                    obs = obsfilt(obs, update=False)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[
                        None
                    ]
                    actions_tensor = actor_critic.select_action(
                        obs_tensor, latent_code, True
                    )
                    action = actions_tensor[0].cpu().numpy()
                obs, _, _, _ = env.step(action)

            # plotting the trajectory
            plt.plot(
                env.loc_history[:, 0],
                env.loc_history[:, 1],
                color=plt.cm.Dark2.colors[j],
            )
            if args.vanilla:
                break  # one trajectory in vanilla mode is enough. if not, then rollout for each separate latent code
            else:
                obs = env.reset()

        env.close()

        plt.savefig(filename + ".png")
        plt.close()


### preparing
### export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/arash/.mujoco/mujoco200/bin
if __name__ == "__main__":
    trained_model_dir = "/mnt/SSD4/tmp_exp_gail/vae_bc/final_ckp"

    device = "cuda:0"
    args = get_args()
    args.is_train = False
    print(args)

    data_name = "circle"
    args.env_name == "Circles-v0"

    checkpoint_path = os.path.join(
        trained_model_dir, data_name, "checkpoints/bestvae_bc_model.pth"
    )

    load_path = os.path.join(args.save_dir, args.name)
    envs = make_vec_envs(
        args.env_name, args.seed, 1, args.gamma, args.log_dir, args.device, False, args
    )

    ob_rms = get_vec_normalize(envs).ob_rms
    obsfilt = get_vec_normalize(envs)._obfilt

    policy_net = torch.load(checkpoint_path)["state_dict_decoder"].to(device)

    visualize_env(args, policy_net, 0, obsfilt, num_steps=200)
