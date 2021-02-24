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
###############################################################################
from a2c_ppo_acktr.algo.vae_model_clean import EncoderRNN
from a2c_ppo_acktr.algo.behavior_clone import MlpPolicyNet

from a2c_ppo_acktr.envs import VecNormalize
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import visualize_env
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize, MujocoPlay, generate_latent_codes
from a2c_ppo_acktr.algo.vae_model_clean import VAE_BC

import matplotlib.pyplot as plt

import numpy as np

from utilities import create_dir

# import cv2
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)

## from https://github.com/ArashVahabpour/sog-gail/blob/master/a2c_ppo_acktr/utils.py#L127
def visualize_env(args, actor_critic, obsfilt, epoch, num_steps=1000):
    plt.figure(figsize=(10, 20))
    plt.set_cmap("gist_rainbow")

    # plotting the actual circles
    if args.env_name == "Circles-v0":
        for r in args.radii:
            t = np.linspace(0, 2 * np.pi, 200)
            plt.plot(r * np.cos(t), r * np.sin(t) + r, color="#d0d0d0")
            max_r = np.max(np.abs(args.radii))
            plt.axis("equal")
            plt.axis("off")
            plt.xlim([-1.5 * max_r, 1.5 * max_r])
            plt.ylim([-3 * max_r, 3 * max_r])

    elif not args.mujoco:
        raise NotImplementedError

    # preparing the environment
    device = next(actor_critic.parameters()).device
    if args.env_name == "Circles-v0":
        import gym_sog

        env = gym.make(args.env_name, args=args)
    elif args.mujoco:
        import rlkit

        env = gym.make(args.env_name)
    obs = env.reset()

    create_dir(args.results_dir)
    filename = os.path.join(args.results_dir, str(epoch))

    if args.mujoco:
        mujoco_play = MujocoPlay(args, env, actor_critic, filename, obsfilt)
        if args.sog_gail and args.latent_optimizer == "bcs":
            mujoco_play.evaluate_continuous()
        else:
            mujoco_play.evaluate_discrete()

    else:
        # generate rollouts and plot them
        # for j, latent_code in enumerate(torch.eye(args.latent_dim, device=device)):
        for j, latent_code in enumerate(torch.eye(args.latent_dim, device=device)):
            latent_code = latent_code.unsqueeze(0)

            for i in range(num_steps):
                # randomize latent code at each step in case of vanilla gail
                if args.vanilla:
                    latent_code = generate_latent_codes(args)
                # interacting with env
                with torch.no_grad():
                    # an extra 0'th dimension is because actor critic works with "environment vectors" (see the training code)
                    # obs = obsfilt(obs, update=False)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)[
                        None
                    ]
                    # actions_tensor = actor_critic.select_action(
                    #     obs_tensor, latent_code, True
                    # )
                    actions_tensor = actor_critic(obs_tensor, latent_code)
                    action = actions_tensor[0].cpu().numpy()
                    # action = action / np.linalg.norm(action) * env.max_ac_mag * 0.1
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
    # trained_model_dir = "/mnt/SSD4/tmp_exp_gail/vae_bc/final_ckp"
    device = "cuda:0"
    args = get_args()
    args.is_train = False
    args.vanilla = False
    print(args)

    data_name = "circle"
    args.env_name == "Circles-v0"
    args.sa_dim = (10, 2)
    # data_name = "cheetah-dir"
    # args.env_name == "cheetah-dir"

    trained_model_dir = "vae_bc_final_ckp"

    checkpoint_path = os.path.join(
        trained_model_dir, data_name, "checkpoints/bestvae_bc_model.pth"
    )
    # checkpoint_path = "/mnt/SSD3/tianyi/pytorch-a2c-ppo-acktr-gail/checkpoints/bestbc_model_leak.pth"

    load_path = os.path.join(args.save_dir, args.name)
    envs = make_vec_envs(
        args.env_name, args.seed, 1, args.gamma, args.log_dir, args.device, False, args
    )

    ob_rms = get_vec_normalize(envs).ob_rms
    obsfilt = get_vec_normalize(envs)._obfilt

    env_ob_var = np.array(
        [
            103.62293212,
            265.72141744,
            103.6513205,
            265.72127209,
            103.67548928,
            265.72123231,
            103.696535,
            265.72145825,
            103.71571987,
            265.72170916,
        ]
    )
    env_ob_mean = np.array(
        [
            4.92447428e-03,
            6.73194081e00,
            3.70561400e-03,
            6.73222692e00,
            2.88408264e-03,
            6.73245506e00,
            2.48335274e-03,
            6.73263316e00,
            2.49720422e-03,
            6.73279605e00,
        ]
    )
    ob_rms.mean = env_ob_mean
    ob_rms.var = env_ob_var

    bc = VAE_BC(
        device=device,
        code_dim=args.latent_dim,
        input_size_sa=args.sa_dim[0] + args.sa_dim[1],
        input_size_state=args.sa_dim[0],
    )

    print("before loading")
    bc.decoder.load_state_dict(torch.load(checkpoint_path)["state_dict_decoder"])
    # bc.decoder.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    policy_net = bc.decoder.to(device)
    print("done loading")

    ############################## Temporary ##############################
    # import ipdb
    # ipdb.set_trace()
    def act(self, state, latent_code, deterministic=False):
        return None, self.select_action(state, latent_code, not deterministic), None

    MlpPolicyNet.act = act
    #######################################################################
    for epoch in range(20):
        visualize_env(args, policy_net, obsfilt, epoch, num_steps=1000)
