'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import optim

# from baselines.common.running_mean_std import RunningMeanStd

import argparse
import tempfile
import os.path as osp
import gym
from tqdm.auto import tqdm

from utilities import normal_log_density, set_random_seed, to_tensor, save_checkpoint, load_pickle, onehot, get_logger
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
# import tensorflow as tf


# from baselines.gail import mlp_policy
from baselines import bench
# TODO: migrate ffjord's logger to here (and suppress the logging of full source code (or at least suppress the output of that part))
# from baselines import logger
import logging
# from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
# from baselines.common.mpi_adam import MpiAdam
# from baselines.gail.run_mujoco import runner
# from baselines.gail.dataset.mujoco_dset import Mujoco_Dset

from torch.utils.data import Dataset, TensorDataset, DataLoader


class ExpertTrajectory:
    def __init__(self, path):
        # path = "/mnt/SSD3/Qiujing_exp/Imitation_learning/data/three_modes_traj_train_everywhere_static.pkl"
        #exp_data = load_pickle("three_modes_traj.pkl")
        exp_data = load_pickle(path)
        #self.exp_states = np.concatenate([val['state'].reshape(-1,2,5) for val in exp_data.values()])
        self.expert_states = np.concatenate([val['state'].transpose(
            (0, 1, 3, 2)).reshape(-1, 10) for val in exp_data.values()])
        self.expert_actions = np.concatenate(
            [val['action'].reshape(-1, 2) for val in exp_data.values()])
        self.n_transitions = self.expert_actions.shape[0]
        self.mode_names = list(exp_data.keys())
        self.num_step = exp_data[self.mode_names[0]]['state'].shape[1]
        print("mode names", self.mode_names)
        print("number of transitions:", self.n_transitions)
        print("number of steps in each traj:", self.num_step)
        # np.random.seed(2)

    def sample(self, batch_size):
        """
        Option1: generate latent code for each sample action
        Option2: fix latent code for each traj. Sample states together with latent code
        FIXME: why can the latent code be inconsistent across batches?
        """
        indexes = np.sort(np.random.randint(
            0, self.n_transitions, size=batch_size))
        traj_index = indexes//self.num_step
        unique_traj, traj_fre = np.unique(traj_index, return_counts=True)
        num_unique_traj = len(unique_traj)
        fake_z0 = np.random.randint(3, size=num_unique_traj)
        #new_z = np.zeros(batch_size, dtype=int)
        #print("begin t before latent code sampling:", time() -  start_t)

        new_z = np.repeat(fake_z0, traj_fre)
        #print("end t after latent code sampling:", time() -  start_t)
        fake_z = onehot(new_z, 3)

        state = self.expert_states[indexes]
        action = self.expert_actions[indexes]
        #print("sampled data shape", np.array(state).shape, np.array(action).shape)
        #print("end t after sampling:", time() -  start_t)

        return np.array(state), np.array(action), fake_z, indexes


class TrajectoryDataset(TensorDataset):
    def __init__(self, X, y, c, transform=None, device="cpu"):
        # assert X.shape[0] == y.shape[0] == c.shape[0],\
        #     f"Data length is not aligned: X.shape[0] == {X.shape[0]}, "\
        #     f"y.shape[0] == {y.shape[0]}, c.shape[0] == {c.shape[0]}"
        self.transform = transform if transform is not None else lambda x: x
        X = self.transform(X)
        self.X = torch.as_tensor(X, device=device)  # state
        self.y = torch.as_tensor(y, device=device)  # action
        self.c = torch.as_tensor(c, device=device)
        super(TrajectoryDataset, self).__init__(self.X, self.y, self.c)


def _train(epoch, net, dataloader, optimizer, criterion, device, writer):
    net.train()
    train_loss = 0
    # dataloader
    num_batch = len(dataloader)
    for batch_idx, (state, action, latent_code) in enumerate(dataloader):

        optimizer.zero_grad()
        state, action, latent_code = to_tensor(state, device),\
            to_tensor(action, device),\
            to_tensor(latent_code, device)

        outputs = net(state, latent_code)
        #print(outputs, state, latent_code)
        loss = criterion(outputs, action)
        loss.backward()
        optimizer.step()
        #print("loss data", loss.data)
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Loss: %.3f ' % (train_loss/((batch_idx+1)*3)))
            if writer is not None:
                writer.add_scalars(
                    "Loss/BC", {"train_loss": train_loss/((batch_idx+1)*3)}, batch_idx + num_batch * (epoch-1))


class BC():
    def __init__(self, epochs=300, lr=0.0001, eps=1e-5, device="cpu", policy_activation=F.relu,
                 tb_writer=None, validate_freq=1, checkpoint_dir=".", code_dim=None):
        self.epochs = epochs
        self.device = device
        self.policy = MlpPolicyNet(
            state_dim=10, code_dim=code_dim, ft_dim=128, activation=policy_activation
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)
        self.criterion = nn.MSELoss()
        self.writer = tb_writer
        self.validate_freq = validate_freq
        self.checkpoint_dir = checkpoint_dir

    def train(self, expert_loader, val_loader):
        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            print('\nEpoch: %d' % epoch)
            _train(epoch, self.policy, expert_loader, self.optimizer,
                   self.criterion, self.device, self.writer)
            if epoch % self.validate_freq == 0:
                best_loss, checkpoint_path = _validate(epoch, self.policy, val_loader,
                                                       self.criterion, self.device, best_loss,
                                                       self.writer, self.checkpoint_dir)
        best_loss, checkpoint_path = _validate(epoch, self.policy, val_loader,
                                               self.criterion, self.device, best_loss,
                                               self.writer, self.checkpoint_dir)
        self.load_best_checkpoint(checkpoint_path)

    def load_best_checkpoint(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path)['state_dict'])


def _validate(epoch, net, val_loader, criterion, device, best_loss, writer, checkpoint_dir):
    net.eval()
    valid_loss = 0
    number_batches = len(val_loader)
    avg_valid_loss = None
    for batch_idx, (state, action, latent_code) in enumerate(val_loader):
        state, action, latent_code = to_tensor(state, device),\
            to_tensor(action, device),\
            to_tensor(latent_code, device)
        outputs = net(state, latent_code)
        loss = criterion(outputs, action)

        valid_loss += loss.item()

        avg_valid_loss = valid_loss/(batch_idx+1)
        if batch_idx % 100 == 0:
            print('Valid Loss: %.3f ' % (valid_loss/(batch_idx+1)))

            if writer is not None:
                writer.add_scalars("Loss/BC_val", {"val_loss": valid_loss/(
                    (batch_idx+1))}, batch_idx + number_batches * (epoch-1))
    checkpoint_path = osp.join(
        checkpoint_dir, 'checkpoints/bestbc_model_new_everywhere.pth')
    assert avg_valid_loss is not None, "Empty avg_valid_loss. Possibly empty dataloader?"
    if avg_valid_loss <= best_loss:
        best_loss = avg_valid_loss
        print('Best epoch: ' + str(epoch))
        save_checkpoint({'epoch': epoch,
                         'avg_loss': avg_valid_loss,
                         'state_dict': net.state_dict(),
                         }, save_path=checkpoint_path)
    return best_loss, checkpoint_path


def load_expert_data(
    data_file, one_hot: bool = True, one_hot_dim: int = None, code_map=None
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, Dict]:
    # TODO: Currently the one-hot encoding is done in memory all at once. Potentially it needs to be moved to a custom DataLoader like ExpertTrajectory above
    """For Arash's format
    """
    data_dict = torch.load(data_file)

    lengths = data_dict["lengths"]
    # lengths = lengths[data_dict["radii"] == 10]

    states_all = data_dict["states"]
    # num_traj, max_traj_len_x, dim_state = states_all.shape
    # X_all = X_all[data_dict["radii"] == 10]
    # X_all = X_all.reshape(-1, dim_state)
    states_all = torch.cat([X[:l] for X, l in zip(states_all, lengths)], dim=0)

    action_all = data_dict["actions"]
    # num_traj, max_traj_len_y, dim_action = action_all.shape
    # y_all = y_all[data_dict["radii"] == 10]
    # y_all = y_all.reshape(-1, dim_action)
    action_all = torch.cat([y[:l] for y, l in zip(action_all, lengths)], dim=0)

    c = data_dict["radii"]
    # c = c[data_dict["radii"] == 10]

    # change to scalar encoding here in case it's useful
    unique_c, inv = np.unique(c, return_inverse=True)
    dim = len(unique_c)

    # make sure that one_hot_dim >= the inferred dim
    if one_hot_dim is None:
        one_hot_dim = dim
    elif one_hot_dim < dim:
        raise ValueError(f"one_hot_dim ({one_hot_dim}) is smaller than the"
                         f" number of unique values in c ({dim})")

    if code_map is None:
        codes = np.argsort(np.argsort(unique_c))
        code_map = dict(zip(unique_c, codes))
        # c, fake_c = codes[inv], np.random.choice(codes, size=len(c))
        # c_all, fake_c_all = np.repeat(c, lengths), np.repeat(fake_c, lengths)
        c = codes[inv]
    else:
        c = np.array([code_map[c_] for c_ in unique_c])[inv]

    try:
        # for torch.tensor lengths
        code_all, fake_code_all = (
            np.repeat(c, lengths),
            np.random.randint(dim, size=lengths.sum().item()),
        )
    except:
        # for np.array lengths
        code_all, fake_code_all = (
            np.repeat(c, lengths),
            np.random.randint(dim, size=lengths.sum()),
        )

    if one_hot:
        code_all, fake_code_all = (
            onehot(code_all, dim=one_hot_dim),
            onehot(fake_code_all, dim=one_hot_dim),
        )

    # c_all = torch.zeros(num_traj, traj_len, dtype=torch.int64)
    # c_all[c == 10, :] = 1
    # c_all[c == 20, :] = 2
    # c_all[c == -10, :] = 3
    # c_all = c_all.flatten()

    return states_all, action_all, code_all, fake_code_all, code_map


def create_dataset(train_data_path, val_data_path=None, fake=True, one_hot=True, one_hot_dim=None):
    from sklearn.model_selection import train_test_split
    X, y, c, fake_c, code_map = load_expert_data(
        train_data_path, one_hot=one_hot, one_hot_dim=one_hot_dim
    )
    if fake:
        c = fake_c
    if val_data_path is None:
        X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
            X, y, c, test_size=0.2)
    else:
        X_train, y_train, c_train = X, y, c
        X_val, y_val, c_val, fake_c_val, code_map = load_expert_data(
            val_data_path, one_hot=one_hot, code_map=code_map
        )
        if fake:
            c_val = fake_c_val

    train_dataset = TrajectoryDataset(X_train, y_train, c_train)
    val_dataset = TrajectoryDataset(X_val, y_val, c_val)
    return train_dataset, val_dataset


def create_dataloader(train_dataset, val_dataset, batch_size=4):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
    return train_dataloader, val_dataloader


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


class PolicyNet(ABC, nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

    @abstractmethod
    def get_log_prob(self, state, actions):
        pass

    @abstractmethod
    def select_action(self, state, stochastic):
        pass


class MlpPolicyNet(PolicyNet):
    def __init__(self, state_dim=10, code_dim=3, ft_dim=128, output_dim=2, activation=F.relu):
        super(MlpPolicyNet, self).__init__()
        self.activation = activation
        self.fc_s1 = nn.Linear(state_dim, ft_dim)
        self.fc_s2 = nn.Linear(ft_dim, ft_dim)
        self.code_dim = code_dim
        if code_dim is not None:
            self.fc_c1 = nn.Linear(code_dim, ft_dim)
        self.fc_sum = nn.Linear(ft_dim, output_dim)
        self.action_logstds = torch.log(
            torch.from_numpy(np.array([2, 2])).clone().float())
        self.action_std = torch.from_numpy(np.array([2, 2])).clone().float()

    def forward(self, state, latent_code):
        output = self.fc_s2(self.activation(self.fc_s1(state), inplace=True))
        if self.code_dim is not None:
            output += self.fc_c1(latent_code)
        final_out = self.fc_sum(self.activation(output, inplace=True))
        return final_out

    def get_log_prob(self, state, latent_code, actions):
        """
        For continuous action space. fixed action log std
        """
        action_mu = self.forward(state, latent_code)
        device = state.device
        return normal_log_density(actions, action_mu, self.action_logstds.to(device), self.action_std.to(device))

    def select_action(self, state, latent_code, stochastic=True):
        action_mu = self.forward(state, latent_code)
        #normal_log_density_fixedstd(x, action_mu)
        device = state.device
        if stochastic:
            action = torch.normal(action_mu, self.action_std.to(device))
        else:
            action = action_mu.to(device)
        return action

    def act(self, state, latent_code, deterministic=False):
        return None, self.select_action(state, latent_code, not deterministic), None


# def argsparser():
#     parser = argparse.ArgumentParser(
#         "PyTorch Adaption of `baselines` Behavior Cloning")
#     parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
#     parser.add_argument('--seed', help='RNG seed', type=int, default=0)
#     parser.add_argument('--expert_path', type=str,
#                         default='data/deterministic.trpo.Hopper.0.00.npz')
#     parser.add_argument(
#         '--checkpoint_dir', help='the directory to save model', default='checkpoint')
#     parser.add_argument(
#         '--log_dir', help='the directory to save log file', default='log')
#     #  Mujoco Dataset Configuration
#     parser.add_argument('--traj_limitation', type=int, default=-1)
#     # Network Configuration (Using MLP Policy)
#     parser.add_argument('--policy_hidden_size', type=int, default=100)
#     # for evaluation
#     boolean_flag(parser, 'stochastic_policy', default=False,
#                  help='use stochastic/deterministic policy to evaluate')
#     boolean_flag(parser, 'save_sample', default=False,
#                  help='save the trajectories or not')
#     parser.add_argument(
#         '--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
#     return parser.parse_args()
def parse_args(*args):
    parser = argparse.ArgumentParser("Behavior Cloning for Circle env")
    parser.add_argument('--seed', help='RNG seed', type=int, default=3)
    parser.add_argument(
        '--train_data', help='Training dataset', type=str, default='trajs_circles_mix',
        choices=['trajs_circles', 'trajs_circles_flip',
                 'trajs_cicles_mix', 'trajs_circles_four']
    )
    parser.add_argument(
        '--code_dim', type=int, default=None,
        help='Latent code dimension, None for disabling'
    )
    parser.add_argument(
        '--consistent', type=bool, default=True,
        help='During inference use consistent code for each trajectory.'
             ' Otherwise the code is random for each state-action.'
    )
    parser.add_argument(
        '--noise_level', help='The noise level in env.', type=float, default=0.1
    )
    parser.add_argument('--name', help='The name of the inference run.',
                        type=str, default="inference_codeless_0.1")
    parser.add_argument(
        '--device', default="cuda:1", help='The device to use.'
    )
    boolean_flag(
        parser, 'fake', default=True,
        help='Train with fake codes. Otherwise true codes will be provided.'
    )
    boolean_flag(parser, 'train', default=True, help='Train the model')
    boolean_flag(parser, 'inference', default=True, help='Inference the model')
    boolean_flag(parser, 'render', default=False,
                 help='Render during the inference')
    return parser.parse_args(*args)


if __name__ == '__main__':
    from inference import get_start_state, model_infer_vis, model_inference_env, visualize_trajs_new
    args = parse_args()
    code_dim = args.code_dim
    train_with_fake_code = args.fake
    inference_noise = args.noise_level
    inference_name = args.name
    device = args.device
    train = args.train
    inference = args.inference
    consistent_inference_code = args.consistent
    render = args.render
    train_data_path = f"/home/shared/datasets/gail_experts/{args.train_data}.pt"
    set_random_seed(args.seed, using_cuda=True)

    ############### Train ###############
    if train:
        # train_data_path = "three_modes_traj_train_everywhere.pkl"
        # val_data_path = "three_modes_traj_val.pkl"
        bc = BC(epochs=30, lr=1e-4, eps=1e-5, device=device, code_dim=code_dim)
        # bc = BC(epochs=30, lr=1e-4, eps=1e-5, device="cuda:0", code_dim=None)
        # train_data_path = "/home/shared/datasets/gail_experts/trajs_circles.pt"
        train_dataset, val_dataset = create_dataset(
            train_data_path, fake=train_with_fake_code, one_hot=True, one_hot_dim=code_dim)
        train_loader, val_loader = create_dataloader(
            train_dataset, val_dataset, batch_size=400)
        bc.train(train_loader, val_loader)
        model = bc.policy
    ############### Load Checkpoint ###############
    else:
        # train_data_path = "/home/shared/datasets/gail_experts/trajs_circles.pt"
        train_dataset, val_dataset = create_dataset(
            train_data_path, fake=False, one_hot=True, one_hot_dim=code_dim)
        model = MlpPolicyNet(code_dim=code_dim)
        # model = MlpPolicyNet(code_dim=None)
        checkpoint = torch.load(
            "checkpoints/bestbc_model_new_everywhere.pth")["state_dict"]
        model.load_state_dict(checkpoint)

    ############### Inference ###############
    if not inference:
        if not train:
            raise ValueError("Why bother if you don't train nor inference")
    else:
        num_trajs = 20
        start_state = get_start_state(
            num_trajs, mode="sample_data", dataset=val_dataset)
        # print(start_state.shape)
        if code_dim is not None and consistent_inference_code:
            fake_code = onehot(np.random.randint(
                code_dim, size=num_trajs), dim=code_dim)
        else:
            fake_code = None
        # fake_code = torch.zeros(num_trajs, code_dim)
        # fake_code[:,0] = 1
        traj_len = 1000
        # model_infer_vis(model, start_state, fake_code, traj_len, save_fig_name=f"{inference_name}")

        ############### use env for inference ###############
        flat_state_arr, action_arr = model_inference_env(
            model, num_trajs, traj_len, state_len=5, radii=[-10, 10, 20],
            codes=fake_code, noise_level=inference_noise, render=render
        )
        visualize_trajs_new(flat_state_arr, action_arr,
                            f"./imgs/circle/env_{inference_name}.png")
