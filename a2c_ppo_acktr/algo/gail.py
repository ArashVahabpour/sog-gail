from a2c_ppo_acktr.utils import load_expert
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import Adam
from torch import autograd
from torch.distributions import Normal

from baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(Discriminator, self).__init__()

        self.args = args
        self.device = args.device

        if args.vae_gail:
            input_dim += args.latent_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)).to(self.device)

        self.trunk.train()

        self.optimizer = Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_data,
                         policy_data,
                         lambda_=10):
        alpha = torch.rand(expert_data.size(0), 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts,  obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_latent_code, policy_action = policy_batch[:3]
            policy_d_input = torch.cat([policy_state, policy_action], dim=1)
            if self.args.vae_gail:
                policy_d_input = torch.cat([policy_d_input, policy_latent_code], dim=1)
            policy_d = self.trunk(policy_d_input)

            expert_state, expert_action = expert_batch[:2]
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d_input = torch.cat([expert_state, expert_action], dim=1)
            if self.args.vae_gail:
                expert_latent_code = expert_batch[2]
                expert_d_input = torch.cat([expert_d_input, expert_latent_code], dim=1)
            expert_d = self.trunk(expert_d_input)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_d_input, policy_d_input)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return loss / n

    def predict_reward(self, state, action, latent_code, gamma, masks):
        with torch.no_grad():
            self.eval()
            if latent_code is None:
                d = self.trunk(torch.cat([state, action], dim=1))
            else:
                d = self.trunk(torch.cat([state, action, latent_code], dim=1))

            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            self.returns = self.returns * masks * gamma + reward
            self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


# Specific to InfoGAIL
class Posterior(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super().__init__()
        self.continuous = args.continuous
        self.model, self.model_target = [self.create_model(input_dim, hidden_dim, args.latent_dim, args.device) for _ in range(2)]
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.batch_size = args.gail_batch_size
        self.latent_dim = args.latent_dim

    def create_model(self, input_dim, hidden_dim, latent_dim, device):
        if self.continuous:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, latent_dim + 1)).to(device)
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, latent_dim), nn.Softmax(dim=1)).to(device)

    @staticmethod
    def categorical_cross_entropy(pred, target):
        eps = 1e-8  # to avoid numerical instability at log(0)
        return -((pred + eps) * target).sum(dim=1).log().mean()

    @staticmethod
    def log_likelihood(mu, log_scale, latent_code):
        scale = torch.exp(log_scale)
        dist = Normal(loc=mu, scale=scale)
        return dist.log_prob(latent_code).sum(dim=1).mean()

    def update(self, rollouts):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=self.batch_size)

        total_loss = 0
        n = 0
        for policy_batch in policy_data_generator:
            state, latent_code, action = policy_batch[:3]
            p = self.model(torch.cat([state, action], dim=1))

            if self.continuous:
                loss = -self.log_likelihood(p[:, :-1], p[:, -1:], latent_code)
            else:
                loss = self.categorical_cross_entropy(p, latent_code)

            total_loss += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # soft update
            tau = 0.5
            for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return total_loss / n

    def predict_reward(self, state, latent_code, action, gamma, masks):
        with torch.no_grad():
            self.eval()

            p = self.model_target(torch.cat([state, action], dim=1))

            if self.continuous:
                reward = self.log_likelihood(p[:, :-1], p[:, -1:], latent_code)
            else:
                reward = -self.categorical_cross_entropy(p, latent_code)

            return reward

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))


def shared_data_loader(self):
    """A data loader that gives batches of (s,a) pairs from shared trajectories"""
    expert = load_expert(self.expert_filename)
    num_traj, traj_len = expert['states'].shape[:2]
    
    bc_batch_size = self.args.bc_batch_size
    assert bc_batch_size <= traj_len, 'batch size cannot be larger than trajectory length'

    for _ in range(num_traj * traj_len // self.args.bc_batch_size):
        traj_idx = np.random.randint(0, num_traj)
        step_idx = np.random.permutation(traj_len)[np.arange(bc_batch_size)]
        yield [expert[key][traj_idx][step_idx] for key in ('states', 'actions')]


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, filename, num_traj=4, subsample_frequency=20, vae_modes=None, sog_expert=False, args=None):
        all_trajectories = load_expert(filename)
        num_all_traj, traj_len = all_trajectories['states'].shape[:2]

        if num_traj is None:
            num_traj = num_all_traj

        if sog_expert and args.shared:
            self.gail_batch_size = args.gail_batch_size
            self.shared = True

            shared_states = []
            shared_actions = []

            for _ in range(num_traj * traj_len // args.gail_batch_size):
                traj_idx = np.random.randint(0, num_traj)
                step_idx = np.random.randint(0, traj_len, args.gail_batch_size)

                shared_states.append(all_trajectories['states'][traj_idx][step_idx])
                shared_actions.append(all_trajectories['actions'][traj_idx][step_idx])

            # (dataset_size * gail_batch_size) x dim
            self.shared_states = torch.cat(shared_states)
            self.shared_actions = torch.cat(shared_actions)

        else:
            self.shared = False
            perm = torch.randperm(num_all_traj)
            idx = perm[:num_traj]

            self.trajectories = {}

            # See https://github.com/pytorch/pytorch/issues/14886
            # .long() for fixing bug in torch v0.4.1
            start_idx = torch.randint(
                0, subsample_frequency, size=(num_traj,)).long()

            for k, v in all_trajectories.items():
                data = v[idx]

                if k in ['states', 'actions']:
                    samples = []
                    for i in range(num_traj):
                        samples.append(data[i, start_idx[i]::subsample_frequency])
                    self.trajectories[k] = torch.stack(samples)
                elif k == 'lengths':
                    self.trajectories[k] = data // subsample_frequency
                else:  # e.g. radii for Circles-v0 environment
                    self.trajectories[k] = data

            if vae_modes is not None:
                self.trajectories['vae_modes'] = vae_modes[idx]

            self.i2traj_idx = {}
            self.i2i = {}

            self.length = self.trajectories['lengths'].sum().item()

            traj_idx = 0
            i = 0

            self.get_idx = []

            for j in range(self.length):

                while self.trajectories['lengths'][traj_idx].item() <= i:
                    i -= self.trajectories['lengths'][traj_idx].item()
                    traj_idx += 1

                self.get_idx.append((traj_idx, i))

                i += 1

    def __len__(self):
        if self.shared:
            return len(self.shared_states)  #TODO check
        else:
            return self.length

    def __getitem__(self, i):
        if self.shared:
            data = [self.shared_states[i], self.shared_actions[i]]
        else:
            traj_idx, i = self.get_idx[i]
            data = [self.trajectories['states'][traj_idx][i], self.trajectories['actions'][traj_idx][i]]
            vae_modes = self.trajectories.get('vae_modes', None)
            if vae_modes is not None:
                data.append(vae_modes[traj_idx])

        return data
