import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import RMSprop, Adam
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(Discriminator, self).__init__()

        self.args = args
        self.device = args.device
        self.wasserstein = args.wasserstein

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)).to(self.device)

        self.trunk.train()

        optimizer = Adam #= RMSprop if args.wasserstein else Adam
        self.optimizer = optimizer(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

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

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_latent_codes, policy_action = policy_batch[:3]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            if self.wasserstein:
                expert_loss = expert_d.mean()
                policy_loss = -policy_d.mean()
            else:
                expert_loss = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.ones(expert_d.size()).to(self.device))
                policy_loss = F.binary_cross_entropy_with_logits(
                    policy_d,
                    torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

            if self.wasserstein:
                self.clip_weights()
        return loss / n

    def clip_weights(self):
        for p in self.trunk.parameters():
            p.data.clamp_(-0.01, 0.01)

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = -d if self.wasserstein else s.log() - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


# Specific to InfoGAIL
class Posterior(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super().__init__()
        self.model, self.model_target = [self.create_model(input_dim, hidden_dim, args.latent_dim, args.device) for _ in range(2)]
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.batch_size = args.gail_batch_size

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    @staticmethod
    def create_model(input_dim, hidden_dim, latent_dim, device):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.Softmax(dim=1)).to(device)

    @staticmethod
    def categorical_cross_entropy(pred, target):
        eps = 1e-8  # to avoid numerical instability at log(0)
        return -((pred + eps) * target).sum(dim=1).log().mean()

    def update(self, rollouts):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=self.batch_size)

        total_loss = 0
        n = 0
        for policy_batch in policy_data_generator:
            state, latent_code, action = policy_batch[:3]
            p = self.model(torch.cat([state, action], dim=1))

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
            reward = -self.categorical_cross_entropy(p, latent_code)

            if self.returns is None:
                self.returns = reward.clone()

            self.returns = self.returns * masks * gamma + reward
            self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)
        num_all_trajectories = all_trajectories['states'].size(0)

        if num_trajectories is None:
            num_trajectories = num_all_trajectories

        perm = torch.randperm(num_all_trajectories)
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k in ['states', 'actions']:
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            elif k == 'lengths':
                self.trajectories[k] = data // subsample_frequency
            else:  # e.g. radii for Circles-v0 environment
                self.trajectories[k] = data

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
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]
