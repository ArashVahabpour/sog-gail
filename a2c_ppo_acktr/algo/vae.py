import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from ..utils import load_expert


class VAE(nn.Module):
    def __init__(self, args, expert_filename):
        super(VAE, self).__init__()

        self.args = args
        self.expert = load_expert(expert_filename)
        state_dim, action_dim = [self.expert[key].shape[2] for key in ['states', 'actions']]

        hidden_dim_lstm, hidden_dim_decoder, latent_dim = args.hidden_dim_lstm, args.hidden_dim_decoder, args.latent_dim

        # encoder
        assert hidden_dim_lstm % 2 == 0, 'number of hidden units should be even for bi-lstm'
        self.lstm = nn.LSTM(state_dim, hidden_dim_lstm // 2, bidirectional=True)
        self.fc_e1 = nn.Linear(hidden_dim_lstm, latent_dim)  # maps to mu
        self.fc_e2 = nn.Linear(hidden_dim_lstm, latent_dim)  # maps to log_var

        # decoder
        self.fc_d1 = nn.Linear(latent_dim + state_dim, hidden_dim_decoder)  # mlp layers on top of z, s
        self.fc_d2 = nn.Linear(hidden_dim_decoder, hidden_dim_decoder)

        self.fc_s = nn.Linear(hidden_dim_decoder, state_dim)  # predicting s
        self.fc_a = nn.Linear(hidden_dim_decoder, action_dim)  # predicting a


    def _encoder(self, traj):
        """
        Args:
            traj: traj_len x state_dim
        Returns:
            mu, log_var: 1 x latent_dim
        """

        h = F.relu(self.lstm(traj.unsqueeze(1))[0].mean(0))
        return self.fc_e1(h), self.fc_e2(h)

    def _sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def _decoder(self, z, s):
        h = F.relu(self.fc_d1(torch.cat([z.expand(s.shape[0], -1), s], dim=1)))
        h = F.relu(self.fc_d2(h))
        s_hat = self.fc_s(h)
        a_hat = self.fc_a(h)

        return s_hat, a_hat

    @staticmethod
    def _loss(s_hat, s, a_hat, a, mu, log_var):
        mse1 = F.mse_loss(s_hat, s, reduction='mean')
        mse2 = F.mse_loss(a_hat, a, reduction='mean')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse1 + mse2 + kld

    def forward(self, traj, idx):
        """
        Args:
            traj: sequence of states with shape `traj_len x state_dim`
            idx: list of length `lstm_batch_size`
        Returns:
            (s_hat, a_hat)
            mu: 1 x latent_dim
            log_var: 1 x latent_dim
        """
        mu, log_var = self._encoder(traj)
        z = self._sampling(mu, log_var)
        return self._decoder(z, traj[idx]), mu, log_var

    def recover_modes(self):
        """

        Returns:
            mus: vae mu output for each trajectory
            log_vars: vae logvar output for each trajectory
            vae_codes: unique codes in the case of using ground truth labels
            vae_codes_all: trajectory codes in the case of using ground truth labels (shared among the trajectories with the same labels)
        """
        print('started training vae-lstm...')

        args = self.args
        device = args.device

        perm = np.random.permutation(len(self.expert['states']))

        optimizer = Adam(self.parameters())
        for epoch in range(1, 1 + args.vae_epochs):
            modes_key = 'radii' if args.env_name == 'Circles-v0' else 'modes'
            for batch_idx, (s, a) in enumerate(
                    zip(self.expert['states'][perm], self.expert['actions'][perm])):

                optimizer.zero_grad()
                idx = np.random.choice(len(s), size=(args.lstm_batch_size,), replace=False)
                s = s.to(device)

                recon_batch, mu, log_var = self.forward(s, idx)
                s_hat, a_hat = recon_batch
                loss = self._loss(s_hat, s[idx], a_hat, a[idx].to(device), mu, log_var)

                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print('VAE Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, len(perm),
                        100. * batch_idx / len(perm), loss.item() / len(s)))

        # recovering latent codes of trajectories with the trained vae
        print('recovering latent codes for expert trajectories...')
        mus, log_vars = [], []
        for s in tqdm(self.expert['states']):
            s = s.to(device)
            _, mu, log_var = self.forward(s, [])
            mus.append(mu.detach())
            log_vars.append(log_var.detach())
        # num_traj x latent_dim
        mus, log_vars = torch.cat(mus), torch.cat(log_vars)
        log_vars -= 2 * mus.std(dim=0).log()
        mus = (mus - mus.mean(dim=0)) / mus.std(dim=0)

        if args.vae_cheat:
            modes = self.expert['modes']
            vae_codes = torch.stack([mus[torch.nonzero(modes == i, as_tuple=True)].mean(dim=0) for i in range(args.vae_num_modes)])
            vae_codes_all = vae_codes[modes.long()]
        else:
            vae_codes, vae_codes_all = mus, mus

        return mus, log_vars, vae_codes, vae_codes_all
