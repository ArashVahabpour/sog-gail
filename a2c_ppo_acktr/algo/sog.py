import torch
import torch.nn as nn


criterion_ = nn.MSELoss(reduction='none')


class SOG:
    def __init__(self, actor_critic, args):
        self.actor_critic = actor_critic
        self.device = args.device
        self.latent_dim = args.latent_dim
        self.criterion = nn.MSELoss()
        self.criterion_no_reduction = nn.MSELoss(reduction='none')

    def predict_loss(self, expert_state, expert_action):
        latent_code = self._resolve_latent_code(expert_state, expert_action)
        _, action, _ = self.actor_critic.act(expert_state, latent_code, deterministic=True)
        return self.criterion(action, expert_action)

    def _resolve_latent_code(self, state, action):
        batch_size = len(state)
        latent_batch_size = self.latent_dim

        # batch_size x latent_batch_size x variable_dim
        all_z = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        all_state = state.unsqueeze(1).expand(-1, latent_batch_size, -1)
        all_action = action.unsqueeze(1).expand(-1, latent_batch_size, -1)

        with torch.no_grad():
            _, action_all, _ = self.actor_critic.act(
                all_state.reshape(batch_size * latent_batch_size, -1),
                all_z.reshape(batch_size * latent_batch_size, -1),
                deterministic=True)
            action_all = action_all.reshape(batch_size, latent_batch_size, -1)

        # batch_size x latent_batch_size x action_dim
        loss = self.criterion_no_reduction(action_all, all_action)

        # batch_size x latent_batch_size
        loss = loss.mean(dim=2)

        # batch_size
        _, argmin = loss.min(dim=1)

        # new_z: batch_size x latent_batch_size x n_latent
        # best_idx: batch_size x 1 x n_latent
        best_idx = argmin[:, None, None].repeat(1, 1, self.latent_dim)

        # batch_size x 1 x n_latent
        best_z = torch.gather(all_z, 1, best_idx)

        # batch_size x n_latent
        best_z = best_z.squeeze(1)

        return best_z
