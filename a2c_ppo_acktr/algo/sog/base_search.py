import torch
import torch.nn as nn


class BaseSearch:
    def __init__(self, actor_critic, args):
        self.actor_critic = actor_critic
        self.device = args.device
        self.cuda = args.cuda
        self.latent_dim = args.latent_dim
        self.latent_batch_size = args.latent_batch_size
        self.shared_code = args.shared_code
        self.criterion = nn.MSELoss()
        self.criterion_no_reduction = nn.MSELoss(reduction='none')

    def search_iter(self, all_z, states, actions):
        batch_size = actions.shape[0]

        # (batch_size * latent_batch_size) x n_latent
        all_z_r = all_z.reshape(batch_size * self.latent_batch_size, self.latent_dim)

        # batch_size x dim_1 x ... x dim_k ---> batch_size x latent_batch_size x dim_1 x ... x dim_k
        all_shape = lambda tensor: [tensor.shape[0], self.latent_batch_size, *tensor.shape[1:]]

        # batch_size x latent_batch_size x dim_1 x ... x dim_kx
        states_all = states.unsqueeze(1).expand(all_shape(states))
        # (batch_size * latent_batch_size) x dim_1 x ... x dim_kx
        states_all_r = states_all.reshape(-1, *states_all.shape[2:])

        if self.cuda:
            torch.cuda.synchronize()
        with torch.no_grad():  # no need to store the gradients while searching
            # (batch_size * latent_batch_size) x dim_1 x ... x dim_ky
            _, actions_pred_all_r, _ = self.actor_critic.act(
                states_all_r,
                all_z_r,
                deterministic=True)

        # batch_size x latent_batch_size x dim_1 x ... x dim_ky
        fake_all = actions_pred_all_r.reshape(all_shape(actions))
        real_all = actions.unsqueeze(1).expand(all_shape(actions))

        # batch_size x latent_batch_size x dim_1 x ... x dim_ky
        loss = self.criterion_no_reduction(real_all, fake_all)

        # batch_size x latent_batch_size x -1
        loss = loss.reshape([batch_size, self.latent_batch_size, -1])

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
        best_z = best_z.squeeze(1)  # TODO unit test with batch size of 1

        return best_z

    def resolve_latent_code(self, state, action):
        pass

    def predict_loss(self, expert_state, expert_action):
        latent_code = self.resolve_latent_code(expert_state, expert_action)
        _, action, _ = self.actor_critic.act(expert_state, latent_code, deterministic=True)
        return self.criterion(action, expert_action)
