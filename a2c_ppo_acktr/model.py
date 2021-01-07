import numpy as np
import torch
import torch.nn as nn

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, args):
        super(Policy, self).__init__()
        if args.env_name == 'Circles-v0':
            base = CirclesMLPBase
            # base = MLPBase
        else:
            raise NotImplementedError

        self.base = base(obs_shape[0], args.latent_dim)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.vanilla = args.vanilla

    def act(self, states, latent_codes, deterministic=False):
        value, actor_features = self.base(states, latent_codes)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()

        return value, latent_codes, action, action_log_probs

    def get_value(self, states, latent_codes):
        value, _ = self.base(states, latent_codes)
        return value

    def evaluate_actions(self, states, latent_codes, action):
        value, actor_features = self.base(states, latent_codes)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


# class NNBase(nn.Module):
#     def __init__(self, recurrent, recurrent_input_size, hidden_size):
#         super(NNBase, self).__init__()
#
#         self._hidden_size = hidden_size
#         self._recurrent = recurrent
#
#
#
#     @property
#     def is_recurrent(self):
#         return self._recurrent
#
#     @property
#     def recurrent_hidden_state_size(self):
#         if self._recurrent:
#             return self._hidden_size
#         return 1
#
#     @property
#     def output_size(self):
#         return self._hidden_size
#
#     def _forward_gru(self, x, hxs, masks):
#         if x.size(0) == hxs.size(0):
#             x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
#             x = x.squeeze(0)
#             hxs = hxs.squeeze(0)
#         else:
#             # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
#             N = hxs.size(0)
#             T = int(x.size(0) / N)
#
#             # unflatten
#             x = x.view(T, N, x.size(1))
#
#             # Same deal with masks
#             masks = masks.view(T, N)
#
#             # Let's figure out which steps in the sequence have a zero for any agent
#             # We will always assume t=0 has a zero in it as that makes the logic cleaner
#             has_zeros = ((masks[1:] == 0.0) \
#                             .any(dim=-1)
#                             .nonzero()
#                             .squeeze()
#                             .cpu())
#
#             # +1 to correct the masks[1:]
#             if has_zeros.dim() == 0:
#                 # Deal with scalar
#                 has_zeros = [has_zeros.item() + 1]
#             else:
#                 has_zeros = (has_zeros + 1).numpy().tolist()
#
#             # add t=0 and t=T to the list
#             has_zeros = [0] + has_zeros + [T]
#
#             hxs = hxs.unsqueeze(0)
#             outputs = []
#             for i in range(len(has_zeros) - 1):
#                 # We can now process steps that don't have any zeros in masks together!
#                 # This is much faster
#                 start_idx = has_zeros[i]
#                 end_idx = has_zeros[i + 1]
#
#                 rnn_scores, hxs = self.gru(
#                     x[start_idx:end_idx],
#                     hxs * masks[start_idx].view(1, -1, 1))
#
#                 outputs.append(rnn_scores)
#
#             # assert len(outputs) == T
#             # x is a (T, N, -1) tensor
#             x = torch.cat(outputs, dim=0)
#             # flatten
#             x = x.view(T * N, -1)
#             hxs = hxs.squeeze(0)
#
#         return x, hxs
#
#
# class CNNBase(NNBase):
#     def __init__(self, num_inputs, recurrent=False, hidden_size=512):
#         super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), nn.init.calculate_gain('relu'))
#
#         self.main = nn.Sequential(
#             init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
#             init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
#             init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
#             init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0))
#
#         self.critic_linear = init_(nn.Linear(hidden_size, 1))
#
#         self.train()
#
#     def forward(self, inputs, rnn_hxs, masks):
#         x = self.main(inputs / 255.0)
#
#         if self.is_recurrent:
#             x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
#
#         return self.critic_linear(x), x, rnn_hxs


# class MLPBase(nn.Module):
#     def __init__(self, num_inputs, latent_dim, hidden_size=64):
#         super(MLPBase, self).__init__()
#
#         self.output_size = hidden_size
#         self.is_recurrent = False
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), np.sqrt(2))
#
#         self.actor = nn.Sequential(
#             init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
#             init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
#
#         self.critic = nn.Sequential(
#             init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
#             init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
#
#         self.critic_linear = init_(nn.Linear(hidden_size, 1))
#
#         self.train()
#
#     def forward(self, inputs, rnn_hxs, masks=None):
#         x = inputs
#
#         if self.is_recurrent:
#             x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
#
#         hidden_critic = self.critic(x)
#         hidden_actor = self.actor(x)
#
#         return self.critic_linear(hidden_critic), hidden_actor
#
#
# class CirclesMLPBase(nn.Module):
#     """
#     Multi-Layer Perceptron (Fully Connected) Model Used in InfoGAIL paper
#     https://arxiv.org/pdf/1703.08840.pdf ---> Appendix A
#     """
#
#     def __init__(self, state_dim, latent_dim, hidden_size=64):
#         super().__init__()
#
#         self.output_size = hidden_size
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), np.sqrt(2))
#
#         self.actor_state_encoder = nn.Sequential(
#             init_(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
#             init_(nn.Linear(hidden_size, hidden_size)))
#
#         self.actor_latent_encoder = init_(nn.Linear(latent_dim, hidden_size))
#
#         self.critic = nn.Sequential(
#             init_(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
#             init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
#
#         self.actor_nonlinearity = nn.Tanh()
#         self.critic_linear = init_(nn.Linear(hidden_size, 1))
#
#         self.train()
#
#     def forward(self, states, latent_codes):
#         hidden_critic = self.critic(states)
#         hidden_actor = self.actor_nonlinearity(self.actor_state_encoder(states) +
#                                                self.actor_latent_encoder(latent_codes))
#
#         return self.critic_linear(hidden_critic), hidden_actor


class CirclesMLPBase(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_size=128):
        super().__init__()

        self.output_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor_states = nn.Sequential(
            init_(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)))

        self.actor_latent = nn.Sequential(
            init_(nn.Linear(latent_dim, hidden_size))
        )

        self.actor_nonlinearity = nn.Tanh()

        self.critic = nn.Sequential(
            init_(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, states, latent_codes):
        hidden_critic = self.critic(states)
        hidden_actor = self.actor_nonlinearity(self.actor_states(states) + self.actor_latent(latent_codes))

        return self.critic_linear(hidden_critic), hidden_actor


class Posterior(nn.Module):
    pass
