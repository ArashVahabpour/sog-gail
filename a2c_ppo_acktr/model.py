import numpy as np
import torch
import torch.nn as nn

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, args):
        super(Policy, self).__init__()
        if args.env_name in {'Circles-v0', 'Ellipses-v0'}:
            base = CirclesMLPBase
        else:
            raise NotImplementedError  # base = MLPBase

        self.base = base(obs_shape[0], args.latent_dim)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs, args)
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

        return value, action, action_log_probs

    def get_value(self, states, latent_codes):
        value, _ = self.base(states, latent_codes)
        return value

    def evaluate_actions(self, states, latent_codes, action):
        value, actor_features = self.base(states, latent_codes)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class MLPBase(nn.Module):
    def __init__(self, num_inputs, latent_dim, hidden_size=64):
        super(MLPBase, self).__init__()

        self.output_size = hidden_size
        self.is_recurrent = False

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, latent_codes):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor


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
