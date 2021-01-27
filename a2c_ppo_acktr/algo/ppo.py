import torch
import torch.nn as nn
import torch.optim as optim
from a2c_ppo_acktr.algo.sog import BlockCoordinateSearch, OneHotSearch


class PPO:
    def __init__(self,
                 actor_critic,
                 args,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        # sog gail
        self.sog_gail = args.sog_gail
        self.sog_gail_coef = args.sog_gail_coef if self.sog_gail else None

        if args.latent_optimizer == 'bcs':
            SOG = BlockCoordinateSearch
        elif args.latent_optimizer == 'ohs':
            SOG = OneHotSearch
        else:
            raise NotImplementedError

        self.sog = SOG(actor_critic, args) if args.sog_gail else None

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.device = args.device

    def update(self, rollouts, sog_train_loader, obsfilt):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        sog_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, latent_codes_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, latent_codes_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                if self.sog_gail:
                    expert_state, expert_action = next(sog_train_loader)
                    expert_state = obsfilt(expert_state.numpy(), update=False)
                    expert_state, expert_action = expert_state.to(self.device), expert_action.to(self.device)
                    sog_loss = self.sog.predict_loss(expert_state, expert_action)
                    loss += sog_loss * self.sog_gail_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                if self.sog_gail:
                    sog_loss_epoch += sog_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        if self.sog_gail:
            sog_loss_epoch /= num_updates
        else:
            sog_loss_epoch = None

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, sog_loss_epoch
