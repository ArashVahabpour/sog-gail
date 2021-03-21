import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from ..utils import generate_latent_codes, get_vec_normalize
from tqdm import tqdm


class BC:
    """
    Behavioral cloning pretraining of the model, in a conventional way or with SOG
    """
    def __init__(self, agent, save_filename, expert_filename, args, obsfilt, vae_modes):
        self.args = args
        self.save_filename = save_filename
        self.data_loader = self.create_data_loader(expert_filename, vae_modes)
        self.actor_critic = agent.actor_critic
        self.sog = agent.sog
        self.obsfilt = obsfilt

    @staticmethod
    def flatten(x):
        return x.reshape(-1, x.shape[-1])

    def create_data_loader(self, expert_filename, vae_modes):
        expert = torch.load(expert_filename, map_location='cpu')
        states, actions = self.flatten(expert['states']), self.flatten(expert['actions'])

        if self.args.vae_gail:
            latent_codes = self.flatten(vae_modes.unsqueeze(1).expand(-1, expert['states'].shape[1], -1))
            dataset = TensorDataset(states, actions, latent_codes)
        else:
            dataset = TensorDataset(states, actions)
        data_loader = DataLoader(dataset, batch_size=self.args.bc_batch_size, shuffle=True)

        return data_loader

    def pretrain(self, envs):
        actor_critic = self.actor_critic
        args = self.args
        device = args.device
        epochs = args.bc_epochs

        if os.path.exists(self.save_filename):
            ob_rms = get_vec_normalize(envs).ob_rms
            saved_actor_critic, _, _, saved_ob_rms = torch.load(self.save_filename, map_location=device)
            actor_critic.load_state_dict(saved_actor_critic.state_dict())
            ob_rms.mean, ob_rms.var, ob_rms.count = saved_ob_rms.mean, saved_ob_rms.var, saved_ob_rms.count
            print('pretrained model loaded...')
            return

        print('behavioral cloning pretraining started...')

        losses = []
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.actor_critic.parameters())

        for epoch in tqdm(range(1, epochs + 1)):
            for data in self.data_loader:
                data_x, data_y = data[:2]
                data_x = self.obsfilt(data_x.numpy(), update=True)
                data_x = torch.tensor(data_x, dtype=torch.float32, device=device)
                data_y = data_y.to(device)

                if args.vae_gail:
                    latent_codes = data[2]
                elif args.sog_gail:
                    latent_codes = self.sog.resolve_latent_code(data_x, data_y)
                else:
                    latent_codes = generate_latent_codes(args, data_x.shape[0])
                # update generator weights
                optimizer.zero_grad()
                _, y, _ = actor_critic.act(data_x, latent_codes, deterministic=True)
                loss = criterion(y, data_y)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            # end of epoch
            if epochs < 10 or epoch % (epochs//10) == 0:  # every 10%
                print(f' End of pretraining epoch {epoch}, loss={np.mean(losses) / len(losses):.3e}')
        # save
        torch.save([
            actor_critic,
            None,
            None,
            get_vec_normalize(envs).ob_rms
        ], self.save_filename)
