import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from ..utils import generate_latent_codes
from tqdm import tqdm


class BC:
    """
    Behavioral cloning pretraining of the model, in a conventional way or with SOG
    """
    def __init__(self, agent, save_filename, expert_filename, args):
        self.args = args
        self.save_filename = save_filename
        self.data_loader = self.create_data_loader(expert_filename)
        self.actor_critic = agent.actor_critic
        self.sog = agent.sog

    @staticmethod
    def flatten(x):
        return x.reshape(-1, x.shape[-1])

    def create_data_loader(self, expert_filename):
        expert = torch.load(expert_filename)
        states, actions = self.flatten(expert['states']), self.flatten(expert['actions'])

        dataset = TensorDataset(states, actions)
        data_loader = DataLoader(dataset, batch_size=self.args.bc_batch_size, shuffle=True)

        return data_loader

    def pretrain(self):
        actor_critic = self.actor_critic
        device = self.args.device
        epochs = self.args.bc_epoch

        if os.path.exists(self.save_filename):
            actor_critic.load_state_dict(torch.load(self.save_filename).state_dict())
            print('already pretrained model loaded...')
            return

        print('behavioral cloning pretraining started...')

        losses = []
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.actor_critic.parameters())

        for epoch in tqdm(range(1, epochs + 1)):
            for _, (data_x, data_y) in enumerate(self.data_loader):
                data_x = data_x.to(device)
                data_y = data_y.to(device)

                if not self.args.sog_gail:
                    latent_codes = generate_latent_codes(self.args, data_x.shape[0]).to(device)
                else:
                    latent_codes = self.sog.resolve_latent_code(data_x, data_y)
                # update generator weights
                optimizer.zero_grad()
                _, y, _ = actor_critic.act(data_x, latent_codes, deterministic=True)
                loss = criterion(y, data_y)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            # end of epoch
            if epoch % (epochs//10) == 0:  # every 10%
                print(f'End of pretraining epoch {epoch}, loss={np.mean(losses) / len(losses):.3e}')
        # save
        torch.save(self.actor_critic, self.save_filename)
