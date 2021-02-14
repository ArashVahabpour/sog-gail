import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm
import sys, os, inspect
import os.path as osp
import argparse

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)

# sys.path.insert(0, parentdir)
# sys.path.insert(0, os.path.dirname(parentdir))
# print(sys.path)
from a2c_ppo_acktr.algo.behavior_clone import MlpPolicyNet, create_dataset
from a2c_ppo_acktr.algo.gail import ExpertDataset
import matplotlib.pyplot as plt 

from utilities import to_tensor, save_checkpoint, onehot
import wandb


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(
        self, file_name, num_trajectories=4, subsample_frequency=20, mode="Traj"
    ):
        """
        Mode:
        1. Trajectory based: each sample will return one traj of (state, action) pairs
        2. State based: each sample will return (state, action) pair
        """
        self.mode = mode
        all_trajectories = torch.load(file_name)

        perm = torch.randperm(all_trajectories["states"].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}

        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories,)
        ).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k == "radii":
                continue
            if k != "lengths":
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i] :: subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.length = self.trajectories["lengths"].sum().item()

        if mode == "state":
            for i in range(num_trajectories):
                self.get_idx.append((i, i))
        else:
            traj_idx = 0
            i = 0
            self.get_idx = []

            for j in range(self.length):

                # when `i` grows beyond one of the trajectories, increment the `traj_idx` and accordingly set back `i`
                while self.trajectories["lengths"][traj_idx].item() <= i:
                    i -= self.trajectories["lengths"][traj_idx].item()
                    traj_idx += 1

                self.get_idx.append((traj_idx, i))
                i += 1

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.mode == "state":
            traj_idx, i = self.get_idx[i]

            return (
                self.trajectories["states"][traj_idx][i],
                self.trajectories["actions"][traj_idx][i],
            )

        traj_idx, _ = self.get_idx[i]

        return (
            self.trajectories["states"][traj_idx][:],
            self.trajectories["actions"][traj_idx][:],
        )


def create_train_val_split(dataset, batch_size=16, shuffle=True, validation_split=0.1):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    random_seed = 1
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler
    )
    return train_loader, validation_loader


"""
class Encoder(nn.Module):
    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if USE_CUDA:
            eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std
"""
# class Latentcode_sampler(nn.Module):
## refer: https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if USE_CUDA:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, categorical_dim=3, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)


class EncoderRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size=3, n_layers=1, bidirectional=True
    ):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.activation = F.relu
        # For LSTM input: (seq_len, batch, input_size)
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.att_layer = nn.Linear(hidden_size, hidden_size)
        self.uw_layer = nn.Linear(hidden_size, 1)
        self.linear_layer = nn.Linear(hidden_size, output_size)
        # self.softmax = torch.nn.Softmax()

        """
        if self.bidirectional:
            self.h2z = nn.Linear(hidden_size*2, output_size)
        else:
            self.h2z = nn.Linear(hidden_size, output_size)
        """

    def forward(self, input, temp, hard):
        outputs, (h_n, c_n) = self.lstm(input)
        # outputs shape: (batch, seq_len, num_directions * hidden_size)
        # print("output shape from bidirectional LSTM:")
        ## elementwise sum of hidden states from backward and forward part
        new_h = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        ### ( batch,  seq_len,hidden_size),
        # new_h = new_h.reshape(-1, self.hidden_size)
        ut = torch.tanh(self.att_layer(new_h))
        uw = self.uw_layer(ut)
        alpha = torch.nn.functional.softmax(uw, dim=1)  ## batch, seq_len, 1

        r = torch.sum(alpha * new_h, dim=1)  ## batch, hidden size
        linear_r = self.linear_layer(r)
        posterior_dist = torch.nn.functional.softmax(linear_r, dim=-1)
        # latent_code
        # mu, logvar = torch.chunk(ps, 2, dim=1)
        z = gumbel_softmax(
            linear_r, temperature=temp, categorical_dim=self.output_size, hard=hard
        )
        """
        z = gumbel_softmax(
            poster_dist,
            temperature=temp,
            categorical_dim=self.output_size,
            hard=hard,
        )
        """
        # z = self.sample(mu, logvar)
        return (z, posterior_dist)


class VAE_BC(nn.Module):
    def __init__(
        self,
        epochs=300,
        lr=0.0001,
        eps=1e-5,
        device="cpu",
        validate_freq=1,
        checkpoint_dir=".",
        code_dim=3,
        input_size_sa=12,
        input_size_state=10,
        hidden_size=128,
        kld_weight=1
    ):
        super(VAE_BC, self).__init__()
        self.epochs = epochs
        self.device = device
        self.validate_freq = validate_freq

        self.code_dim = code_dim
        self.encoder = EncoderRNN(
            input_size_sa, hidden_size=hidden_size, output_size=code_dim
        ).to(device)
        self.decoder = MlpPolicyNet(
            state_dim=input_size_state,
            code_dim=code_dim,
            output_dim=input_size_sa - input_size_state,
            ft_dim=hidden_size,
        ).to(device)

        self.checkpoint_dir = checkpoint_dir
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            eps=eps,
        )
        self.kld_weight = kld_weight

   
    # def forward_GG(self, inputs, temp, hard=True):
    #     """
    #     Encode the whole trajectory into one latent code.
    #     Decode each state with given latent code into action.
    #     """
    #     latent_code, z = self.encoder(inputs, temp, hard)
    #     # n_steps = inputs.size(0)
    #     input_fts = inputs.size(-1)
    #     ## reshape as a batch forward
    #     inputs = inputs.view(-1, input_fts)
    #     ## TODO: double check whether this is right!!!
    #     outputs_a = self.mlp_policy_net(torch.cat([inputs, latent_code]))
    #     return latent_code, z, outputs_a
    
    def train(self, expert_loader, val_loader, hard=True):
        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            print("\nEpoch: %d" % epoch)
            if epoch % self.validate_freq == 0:
                best_loss, checkpoint_path = validate(
                    epoch,
                    self,
                    val_loader,
                    self.device,
                    best_loss,
                    hard,
                    self.checkpoint_dir,
                )
            train(epoch, self, expert_loader, self.optimizer, self.device, hard)
            
        # self.load_best_checkpoint(checkpoint_path)

    def load_best_checkpoint(self, checkpoint_path):
        print("TO BE IMPLEMENTED")
        self.decoder.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        self.encoder.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def compute_loss(self, recon_a, gt_a, q):
        """
        MSE + KLD
        """
        # KLD = (-0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        MSE_loss = self.criterion(recon_a, gt_a)

        log_ratio = torch.log(q * self.code_dim + 1e-20)
        KLD = torch.sum(q * log_ratio, dim=-1).mean()
        return MSE_loss + KLD * self.kld_weight, MSE_loss, KLD

def visualize_circle(state, action, ax):
    action = action.squeeze()
    scale = 1
    state = state.numpy()
    action = action
    ax.quiver(state[:,-2], state[:,-1], action[:,0]*scale, action[:,1]*scale, angles='xy', color='g')
    ax.quiver(state[:-1,-2], state[:-1,-1], (state[1:,-2]- state[:-1,-2]) * scale, (state[1:,-1] - state[:-1,-1])*scale, angles='xy',color="r")
    print("visualized actions", action, state)


# TODO: Add adpat weight mechanisms
def validate(epoch, net, val_loader, device, best_loss, hard, checkpoint_dir):
    net.encoder.eval()
    net.decoder.eval()
    valid_loss = 0
    temp = 1
    temp_min = 0.5
    ANNEAL_RATE = 0.00003

    avg_valid_loss = best_loss + 2
    number_batches = len(val_loader)
    ## generate all codes for BC envless inference
    
    
    for batch_idx, (traj_state, traj_action) in enumerate(val_loader):
        traj_state = to_tensor(traj_state, device)
        traj_action = to_tensor(traj_action, device)
        data_input = torch.cat([traj_state, traj_action], axis=2)

        latent_code, z = net.encoder(data_input, temp, hard=hard)
        latent_code_tuple = latent_code.unsqueeze(1).repeat((1, traj_state.shape[1], 1))

        decoded_actions = net.decoder(traj_state, latent_code_tuple)

        loss, mse, kld = net.compute_loss(decoded_actions, traj_action, z)

        valid_loss += loss.item()
        avg_valid_loss = valid_loss / (batch_idx + 1)

        wandb.log({"val_loss": avg_valid_loss, "val_mse_loss": mse, "val_kld_loss": kld, "val_gumbel_temp": temp})


    latent_dim = net.encoder.output_size
    all_latent_codes = torch.eye(latent_dim, device=device)
    #plt.figure()
    print("latent_dim", latent_dim)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    vis_state_length = 50
    for batch_idx, (traj_state, traj_action) in enumerate(val_loader):
        if batch_idx == 0:
            for j in range(latent_dim):
                latent_code = all_latent_codes[j:j+1]
                input_states = to_tensor(traj_state[0:1, :vis_state_length, :], device)
                latent_code_tuple = latent_code.unsqueeze(1).repeat((1, input_states.size(1), 1))
                decoded_actions = net.decoder(input_states, latent_code_tuple)
                visualize_circle(traj_state[0, :vis_state_length, :], decoded_actions.cpu().detach().numpy(), ax[j])
        else:
            break
    #plt.show()
    plt.savefig("decode_action.png")
    #wandb.log({"decoded actions": plt}) ## failed don't know why
    wandb.log({"decoded actions": wandb.Image("decode_action.png")})
    plt.close()


    checkpoint_path = osp.join(checkpoint_dir, "checkpoints/bestvae_bc_model.pth")
    if avg_valid_loss <= best_loss:
        best_loss = avg_valid_loss
        print("Best epoch: " + str(epoch))
        # TODO: save state_dict instead of the encoder/decoder objects
        save_checkpoint({'epoch': epoch,
                         'avg_loss': avg_valid_loss,
                         'state_dict_encoder': net.encoder.state_dict(),
                         'state_dict_decoder': net.decoder.state_dict(),
                         }, save_path=checkpoint_path)
    return best_loss, checkpoint_path


def train(epoch, net, dataloader, optimizer, device, hard):
    net.encoder.train()
    net.decoder.train()
    train_loss = 0
    # dataloader
    num_batch = len(dataloader)
    mode_dim = 3
    temp = 1
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    loss = 0

    for batch_idx, (traj_state, traj_action) in enumerate(dataloader):
        optimizer.zero_grad()
        ## traj_state: (seq_len, batch, hidden_size)
        # tmp_input = torch.cat([traj_state.reshape(-1, 10), traj_action.reshape(-1, 2)], axis=1).unsqueeze(1)
        traj_state = to_tensor(traj_state, device)
        traj_action = to_tensor(traj_action, device)
        reshaped_input = torch.cat([traj_state, traj_action], axis=2)
        # reshaped_input = torch.cat([traj_state, traj_action], axis=2).permute(1, 0, 2)
        # print("input shape:", reshaped_input.shape)

        latent_code, z = net.encoder(reshaped_input, temp, hard=hard)
        latent_code_tuple = latent_code.unsqueeze(1).repeat((1, traj_state.shape[1], 1))

        decoded_actions = net.decoder(traj_state, latent_code_tuple)
        ### TODO: double check the last input~!
        loss, mse, kld = net.compute_loss(decoded_actions, traj_action, z)

        loss.backward()
        optimizer.step()
        # print("loss data", loss.data)
        train_loss += loss.item()

        if batch_idx % 10 == 0:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        wandb.log({"train_loss": train_loss / (batch_idx + 1), "train_gumbel_temp": temp, "train_mse_loss": mse, "train_kld_loss": kld})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC VAE Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="tau(temperature) (default: 1.0)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--hard", action="store_true", default=False, help="hard Gumbel softmax"
    )
    parser.add_argument(
        "--kld-weight", type=float, default=1, help="KLD loss weight in vae"
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        USE_CUDA = True

    ############### Train ###############
    wandb.init(project="VAE-BC")

    args.epochs = 30
    args.lr = 1e-4
    args.eps = 1e-5
    args.batch_size = 16

    # args.model_log_dir = "/mnt/SSD3/Qiujing_exp/pytorch-a2c-ppo-acktr-gail/logs/BC_VAE"

    ##
    # /home/shared/gail_experts
    
    ## -------------------Set up for circle env -------------------##
    ##from arash 2021-2-1, 500 traj with shape (500, 1000, 10)
    args.train_data_path = "/mnt/SSD4/tmp_exp_gail/pytorch-a2c-ppo-acktr-gail/final_train_data/trajs_circles.pt"
    expert_dataset = ExpertDataset(
        args.train_data_path, num_trajectories=500, subsample_frequency=20
    )
    args.code_dim = 3
    args.sa_dim = (10, 2)  ## 10 + 2
    args.data_name = "circle"
    """

    ## -------------------Set up for cheetah vel -------------------##
    args.train_data_path = "/home/shared/gail_experts/trajs_halfcheetahvel.pt"
    expert_dataset = ExpertDataset(
        args.train_data_path, num_trajectories=2100, subsample_frequency=4
    )
    args.code_dim = 30
    args.sa_dim = (20, 6)  ## 20+6
    args.data_name = "cheetah-vel"

   
    ## -------------------Set up for cheetah dir -------------------##
    
    args.train_data_path = "/home/shared/gail_experts/trajs_halfcheetahdir.pt"
    expert_dataset = ExpertDataset(
        args.train_data_path, num_trajectories=2000, subsample_frequency=4
    )
    args.code_dim = 2
    args.sa_dim = (20, 6)  ## 20+6
    args.data_name = "cheetah-dir"
    
    ##
    ## -------------------Set up for ant dir -------------------##
    args.train_data_path = "/home/shared/gail_experts/trajs_antdir.pt"
    expert_dataset = ExpertDataset(
        args.train_data_path, num_trajectories=2000, subsample_frequency=4
    )
    args.code_dim = 2
    args.sa_dim = (27, 8)  ## 20+6
    args.data_name = "ant-dir"
    """

    # bc = VAE_BC(epochs=30, lr=1e-4, eps=1e-5, device="cuda:0", code_dim=None)
    # bc = VAE_BC(epochs=30, lr=1e-4, eps=1e-5, device="cuda:0", code_dim=3)
    # wandb.config.train_data_path = "/home/shared/gail_experts/trajs_halfcheetahvel.pt"
    wandb.config = args
    wandb.config.checkpoint_dir = os.path.join(
        "vae_bc_final_ckp", wandb.config.data_name
    )
    if not os.path.exists(wandb.config.checkpoint_dir):
        os.makedirs(wandb.config.checkpoint_dir)
    bc = VAE_BC(
        args.epochs,
        args.lr,
        args.eps,
        device="cuda:0",
        code_dim=args.code_dim,
        input_size_sa=args.sa_dim[0] + args.sa_dim[1],
        input_size_state=args.sa_dim[0],
        checkpoint_dir=wandb.config.checkpoint_dir,
        kld_weight=args.kld_weight
    )

    train_loader, val_loader = create_train_val_split(
        expert_dataset, batch_size=args.batch_size, shuffle=True, validation_split=0.1
    )
    drop_last = len(train_loader) > args.batch_size
    bc.train(train_loader, val_loader, hard=args.hard)
