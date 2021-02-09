import torch
import numpy as np


src_root = '/Users/arash'
all_src = ['Downloads/ant_five_goals']
all_dst = ['trajs_antgoal_5']

for src, dst in zip(all_src, all_dst):
    expert = dict()
    expert['states'] = torch.tensor(np.load(f'{src_root}/{src}/SA_arr/obs_arr_final.npy'), dtype=torch.float32)
    expert['actions'] = torch.tensor(np.load(f'{src_root}/{src}/SA_arr/actions_arr_final.npy'), dtype=torch.float32)
    expert['lengths'] = torch.tensor([len(expert['states'][0])] * len(expert['states']))

    torch.save(expert, f'gail_experts/{dst}.pt') #TODO upload our good experts to a google drive
