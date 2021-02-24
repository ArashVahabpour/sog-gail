import torch
import numpy as np
import numpy.linalg as LA
import os
import gym
import gym_sog


class CirclesExpert:
    def __init__(self, args):
        self.args = args
        self.env_name = args.env_name

    def policy(self, state, radius):
        """
        Expert policy function

        Args:
            states
            radius

        Returns:
            actions
        """

        max_ac_mag = self.args.max_ac_mag  # max action magnitude

        delta_theta = 2 * np.pi / 100
        start = state[-2:]
        center = np.array([0, radius])
        rot_mat_T = np.array([
            [np.cos(delta_theta), -np.sin(delta_theta)],
            [np.sin(delta_theta), np.cos(delta_theta)]
        ]).T
        radial_dist = (start - center).dot(rot_mat_T)
        circ_dest = radial_dist + center
        circ_speed = circ_dest - start
        length = LA.norm(radial_dist)
        speed = circ_speed - (radial_dist / length) * (length - abs(radius))

        # clip action to fit inside its box
        ac_mag = LA.norm(speed, np.inf)
        if ac_mag > max_ac_mag:
            speed = speed / ac_mag * max_ac_mag

        action = speed
        return action

    def generate_data(self):
        env = gym.make(self.args.env_name, args=self.args, state_len=5)

        num_traj = 500  # number of trajectories
        traj_len = 1000  # length of each trajectory --- WARNING: DO NOT CHANGE THIS TO VALUES LOWER THAN 1000 OR IT CAN CAUSE ISSUES IN GAIL RUN
        expert_data = {'states': [],
                       'actions': [],
                       'radii': [],
                       'lengths': torch.tensor([traj_len] * num_traj, dtype=torch.int32)}

        for traj_id in range(num_traj):
            print('traj #{}'.format(traj_id + 1))

            observation = env.reset()
            step = 0
            states = []
            actions = []
            while step < traj_len:
                radius = env.radius
                action = self.policy(observation, radius)
                states.append(observation)
                observation, reward, done, info = env.step(action)
                actions.append(action)

                step += 1

                if done:
                    # start over a new trajectory hoping that this time it completes
                    observation = env.reset()
                    step = 0
                    states = []
                    actions = []
                    print('warning: an incomplete trajectory occured.')

            expert_data['states'].append(torch.FloatTensor(np.array(states)))
            expert_data['actions'].append(torch.FloatTensor(np.array(actions)))
            expert_data['radii'].append(radius)

        env.close()

        expert_data['states'] = torch.stack(expert_data['states'])
        expert_data['actions'] = torch.stack(expert_data['actions'])
        expert_data['radii'] = torch.tensor(expert_data['radii'])

        data_dir = self.args.gail_experts_dir
        os.makedirs(data_dir, exist_ok=True)
        filename = 'trajs_{}.pt'.format(self.args.env_name.split('-')[0].lower())
        torch.save(expert_data, os.path.join(data_dir, filename))
        print('expert data saved successfully.')
