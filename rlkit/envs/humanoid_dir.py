#!/usr/bin/env python3

import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv

# from . import register_env

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


# @register_env('humanoid-dir')
class HumanoidDirEnv(HumanoidEnv):

    def __init__(self, n_tasks=6): #, noise: float = 0):
        # self.tasks = self.sample_tasks(n_tasks)
        # self.reset_task(0)
        # self.goal = goal # in radians
        # self.noise = noise
        # print(f"HumanoidDir-v0 direction: {self._goal}")
        self.max_steps = 1000
        self.step_num = 0  # how many steps passed since environment reset
        self.tasks = np.linspace(0, 2*np.pi, n_tasks + 1)[:-1]  # self.sample_tasks(n_tasks)
        super(HumanoidDirEnv, self).__init__()

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal):
        self._goal = goal
        # print(f"HumanoidDir-v0 direction: {self._goal}")

    # @property
    # def noise(self):
    #     return self._noise

    # @noise.setter
    # def noise(self, noise):
    #     self._noise = noise
    #     print(f"HumanoidDir-v0 action noise std: {self._noise}")


    def step(self, action):
        self.step_num += 1
        
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        # if self.noise:
        #     # print(action)
        #     # print(action.shape)
        #     # print(np.random.randn(action.shape))
        #     action = action + np.random.normal(0, self.noise, action.shape)
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2] # x, y

        alive_bonus = 5.0

        data = self.sim.data
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

        rewards = []
        for task in self.tasks:
            goal_direction = (np.cos(task), np.sin(task))
            lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
            reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            rewards.append(reward)

        qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        done = bool((qpos[2] < 0.8) or (qpos[2] > 2.2)) or (self.step_num >= self.max_steps)

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost,
                                                   rewards=rewards)

    def _get_obs(self):
        data = self.sim.data
        # for q in [data.qpos.flat[2:],
        #                        data.qvel.flat,
        #                        data.cinert.flat,
        #                        data.cvel.flat,
        #                        data.qfrc_actuator.flat,
        #                        data.cfrc_ext.flat]:
        #     print(len(q))
        return np.concatenate([data.qpos.flat[2:],      # 22
                               data.qvel.flat,          # 23
                               data.cinert.flat,        # 140
                               data.cvel.flat,          # 84
                               data.qfrc_actuator.flat, # 23
                               data.cfrc_ext.flat])     # 84

    # def get_all_task_idx(self):
    #     return range(len(self.tasks))

    # def reset_task(self, idx):
    #     self._task = self.tasks[idx]
    #     self._goal = self._task['goal'] # assume parameterization of task by single vector

    # def sample_tasks(self, num_tasks):
    #     # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
    #     directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
    #     tasks = [{'goal': d} for d in directions]
    #     return tasks

    def reset(self):
        self.step_num = 0
        return super().reset()