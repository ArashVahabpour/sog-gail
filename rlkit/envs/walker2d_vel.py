#!/usr/bin/env python3

import numpy as np
from gym.envs.mujoco.walker2d import Walker2dEnv
# import wandb


class Walker2dVelEnv(Walker2dEnv):
    def __init__(self, n_tasks=6, goal: float = 10): #, noise: float = 0):
        self.goal = goal
        # self.noise = noise
        self.max_steps = 1000
        self.step_num = 0  # how many steps passed since environment reset
        self.tasks = np.array([-1.5, -1. , -0.5,  0.5,  1. ,  1.5])
        super(Walker2dVelEnv, self).__init__()

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal):
        self._goal = goal
        # print(f"Walker2dVel-v0 target speed: {self._goal}")

    # @property
    # def noise(self):
    #     return self._noise

    # @noise.setter
    # def noise(self, noise):
    #     self._noise = noise
    #     print(f"Walker2dVel-v0 action noise std: {self._noise}")

    def step(self, action: np.ndarray):
        self.step_num += 1
        
        xposbefore = self.sim.data.qpos[0]
        # if self.noise:
        #     action = action + np.random.normal(0, self.noise, size=action.shape)
        self.do_simulation(action, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0

        forward_vel = (xposafter - xposbefore) / self.dt
        # print("vel:", forward_vel)
        # forward_reward = forward_vel
        ctrl_cost = 1e-3 * np.sum(np.square(action))

        observation = self._get_obs()

        rewards = []
        for task in self.tasks:
            goal_vel = task
            forward_reward = 1.0 * (abs(goal_vel) - abs(forward_vel - goal_vel))
            reward = forward_reward - ctrl_cost + alive_bonus
            rewards.append(reward)
        # done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        done = not (height > 0.5 and height < 2.5 and ang > -1.3 and ang < 1.3) and (self.step_num >= self.max_steps)
        infos = dict(
            reward_forward=forward_reward, reward_ctrl=-ctrl_cost, goal_vel=self.goal, forward_vel=forward_vel, rewards=rewards
        )
        # wandb.log(infos, commit=False)
        return (observation, reward, done, infos)

    def reset(self):
        self.step_num = 0
        return super().reset()
