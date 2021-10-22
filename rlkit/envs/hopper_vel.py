#!/usr/bin/env python3

import numpy as np
from gym.envs.mujoco.hopper import HopperEnv as HopperEnv

# import wandb


class HopperVelEnv(HopperEnv):
    """Hopper environment with target velocity

    The hopper follows the dynamics from MuJoCo [1], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity.

    [1] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, n_tasks=6, goal: float = 10): #, noise: float = 0):
        self.goal = goal
        # self.noise = noise
        self.max_steps = 1000
        self.step_num = 0  # how many steps passed since environment reset
        self.tasks = np.array([-2. , -1.5,  0.5,  1. ,  1.5,  2. ])
        super(HopperVelEnv, self).__init__()

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal):
        self._goal = goal
        # print(f"HopperVel-v0 target speed: {self._goal}")

    # @property
    # def noise(self):
    #     return self._noise

    # @noise.setter
    # def noise(self, noise):
    #     self._noise = noise
    #     print(f"HopperVel-v0 action noise std: {self._noise}")

    def step(self, action: np.ndarray):
        self.step_num += 1
        
        xposbefore = self.sim.data.qpos[0]
        # if self.noise:
        #     action = action + np.random.normal(0, self.noise, size=action.shape)
        self.do_simulation(action, self.frame_skip)
        xposafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0

        forward_vel = (xposafter - xposbefore) / self.dt
        ctrl_cost = 1e-3 * np.sum(np.square(action))

        observation = self._get_obs()

        rewards = []
        for task in self.tasks:
            goal_vel = task
            forward_reward = 1.0 * (abs(goal_vel) - abs(forward_vel - goal_vel))
            reward = forward_reward - ctrl_cost + alive_bonus
            rewards.append(reward)

        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            # and (np.abs(s[2:]) < 100).all()
            # and (height > 0.7)
            # and (abs(ang) < 0.2)
            and (np.abs(s[2:]) < 125).all()
            and (height > 0.5)
            and (abs(ang) < 0.3)
        ) or self.step_num >= self.max_steps
        infos = dict(
            reward_forward=forward_reward, reward_ctrl=-ctrl_cost, goal_vel=self.goal, forward_vel=forward_vel, rewards=rewards
        )
        # wandb.log(infos, commit=False)
        return (observation, reward, done, infos)

    def reset(self):
        self.step_num = 0
        return super().reset()