import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
# from . import register_env


# @register_env('ant-dir')
class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        self.max_steps = 200
        self.step_num = 0  # how many steps passed since environment reset
        super(AntDirEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.step_num += 1
        torso_xyz_before = np.array(self.get_body_com("torso"))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        rewards = []
        for task in self.tasks:
            direct = (np.cos(task), np.sin(task))
            forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward
            rewards.append(reward)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 #and state[2] <= 1.0
        done = not notdone or self.step_num >= self.max_steps
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            xpos=np.array(self.get_body_com("torso"))[:2],
            rewards=rewards
        )

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            velocities = np.array([0., np.pi])
        else:
            velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks

    def reset(self):
        self.step_num = 0
        return super().reset()
