import gym
from gym import spaces
import numpy as np

gym.logger.set_level(40)


class CirclesEnv(gym.Env):
    """
    Description:
        A moving agent should be moving on perimeter of a circle. If it is off the circle, the agent first moves towards
        the center of the circle, then it moves counter-clockwise around it.

    Observation:
        Type: Box(2*N), where N is `state_len`
        Num        Observation                          Min                    Max
        0          Agent Position X at time t           -L                     L
        1          Agent Position Y at time t           -2L                    2L
        ...
        2N         Agent Position X at time t - N       -L                     L
        2N+1       Agent Position Y at time t - N       -2L                    2L

        |-----|-----|
        |     |     |
        |     2L    |
        |     |     |
        |--L--|--L--|
        |     |     |
        |     2L    |
        |     |     |
        |-----|-----|

    Actions:
        Type: Box(2)
        Num        Observation                          Min                    Max
        0          Agent Velocity X                     -L                     L
        1          Agent Velocity Y                     -L                     L

    Reward:
        We do not defined the reward, as we only work with experts with hidden rewards.

    Starting State:
        All observations are assigned a normal random value in [-0.05..0.05]

    Episode Termination:
        Agent Position is more than 20 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average distance from the circle is less than 0.05 x radius over 100 consecutive
        trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, args, state_len=5):
        super(CirclesEnv, self).__init__()

        self.radii = args.radii
        self.radius = None

        # the agent can move in an area of x, y between boundaries (same as rendering boundaries)
        L = max(map(abs, self.radii)) * 1.5
        self.x_threshold = L
        self.y_threshold = L * 2

        self.max_steps = 2000
        self.step_num = None  # how many steps passed since environment reset

        self.state_len = state_len  # number of consecutive locations to be concatenated as state

        self.max_ac_mag = args.max_ac_mag
        self.action_space = spaces.Box(low=np.array([-self.max_ac_mag, -self.max_ac_mag]),
                                       high=np.array([self.max_ac_mag, self.max_ac_mag]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-self.x_threshold, -self.y_threshold] * self.state_len),
                                            high=np.array([self.x_threshold, self.y_threshold] * self.state_len),
                                            dtype=np.float32)

        self.viewer = None
        self._viewer_geom = {}
        self.is_expert = args.is_train  # draw blue circles only if expert is controlling the environment

        self._init_circle()
        self.loc_history = None  # 2D array of (x, y) locations visited so far in the episode.
        self._init_loc()

        self.steps_beyond_done = None

    @property
    def state(self):
        return self.loc_history[-self.state_len:].ravel()

    def _init_circle(self):
        self.radius = np.random.choice(self.radii)

    def _init_loc(self):
        """Initializes the first `state_len` locations when episode starts
        """
        self.loc_history = np.zeros([self.state_len, 2])

    def manual_init(self, init_loc):
        self.loc_history = init_loc

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        scale = 10  # scale for mapping actual coordinates to pixels
        screen_width = scale * 2 * self.x_threshold
        screen_height = scale * 2 * self.y_threshold

        if self.viewer is None:
            self.viewer = rendering.Viewer(int(screen_width), int(screen_height))

            coordinate_offset = np.array((screen_width / 2, screen_height / 2))
            coordinate_trans = rendering.Transform(translation=coordinate_offset)

            x_axis = rendering.Line((0, screen_height / 2), (screen_width, screen_height / 2))
            y_axis = rendering.Line((screen_width / 2, 0), (screen_width / 2, screen_height))
            x_axis.set_color(0.66, 0.66, 0.66)
            y_axis.set_color(0.66, 0.66, 0.66)
            self.viewer.add_geom(x_axis)
            self.viewer.add_geom(y_axis)

            circles = []
            for radius in self.radii:
                radius_scaled = radius * scale
                circle_offset = (0, radius_scaled)
                circle_trans = rendering.Transform(translation=circle_offset)
                circle = rendering.make_circle(radius_scaled, res=int(screen_width), filled=False)
                circle.add_attr(coordinate_trans)
                circle.add_attr(circle_trans)
                circle.set_color(0.9, 0.9, 0.9)
                self.viewer.add_geom(circle)
                circles.append(circle)

            self._viewer_geom['circles'] = circles

            loc_history_scaled = self.loc_history * scale
            traj = rendering.PolyLine(loc_history_scaled, close=False)
            traj.add_attr(coordinate_trans)
            traj.set_color(1., 0., 0.)
            self.viewer.add_geom(traj)

            self._viewer_geom['traj'] = traj

        # Edit the trajectory vertices + highlight selected circle
        self._viewer_geom['traj'].v = self.loc_history * scale
        for i, radius in enumerate(self.radii):
            circle = self._viewer_geom['circles'][i]
            if self.is_expert and radius == self.radius:
                circle.set_color(0., 0., 1.)
            else:
                circle.set_color(0.9, 0.9, 0.9)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def step(self, action):
        self.step_num += 1

        loc = self.loc_history[-1]
        vel = action  # 2D instantaneous velocity
        new_loc = loc + vel

        new_loc += np.random.randn(2) * self.max_ac_mag * 0.1  # add noise   # TODO: allow control of noise parameters
        self.loc_history = np.vstack([self.loc_history, new_loc])

        x, y = new_loc

        rewards = []  # stores rewards assuming each mode
        if len(self.radii) < 10:
            for radius in self.radii:
                rewards.append(self._reward(loc, new_loc, radius))
            reward = rewards[np.where(np.array(self.radii) == self.radius)[0][0]]

        else:
            reward = None

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or y < -self.y_threshold
            or y > self.y_threshold
            or self.step_num >= self.max_steps
        )

        if done:
            if self.steps_beyond_done is None:
                # Agent just out of boundary!
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    gym.logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive 'done = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_done += 1

        return np.array(self.state), reward, done, {'rewards': rewards}

    def reset(self):
        self._init_circle()
        self._init_loc()
        self.steps_beyond_done = None
        self.step_num = 0

        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @staticmethod
    def _reward(old_loc, new_loc, radius):
        # x_old, y_old = old_loc
        x_new, y_new = new_loc
        # theta, theta_old = np.arctan2(y_new - radius, x_new), np.arctan2(y_old - radius, x_old)
        # delta_theta = (theta - theta_old + np.pi) % (
        #             2 * np.pi) - np.pi  # make sure that the difference is between -pi, pi
        # delta_theta_expert = 2 * np.pi / 100
        r = np.sqrt((y_new - radius) ** 2 + x_new ** 2)
        return np.exp(-10 * ((r - abs(radius)) / abs(radius)) ** 2)
