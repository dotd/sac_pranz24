import numpy as np
from gym import Wrapper, spaces
import random
# Same environment as in LILAC for Half cheetah Windvel


class StationaryCheetahWindVelEnv(Wrapper):

    def __init__(self, env):

        super(StationaryCheetahWindVelEnv, self).__init__(env)

        self.action_dim = self.env.action_space.shape[0]

        self.low_target_vel: float = 0.
        self.high_target_vel: float = 3.
        self.low_wind_frc: float = 0.
        self.high_wind_frc: float = 20.

        self.default_target_vel = (self.high_target_vel + self.low_target_vel) / 2
        self.default_wind_frc = (self.high_wind_frc + self.low_wind_frc) / 2

        self.task_space = spaces.Box(low=np.array ([self.low_target_vel,  self.low_wind_frc], dtype=np.float32),
                                     high=np.array([self.high_target_vel, self.high_wind_frc], dtype=np.float32),
                                     dtype=np.float32)

        self.task = np.asarray([self.default_target_vel, self.default_wind_frc])

    def get_task(self):
        return self.task

    def reset_task(self, task):

        self.task = task
        return self.unwrapped._get_obs()

    def set_task(self, task):
        if task is None:
            self.task = self.task_space.sample()
        else:
            self.task = task

    def step(self, action):

        pos_before = self.unwrapped.sim.data.qpos[0]
        curr_target_vel = self.task[0]
        curr_wind_frec = self.task[1]

        force = [0.] * 5 + [curr_wind_frec]

        for part in self.unwrapped.sim.model._body_name2id.values():
            self.unwrapped.sim.data.xfrc_applied[part, :] = force

        next_obs, reward, done, info = self.env.step(action)

        pos_after = self.unwrapped.sim.data.qpos[0]
        forward_vel = (pos_after - pos_before) / self.unwrapped.dt
        reward -= forward_vel  # remove this term
        reward *= 0.5  # to be consistent with varibad and LILAC control costs in HC velocity
        reward += -1 * abs(forward_vel - curr_target_vel)

        return next_obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 0