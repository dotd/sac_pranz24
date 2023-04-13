import numpy as np
import gym
from gym import Wrapper, spaces
import random
# Same environment as in LILAC for Half cheetah Windvel
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig


class StationaryCheetahWindVelEnv(Wrapper):

    def __init__(self,
                 env: gym.Env,
                 config: BaseConfig,
                 logger: SummaryWriter):

        super(StationaryCheetahWindVelEnv, self).__init__(env)

        self.config: BaseConfig = config
        self.logger = logger
        self.action_dim: np.ndarray = self.env.action_space.shape[0]

        self.default_target_vel: float = (config.env.high_target_vel + config.env.low_target_vel) / 2
        self.default_wind_frc: float = (config.env.high_wind_frc + config.env.low_wind_frc) / 2

        self.task_space = spaces.Box(low=np.array ([config.env.low_target_vel, config.env.low_wind_frc], dtype=np.float32),
                                     high=np.array([config.env.high_target_vel, config.env.high_wind_frc], dtype=np.float32),
                                     dtype=np.float32,
                                     seed=config.seed)

        self._max_episode_steps = config.train_buffer.max_episode_len
        self.task: np.ndarray = np.asarray([self.default_target_vel, self.default_wind_frc])

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