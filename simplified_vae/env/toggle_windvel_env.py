import random
from typing import List

import gym
from gym import Wrapper
import numpy as np
# import wandb
from gym import spaces

# Same environment as in LILAC for Half cheetah Windvel
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import Config


def change_increments(change_freq):
    return int(np.ceil(np.log(1 - random.random()) / np.log(1 - (1 / change_freq)) - 1))


class ToggleWindVelEnv(Wrapper):

    def __init__(self,
                 env: gym.Env,
                 config: Config):

        super(ToggleWindVelEnv, self).__init__(env)

        self.config: Config = config
        self.summary_writer = config.logger

        self.change_freq: int = config.task.env_change_freq
        self.next_change: int = change_increments(self.change_freq)  # for poisson_dist process only
        self.action_dim: int = self.env.action_space.shape[0]
        self.counter: int = 0

        self._max_episode_steps: int = config.train_buffer.max_episode_len

        self.low_target_vel: float = config.task.low_target_vel
        self.high_target_vel: float = config.task.high_target_vel
        self.low_wind_frc: float = config.task.low_wind_frc
        self.high_wind_frc: float = config.task.high_wind_frc

        self.default_target_vel: float = (self.high_target_vel + self.low_target_vel) / 2
        self.default_wind_frc: float = (self.high_wind_frc + self.low_wind_frc) / 2

        self.task = np.asarray([self.default_target_vel, self.default_wind_frc])

        self.task_space = spaces.Box(low=np.array ([self.low_target_vel,  self.low_wind_frc], dtype=np.float32),
                                     high=np.array([self.high_target_vel, self.high_wind_frc], dtype=np.float32),
                                     dtype=np.float32, seed=config.seed)

        self.tasks: List = [self.task_space.sample(), self.task_space.sample()]
        self.task_idx: int = 1

        self.ep_length: int = 0
        self.cum_rwd: int = 0
        self.poisson_dist: bool = config.task.poisson_dist  # whether or not to generate random time changes

    @property
    def current_task(self):
        return self.task

    def get_task(self):
        return self.task

    def get_default_task(self):
        return np.asarray([self.default_target_vel, self.default_wind_frc])

    def set_task(self, task):
        if task is None:
            self.task = self.task_space.sample()
        else:
            self.task = task

    def step(self, action):

        if self.counter - self.next_change == 0 and self.counter > 0 and self.poisson_dist:
            jump = change_increments(self.change_freq)
            if jump == 0:
                jump += 1

            self.next_change += jump
            self.task_idx = int(not self.task_idx)
            self.set_task(task=self.tasks[self.task_idx])

            print("CHANGED TO TASK {} AT STEP {}!".format(self.current_task, self.counter))

            self.summary_writer.add_scalar(tag='env/target_velocity', scalar_value=self.task[0], global_step=self.counter)
            self.summary_writer.add_scalar(tag='env/wind_friction', scalar_value=self.task[1], global_step=self.counter)

        if self.counter % self.change_freq == 0 and self.counter > 0 and not self.poisson_dist:
            self.task_idx = int(not self.task_idx)
            self.set_task(task=self.tasks[self.task_idx])
            print("CHANGED TO TASK {} AT STEP {}!".format(self.current_task, self.counter))

            self.summary_writer.add_scalar(tag='env/target_velocity', scalar_value=self.task[0], global_step=self.counter)
            self.summary_writer.add_scalar(tag='env/wind_friction', scalar_value=self.task[1], global_step=self.counter)

        curr_target_vel = self.task[0]
        curr_wind_frec = self.task[1]

        pos_before = self.unwrapped.sim.data.qpos[0]
        force = [0.] * 5 + [curr_wind_frec]

        for part in self.unwrapped.sim.model._body_name2id.values():
            self.unwrapped.sim.data.xfrc_applied[part, :] = force

        next_obs, reward, done, info = self.env.step(action)

        info['task'] = self.task

        pos_after = self.unwrapped.sim.data.qpos[0]
        forward_vel = (pos_after - pos_before) / self.unwrapped.dt
        reward -= forward_vel  # remove this term
        reward *= 0.5  # to be consistent with varibad and LILAC control costs in HC velocity
        reward += -1 * abs(forward_vel - curr_target_vel)

        self.ep_length += 1
        self.cum_rwd += reward
        if done:
            info['episode'] = dict({})
            info['episode']['l'] = self.ep_length
            info['episode']['r'] = self.cum_rwd
            self.ep_length = self.cum_rwd = 0

        self.counter += 1

        return next_obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 0