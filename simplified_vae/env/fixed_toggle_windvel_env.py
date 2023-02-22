import numpy as np
import gym
from gym import Wrapper
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig


class FixedToggleWindVelEnv(Wrapper):

    def __init__(self,
                 env: gym.Env,
                 config: BaseConfig,
                 logger: SummaryWriter):

        super(FixedToggleWindVelEnv, self).__init__(env)

        self.config: BaseConfig = config
        self.logger: SummaryWriter = logger

        self.action_dim: int = self.env.action_space.shape[0]
        self.counter: int = 0

        self._max_episode_steps: int = config.train_buffer.max_episode_len

        self.task_idx: int = 0
        self.tasks = [np.array([0.16308017, 19.30782]), np.array([1.8980728, 5.800347])]

        self.increments_counter = 0
        self.time_increments = [3385, 6726, 9980, 13209, 16480, 19731, 23083, 26318, 29582, 32869, 36306, 39575, 42807,
                                46147, 49442, 52670, 56109, 59713, 63078, 66509, 69745, 73075, 76502, 79816, 83079,
                                86289, 89545, 92838, 96280, 99818, 103082, 106481, 109711, 113073, 116352, 119553,
                                122879, 126129, 129502, 132811, 136011, 139278, 142679, 145906, 149145, 152548,
                                155769, 159052, 162279, 165820, 169181, 172440, 175648, 178886, 182156,185624,
                                188835, 192114, 195435, 198713, 202080, 205357, 208887, 212178, 215466, 218724,
                                222014, 225262, 228547, 231781, 235001, 238221, 241515, 244821, 248085, 251294,
                                254635, 258043, 261498, 264881, 268308, 271763, 275040, 278289, 281610, 284842,
                                288208, 291596, 295020, 298308, 301805, 305091, 308350, 311657, 315413, 318860,
                                322216, 325424, 328718, 331984, 335282, 338667, 341894, 345224, 348436, 351660,
                                355017, 358257, 361625, 364835, 368050, 371369, 374573, 377857,  381296, 384572,
                                387885, 391087, 394387, 397679, 400964, 404213, 407458, 411049, 414252, 417454, 420976,
                                424196, 427409, 430632, 433992, 437467, 440669, 443924, 447134, 450363, 453587, 456890,
                                460132, 463351, 466620, 469823, 473033, 476675, 479897, 483141, 486471, 489852, 493301,
                                496519, 499830,503368, 506573, 509885, 513270, 516511, 519739, 523029, 526287, 529506,
                                532769, 536021, 539304, 542574, 545811, 549054, 552434, 555662, 558943, 562144, 565478,
                                568718, 571922, 575154, 578381, 581885, 585128, 588361, 591605, 595097, 598396, 601692,
                                605017, 608265, 611518, 614822, 618022, 621243, 624483, 627710, 631010, 634257, 637664,
                                640947, 644200, 647451, 650771, 654024, 657331, 660535, 663793, 667022, 670239, 673513,
                                676779, 680061, 683401, 686815, 690082, 693319, 696581, 699945, 703351, 706717, 709937,
                                713878, 717177, 720385, 723713, 727343, 730594, 733906, 737143, 740366, 743691, 746891,
                                750263, 753537, 756747, 759959, 763263, 766668, 769900, 773482, 776692, 780083, 783333,
                                786541, 789772, 793032, 796388, 799784, 802998, 806271, 809575, 812817, 816221,
                                819453, 822654, 825858, 829171, 832452, 835943, 839420, 842859, 846063, 849400, 852720,
                                856025, 859348, 862779, 866080, 869326, 872602, 875825, 879113, 882313, 885529, 888769,
                                892124, 895450, 898691, 901987, 905191, 908408, 912007, 915241, 918490, 921769, 925003,
                                928267, 931494, 934698, 937917, 941190, 944397, 947648, 950887, 954140, 957350, 960788,
                                964051, 967433, 971005, 974246, 977510, 980829, 984084, 987319, 990651, 994074, 997524,
                                1000822]

        self.next_change = self.time_increments[self.increments_counter]

        self.ep_length: int = 0
        self.cum_rwd: int = 0

    @property
    def current_task(self):
        return self.task

    def get_task(self):
        return self.task

    def get_default_task(self):
        return np.asarray([self.default_target_vel, self.default_wind_frc])

    def set_task(self, task):

        if task is None:
            raise NotImplemented

        self.task = task

    def step(self, action):

        if self.counter == self.next_change and self.counter > 0:

            self.increments_counter += 1
            self.next_change = self.time_increments[self.increments_counter]
            self.task_idx = int(not self.task_idx)
            self.set_task(task=self.tasks[self.task_idx])

            print(f'CHANGED TO TASK {self.current_task} AT STEP {self.counter}!')

            self.logger.add_scalar(tag='env/target_velocity', scalar_value=self.task[0], global_step=self.counter)
            self.logger.add_scalar(tag='env/wind_friction', scalar_value=self.task[1], global_step=self.counter)

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