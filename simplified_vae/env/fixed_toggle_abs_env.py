from typing import Optional, List

import gym
import numpy as np
from gym import spaces
import random

from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig


def change_increments(change_freq):
    return int(np.ceil(np.log(1 - random.random()) / np.log(1 - (1 / change_freq)) - 1))


class FixedToggleABSEnv(gym.Env):
    """ Simple Model of the Single Wheel system dynamics.
    Model includes the actuator dynamics represented by the double low-pass filter.
    Explicit Euler method is used for integration.
    A system state consist of {`omega`, `pressure_f`, `pressure_ff`}, where `pressure_f`, `pressure_ff`
    are low-pass filtered and double low-pass filtered pressure respectively.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 config: BaseConfig,
                 logger: SummaryWriter):

        super().__init__()

        self.config: BaseConfig = config
        self.logger = logger
        self._seed: int = config.seed

        self.change_freq: int = int((config.cpd.cusum_window_length + config.cpd.env_window_delta) * config.cpd.freq_multiplier)
        self.next_change: int = self.change_freq + change_increments(config.cpd.poisson_freq)  # for poisson_dist process only
        self.counter: int = 0
        self.poisson_dist: bool = config.cpd.poisson_dist

        self.task_idx: int = 0
        self.tasks: List = [np.array([12, 2.3, 1, 0.97]), np.array([4, 2.3, 1, 1])]

        self.increments_counter = 0
        self.time_increments = [3385, 6726, 9980, 13209, 16480, 19731, 23083, 26318, 29582, 32869, 36306, 39575, 42807,
                                46147, 49442, 52670, 56109, 59713, 63078, 66509, 69745, 73075, 76502, 79816, 83079,
                                86289, 89545, 92838, 96280, 99818, 103082, 106481, 109711, 113073, 116352, 119553,
                                122879, 126129, 129502, 132811, 136011, 139278, 142679, 145906, 149145, 152548,
                                155769, 159052, 162279, 165820, 169181, 172440, 175648, 178886, 182156, 185624,
                                188835, 192114, 195435, 198713, 202080, 205357, 208887, 212178, 215466, 218724,
                                222014, 225262, 228547, 231781, 235001, 238221, 241515, 244821, 248085, 251294,
                                254635, 258043, 261498, 264881, 268308, 271763, 275040, 278289, 281610, 284842,
                                288208, 291596, 295020, 298308, 301805, 305091, 308350, 311657, 315413, 318860,
                                322216, 325424, 328718, 331984, 335282, 338667, 341894, 345224, 348436, 351660,
                                355017, 358257, 361625, 364835, 368050, 371369, 374573, 377857, 381296, 384572,
                                387885, 391087, 394387, 397679, 400964, 404213, 407458, 411049, 414252, 417454, 420976,
                                424196, 427409, 430632, 433992, 437467, 440669, 443924, 447134, 450363, 453587, 456890,
                                460132, 463351, 466620, 469823, 473033, 476675, 479897, 483141, 486471, 489852, 493301,
                                496519, 499830, 503368, 506573, 509885, 513270, 516511, 519739, 523029, 526287, 529506,
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
        self.task_idx: int = 0
        self.task = self.tasks[self.task_idx]

        # Simulation parameters
        self.t_sim: float = 0.
        self.dt_sim: float = config.env.dt_sim
        self.T_sim: float = config.env.T_sim

        self.curr_state: Optional[np.ndarray] = None
        self.optimal_slip: Optional[np.ndarray] = None
        self.optimal_friction: Optional[np.ndarray] = None

        # Spaces
        self.observation_space = spaces.Box(low=np.array(config.env.observation_lims_low, dtype=np.float32),
                                            high=np.array(config.env.obsevation_lims_high, dtype=np.float32),
                                            dtype=np.float32,
                                            seed=config.seed)

        self.action_space = spaces.Box(low=config.env.action_lims_low,
                                       high=config.env.action_lims_high,
                                       shape=(1,),
                                       dtype=np.float32,
                                       seed=config.seed)

        # Simulation initialization
        self.viewer = None

    @property
    def current_task(self):
        return self.task

    def get_task(self):
        return self.task

    def set_task(self, task: np.ndarray):

        if task is None:
            raise NotImplemented

        self.task = task

    def get_initial_state(self):
        """ Get the initial system state for the specified parameters.
        Returns
        -------
        state_0: ndarray
            An (3) numpy array with initial system state.
        """
        omega = self.config.env.vx_vehicle / self.config.env.r_wheel
        return np.array([omega, 0., 0.], dtype=np.float32)

    def reset(self):

        self.optimal_slip, self.optimal_friction = self.get_optimal_slip_friction()
        self.curr_state = self.get_initial_state()
        self.t_sim = 0.
        return self.curr_state

    def seed(self, seed=None):

        self._seed = seed
        self.observation_space = spaces.Box(low=np.array(self.config.env.observation_lims_low, dtype=np.float32),
                                            high=np.array(self.config.env.obsevation_lims_high, dtype=np.float32),
                                            dtype=np.float32,
                                            seed=self.config.seed)

        self.action_space = spaces.Box(low=self.config.env.action_lims_low,
                                       high=self.config.env.action_lims_high,
                                       shape=(1,),
                                       dtype=np.float32,
                                       seed=self.config.seed)

    def step(self, action: float):
        """ Step forward through the system dynamics.
        Given the current system state and control signal compute the next system state for the fixed `dt`.
        Forward Euler method is used for integration.
        Parameters
        ----------
        action: float
            A numpy array with Actions.
        Returns
        -------
        state: ndarray
            A (3) array with updated system state.
        reward: float
            Reward for state and action.
        done: bool
            Termination flag.
        info: dict
            Dictionary with additional information.
        """

        if self.counter == self.next_change and self.counter > 0:
            self.increments_counter += 1
            self.next_change = self.time_increments[self.increments_counter]
            self.task_idx = int(not self.task_idx)
            self.set_task(task=self.tasks[self.task_idx])
            self.optimal_slip, self.optimal_friction = self.get_optimal_slip_friction()

            print(f'CHANGED TO TASK {self.current_task} AT STEP {self.counter}!')

            self.logger.add_scalar(tag='env/tire_b', scalar_value=self.task[0], global_step=self.counter)
            self.logger.add_scalar(tag='env/tire_c', scalar_value=self.task[1], global_step=self.counter)
            self.logger.add_scalar(tag='env/tire_d', scalar_value=self.task[2], global_step=self.counter)
            self.logger.add_scalar(tag='env/tire_e', scalar_value=self.task[3], global_step=self.counter)

        action = np.atleast_1d(action).astype(np.float32)

        if not self.action_space.contains(action):
            action = np.clip(action, self.action_space.low, self.action_space.high)
            # raise ValueError("Invalid action {} of type ({})".format(action, type(action)))
        if action < 0:
            print(action)
        # System dynamics
        state_dot = self.dynamics(state=self.curr_state,
                                  action=action)

        # Integration
        self.curr_state = self.curr_state + state_dot * self.dt_sim
        self.t_sim += self.dt_sim

        curr_slip, curr_friction = self.get_slip_friction(omega=self.curr_state[0])
        reward = ((curr_friction - self.optimal_friction) - np.abs(curr_slip - self.optimal_slip))

        # Check for termination
        done = False
        # if self.t_sim > self.T_sim or \
        #         np.abs(curr_slip - self.optimal_slip) < 1e-4: # or \
        #         # not self.observation_space.contains(self.curr_state):
            # done = True

        infos = dict(task=self.task)

        self.counter += 1
        return self.curr_state, reward, done, infos

    def dynamics(self,
                 state: np.ndarray,
                 action: np.ndarray):

        """ Continuous version of system dynamics.
        Parameters
        ----------
        state: ndarray
            A (..., 3) numpy array with system States.
        action: ndarray
            A (..., 1) numpy array with Actions.
        Returns
        -------
        state_dot: ndarray
            A (..., 3) System state derivatives.
        """
        if state.ndim != action.ndim:
            raise ValueError("State ndim {} is not equal to action ndim {}.".format(state.ndim, action.ndim))

        omega, pressure_f, pressure_ff = np.split(state, self.observation_space.shape[0], axis=-1)

        # Actuation dynamics
        pressure_f_dot = (np.clip(action, 0., None) - pressure_f) * self.config.env.lp_filter_1 / self.dt_sim
        pressure_ff_dot = (pressure_f - pressure_ff) * self.config.env.lp_filter_2 / self.dt_sim

        # Tire-road interaction
        slip, friction = self.get_slip_friction(omega=omega)

        # Wheel dynamics
        m_brake = np.where(omega > 0., pressure_ff * self.config.env.cp_brake, np.zeros_like(omega))
        omega_dot = (
                                friction * self.config.env.fn_vehicle * self.config.env.r_wheel - m_brake) / self.config.env.j_wheel

        state_dot = np.concatenate([omega_dot, pressure_f_dot, pressure_ff_dot], axis=-1)
        return state_dot

    def get_slip_friction(self, omega: np.ndarray):

        """ Compute slip and friction for the given angular velocity.
        Parameters
        ----------
        omega: ndarray
            A (..., 1) numpy array with angular velocities.
        Returns
        -------
        slip: ndarray
            A (..., 1) numpy array with Slips.
        friction: ndarray
            A (..., 1) numpy array with Frictions.
        """
        slip = np.clip(
            (self.config.env.vx_vehicle - omega * self.config.env.r_wheel) / np.clip(self.config.env.vx_vehicle, 1.,
                                                                                     None), 0., 1.)
        friction = self.slip_friction_curve(slip)

        return slip, friction

    def slip_friction_curve(self, slip: np.ndarray):

        """ "Magic" formula for road-tires interaction.
        Parameters
        ----------
        slip: ndarray
            A (..., 1) numpy array with wheel slips.
        Returns
        -------
        friction: ndarray
            A (..., 1) numpy array with wheel frictions.
        """
        tire_b, tire_c, tire_d, tire_e = self.task

        return tire_d * np.sin(tire_c * np.arctan(
            (slip * tire_b) - tire_e * ((slip * tire_b) - np.arctan(slip * tire_b))))

    def get_optimal_slip_friction(self):
        """ Perform the grid search to find the optimal slip and friction for current Tire-Road configuration.
        Returns
        -------
        optimal_slip: float
            Optimal slip.
        optimal_friction: float
            Optimal friction.
        """
        slip_grid = np.arange(0., 1., 1e-3)
        friction_grid = self.slip_friction_curve(slip_grid)
        return slip_grid[np.argmax(friction_grid)], np.max(friction_grid)

    def render(self,
               mode: str = "human",
               close: bool = False):

        super(FixedToggleABSEnv, self).render(mode=mode)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def get_curve_task(self):
        slip_grid = np.arange(0., 1., 1e-3)
        friction_grid = self.slip_friction_curve(slip_grid)
        return np.array([slip_grid, friction_grid])


