from typing import Optional

import gym
import numpy as np
from gym import spaces
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig


class StationaryABSEnv(gym.Env):
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
        self.task: np.ndarray = np.array([config.env.tire_b,
                                          config.env.tire_c,
                                          config.env.tire_d,
                                          config.env.tire_e])

        # Simulation parameters
        self.t_sim: float = 0.
        self.dt_sim: float = config.env.dt_sim
        self.T_sim: float = config.env.T_sim

        self.curr_state: Optional[np.ndarray] = None
        self.optimal_slip: Optional[np.ndarray] = None
        self.optimal_friction: Optional[np.ndarray] = None

        # Spaces
        self.task_space = spaces.Box(low=np.array(config.env.task_lims_low, dtype=np.float32),
                                     high=np.array(config.env.task_lims_high, dtype=np.float32),
                                     dtype=np.float32,
                                     seed=config.seed)

        self.observation_space = spaces.Box(low=np.array(config.env.observation_lims_low, dtype=np.float32),
                                            high=np.array(config.env.obsevation_lims_high, dtype=np.float32),
                                            dtype=np.float32,
                                            seed=config.seed)

        self.action_space = spaces.Box(low=config.env.action_lims_low,
                                       high=config.env.action_lims_high,
                                       shape=(1,),
                                       dtype=np.float32,
                                       seed=config.seed)

        self.obs_dim: int = self.observation_space.shape[0]
        self.max_episode_steps = config.env.max_episode_steps
        self.step_count = 0

        # Simulation initialization
        self.viewer = None

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
        self.step_count = 0
        return self.curr_state

    def seed(self, seed=None):

        self._seed = seed

        self.task_space = spaces.Box(low=np.array(self.config.env.task_lims_low, dtype=np.float32),
                                     high=np.array(self.config.env.task_lims_high, dtype=np.float32),
                                     dtype=np.float32,
                                     seed=self.config.seed)

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

        if self.step_count > self.max_episode_steps:
            done = True
        else:
            done = False

        # if self.t_sim > self.T_sim or \
        #    np.abs(curr_slip - self.optimal_slip) < 1e-4: # or \
        #    # not self.observation_space.contains(self.curr_state):

            # done = True

        infos = dict(task=self.task)
        self.step_count += 1

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
        omega_dot = (friction * self.config.env.fn_vehicle * self.config.env.r_wheel - m_brake) / self.config.env.j_wheel

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
        slip = np.clip((self.config.env.vx_vehicle - omega * self.config.env.r_wheel) / np.clip(self.config.env.vx_vehicle, 1., None), 0., 1.)
        friction = self.get_friction(slip)

        return slip, friction

    def get_friction(self, slip: np.ndarray):

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
        friction_grid = self.get_friction(slip_grid)
        return slip_grid[np.argmax(friction_grid)], np.max(friction_grid)

    def render(self,
               mode: str = "human",
               close: bool = False):

        super(StationaryABSEnv, self).render(mode=mode)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def get_task_curve(self):
        slip_grid = np.arange(0., 1., 1e-3)
        friction_grid = self.get_friction(slip_grid)
        return np.array([slip_grid, friction_grid])

    def set_task(self, task: np.ndarray):
        if task is None:
            self.task = self.task_space.sample()
        else:
            self.task = task

        self.optimal_slip, self.optimal_friction = self.get_optimal_slip_friction()

    def get_task(self):
        return self.task

