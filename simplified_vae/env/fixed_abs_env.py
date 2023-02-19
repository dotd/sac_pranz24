import gym
import numpy as np
import scipy.stats as sps
import torch
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding


class FixedSingleWheelEnv(gym.Env):
    """ Simple Model of the Single Wheel system dynamics.
    Model includes the actuator dynamics represented by the double low-pass filter.
    Explicit Euler method is used for integration.
    A system state consist of {`omega`, `pressure_f`, `pressure_ff`}, where `pressure_f`, `pressure_ff`
    are low-pass filtered and double low-pass filtered pressure respectively.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 cp_brake=43.,
                 r_wheel=0.3657,
                 j_wheel=2.3120,
                 fn_vehicle=1e4,
                 vx_vehicle=30.,
                 lp_filter_1=0.02,
                 lp_filter_2=0.066,
                 tire_b=12, # 32.609
                 tire_c=2.3, # 1.533
                 tire_d=1., # 1.3
                 tire_e=0.97, # 0.8
                 dt_sim=1e-3,
                 T_sim=1.0,
                 seed=0,
                 max_episode_steps=500):

        """ Constructor.
        Parameters
        ----------
        cp_brake: float
            Brake pressure coefficient [n/a].
        r_wheel: float
            Wheel radius [$m$].
        j_wheel: float
            Wheel moment of inertia [$kg m^2$].
        fn_vehicle: float
            Normal force (i.e. downward force from car weight) [$N$].
        vx_vehicle: float
            Vehicle speed in x direction [$m/s$].
        lp_filter_1: float
            Hydraulic low-pass filter coefficient [n/a].
        lp_filter_2: float
            Hydraulic low-pass filter coefficient [n/a]
        tire_b: float
            Magic-formula tire B coefficient [n/a].
        tire_c: float
            Magic-formula tire C coefficient [n/a].
        tire_d: float
            Magic-formula tire D coefficient [n/a].
        tire_e: float
            Magic-formula tire E coefficient [n/a].
        dt_sim: float
            Simulation discretization [s].
        T_sim: float
            Simulation time horizon [s].
        """
        super().__init__()

        self._max_episode_steps = max_episode_steps
        self._seed = seed

        # Vehicle parameters
        self.cp_brake = cp_brake
        self.r_wheel = r_wheel
        self.j_wheel = j_wheel
        self.fn_vehicle = fn_vehicle
        self.vx_vehicle = vx_vehicle

        # Actuator parameters
        self.lp_filter_1 = lp_filter_1
        self.lp_filter_2 = lp_filter_2

        # Tires parameters
        self.tire_b = tire_b
        self.tire_c = tire_c
        self.tire_d = tire_d
        self.tire_e = tire_e
        self._task = np.array([tire_b, tire_c, tire_d, tire_e])

        # Simulation parameters
        self.dt_sim = dt_sim
        self.t_sim = 0.
        self.T_sim = T_sim
        self.optimal_slip, self.optimal_friction = self.get_optimal_slip_friction()
        self.state = None

        # Spaces
        self.observation_space = spaces.Box(np.array([0., 0., 0.], dtype=np.float32),
                                            np.array([100., 300., 300.], dtype=np.float32),
                                            dtype=np.float32,
                                            seed=seed)

        self.action_space = spaces.Box(0., 300., shape=(1,), dtype=np.float32, seed=seed)

        # Reward initialization
        self.slip_reward_nd = sps.norm(self.optimal_slip, 0.05)
        self.slip_reward_ct = 1.
        self.friction_reward_nd = sps.norm(self.optimal_friction, 0.05)
        self.friction_reward_ct = 1.

        # Simulation initialization
        self.viewer = None
        self.steps_beyond_done = None

    def seed(self, seed):

        self._seed = seed
        self.observation_space = spaces.Box(np.array([0., 0., 0.], dtype=np.float32),
                                            np.array([100., 300., 300.], dtype=np.float32),
                                            dtype=np.float32,
                                            seed=seed)

        self.action_space = spaces.Box(0., 300., shape=(1,), dtype=np.float32, seed=seed)

    def step(self, action):
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
        state_dot = self.dynamics(self.state, action)

        # Integration
        self.state = self.state + state_dot * self.dt_sim
        self.t_sim += self.dt_sim

        slip, friction = self.get_slip_friction(self.state[0])
        opt_slip, opt_friction = self.get_optimal_slip_friction()
        reward = ((friction - opt_friction) - np.abs(slip - opt_slip))

        # Check for termination
        done = False
        if self.t_sim > self.T_sim or \
           np.abs(slip - opt_slip) < 1e-4 or \
           not self.observation_space.contains(self.state):

            done = True
            # if np.abs(slip - opt_slip) < 1e-4:
            #     print('success!')

        infos = dict(task=self._task)

        return (self.state, reward, done, infos)

    def dynamics(self, state, action):
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
        pressure_f_dot = (np.clip(action, 0., None) - pressure_f) * self.lp_filter_1 / self.dt_sim
        pressure_ff_dot = (pressure_f - pressure_ff) * self.lp_filter_2 / self.dt_sim

        # Tire-road interaction
        slip, friction = self.get_slip_friction(omega)

        # Wheel dynamics
        m_brake = np.where(omega > 0., pressure_ff * self.cp_brake, np.zeros_like(omega))
        omega_dot = (friction * self.fn_vehicle * self.r_wheel - m_brake) / self.j_wheel

        state_dot = np.concatenate([omega_dot, pressure_f_dot, pressure_ff_dot], axis=-1)
        return state_dot

    def get_slip_friction(self, omega):
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
        slip = np.clip((self.vx_vehicle - omega * self.r_wheel) / np.clip(self.vx_vehicle, 1., None), 0., 1.)
        friction = self.slip_friction_curve(slip, self.tire_b, self.tire_c, self.tire_d, self.tire_e)

        return slip, friction

    @staticmethod
    def slip_friction_curve(slip, tire_b, tire_c, tire_d, tire_e):
        """ "Magic" formula for road-tires interaction.
        Parameters
        ----------
        slip: ndarray
            A (..., 1) numpy array with wheel slips.
        tire_b: float
            Magic-formula tire B coefficient.
        tire_c: float
            Magic-formula tire C coefficient.
        tire_d: float
            Magic-formula tire D coefficient.
        tire_e: float
            Magic-formula tire E coefficient.
        Returns
        -------
        friction: ndarray
            A (..., 1) numpy array with wheel frictions.
        """
        return tire_d * np.sin(tire_c * np.arctan(
            (slip * tire_b) - tire_e * ((slip * tire_b) - np.arctan(slip * tire_b)))
        )

    def get_state_0(self):
        """ Get the initial system state for the specified parameters.
        Returns
        -------
        state_0: ndarray
            An (3) numpy array with initial system state.
        """
        omega = self.vx_vehicle / self.r_wheel
        return np.array([omega, 0., 0.], dtype=np.float32)

    def reset(self):

        self.state = self.get_state_0()
        self.t_sim = 0.
        return self.state

    def render(self, mode="human", close=False):
        super(FixedSingleWheelEnv, self).render(mode=mode)

    def close(self):
        if self.viewer:
            self.viewer.close()

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
        friction_grid = self.slip_friction_curve(slip_grid, self.tire_b, self.tire_c, self.tire_d, self.tire_e)
        # return np.argmax(friction_grid), np.max(friction_grid) -- mistake here, changed to:
        return slip_grid[np.argmax(friction_grid)], np.max(friction_grid)

    def get_curve_task(self):
        slip_grid = np.arange(0., 1., 1e-3)
        friction_grid = self.slip_friction_curve(slip_grid, self.tire_b, self.tire_c, self.tire_d, self.tire_e)
        return np.array([slip_grid, friction_grid])

    def set_task(self, task):
        self._task = np.asarray(task)

    def is_goal_state(self):
        slip, friction = self.get_slip_friction(self.state[0])
        opt_slip, opt_friction = self.get_optimal_slip_friction()
        if np.abs(slip - opt_slip) < 1e-4:
            return True
        else:
            return False