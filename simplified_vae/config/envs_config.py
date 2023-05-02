from typing import List

from simplified_vae.config.pydantic_config import BaseModel


class EnvConfig(BaseModel):

    name: str = 'HalfCheetah-v2'
    max_episode_steps = 100

    cp_brake: float = None
    r_wheel: float = None
    j_wheel: float = None
    fn_vehicle: float = None
    vx_vehicle: float = None
    lp_filter_1: float = None
    lp_filter_2: float = None
    tire_b: float = None
    tire_c: float = None
    tire_d: float = None
    tire_e: float = None
    dt_sim: float = None
    T_sim: float = None

    low_target_vel: float = None
    high_target_vel: float = None
    low_wind_frc: float = None
    high_wind_frc: float = None


class StationaryCheetahWindvelEnvConfig(EnvConfig):

    name: str = 'HalfCheetah-v2'

    max_episode_steps = 100

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class ToggleCheetahWindvelEnvConfig(EnvConfig):

    name: str = 'HalfCheetah-v2'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class FixedToggleCheetahWindvelEnvConfig(EnvConfig):

    name: str = 'HalfCheetah-v2'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class StationarySwimmerWindvelEnvConfig(EnvConfig):

    name: str = 'Swimmer-v3'

    max_episode_steps = 100

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class ToggleSwimmerWindvelEnvConfig(EnvConfig):

    name: str = 'Swimmer-v2'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class FixedToggleSwimmerWindvelEnvConfig(EnvConfig):

    name: str = 'Swimmer-v2'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class StationaryHopperWindvelEnvConfig(EnvConfig):

    name: str = 'Hopper-v3'

    max_episode_steps = 100

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class FixedToggleHopperWindvelEnvConfig(EnvConfig):

    name: str = 'Hopper-v3'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class ToggleHopperWindvelEnvConfig(EnvConfig):

    name: str = 'Hopper-v3'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class StationaryABSEnvConfig(EnvConfig):

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

    name: str = 'FixedABS'

    cp_brake: float = 43.
    r_wheel: float = 0.3657
    j_wheel: float = 2.3120
    fn_vehicle: float = 1e4
    vx_vehicle: float = 30.
    lp_filter_1: float = 0.02
    lp_filter_2: float = 0.066
    tire_b: float = 4 #12
    tire_c: float = 2.3
    tire_d: float = 1.
    tire_e: float = 1 # 0.97
    dt_sim: float = 1e-3
    T_sim: float = 1.0
    max_episode_steps = 500

    # Spaces
    task_lims_low: List = [4., 1.9, 0.1, 0.97]
    task_lims_high: List = [12., 2.3, 1., 1.]

    observation_lims_low: List = [0., 0., 0.]
    obsevation_lims_high: List = [100., 300., 300.]

    action_lims_low = 0.
    action_lims_high = 300.


class ToggleABSEnvConfig(EnvConfig):

    name: str = 'ToggleABS'

    cp_brake: float = 43.
    r_wheel: float = 0.3657
    j_wheel: float = 2.3120
    fn_vehicle: float = 1e4
    vx_vehicle: float = 30.
    lp_filter_1: float = 0.02
    lp_filter_2: float = 0.066
    tire_b: float = 12  # 32.609
    tire_c: float = 2.3  # 1.533
    tire_d: float = 1.  # 1.3
    tire_e: float = 0.97  # 0.8
    dt_sim: float = 1e-3
    T_sim: float = 1.0
    max_episode_steps = 1000000

    # Spaces
    task_lims_low: List = [4., 1.9, 0.1, 0.97]
    task_lims_high: List = [12., 2.3, 1., 1.]

    observation_lims_low: List = [0., 0., 0.]
    obsevation_lims_high: List = [100., 300., 300.]

    action_lims_low = 0.
    action_lims_high = 300.


class FixedToggleABSEnvConfig(EnvConfig):

    name: str = 'ToggleABS'

    cp_brake: float = 43.
    r_wheel: float = 0.3657
    j_wheel: float = 2.3120
    fn_vehicle: float = 1e4
    vx_vehicle: float = 30.
    lp_filter_1: float = 0.02
    lp_filter_2: float = 0.066
    tire_b: float = 12  # 32.609
    tire_c: float = 2.3  # 1.533
    tire_d: float = 1.  # 1.3
    tire_e: float = 0.97  # 0.8
    dt_sim: float = 1e-3
    T_sim: float = 1.0
    max_episode_steps = 1000000

    # Spaces
    task_lims_low: List = [4., 1.9, 0.1, 0.97]
    task_lims_high: List = [12., 2.3, 1., 1.]

    observation_lims_low: List = [0., 0., 0.]
    obsevation_lims_high: List = [100., 300., 300.]

    action_lims_low = 0.
    action_lims_high = 300.