import numpy as np
from gym import Env

from simplified_vae.config import Config
from simplified_vae.vae_storage import VAEBuffer


def sample_trajectory(env: Env, max_env_steps):

    # initialize env for the beginning of a new rollout
    obs = env.reset()

    # init vars
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    steps = 0
    while True:

        # use the most recent ob to decide what to do
        all_obs.append(obs)
        curr_action = env.action_space.sample()
        all_actions.append(curr_action)

        # take that action and record results
        obs, reward, done, _ = env.step(curr_action)

        # record result of taking that action
        steps += 1
        all_next_obs.append(obs)
        all_rewards.append(reward)

        # End the rollout if the rollout ended
        # Note that the rollout can end due to done, or due to max_path_length
        rollout_done = done or (steps >= max_env_steps)
        all_dones.append(done)

        if rollout_done:
            break

    return np.asarray(all_obs), \
           np.asarray(all_actions), \
           np.asarray(all_rewards)[:, np.newaxis], \
           np.asarray(all_next_obs), \
           np.asarray(all_dones)[:, np.newaxis]


def collect_trajectories(config: Config, env, vae_buffer: VAEBuffer):

    for trajectory_idx in range(config.epiosde_num):

        obs, actions, rewards, next_obs, dones = sample_trajectory(env=env, max_env_steps=100)

        vae_buffer.insert(obs=obs,
                          actions=actions,
                          rewards=rewards,
                          next_obs=next_obs,
                          dones=dones)

