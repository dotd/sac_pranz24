import numpy as np
from gym import Env


def sample_trajectory(env: Env, max_env_steps):

    # initialize env for the beginning of a new rollout
    obs = env.reset()

    # init vars
    all_obs, all_actions, all_rewards, all_next_obs, all_terminals = [], [], [], [], []
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
        all_terminals.append(done)

        if rollout_done:
            break

    return np.asarray(all_obs), \
           np.asarray(all_actions), \
           np.asarray(all_rewards), \
           np.asarray(all_next_obs), \
           np.asarray(all_terminals)

def sample_trajectories(env, time_steps, max_env_steps):

    curr_time_steps = 0
    episodes = []

    while curr_time_steps < time_steps:
        episodes.append(sample_trajectory(env, max_env_steps))
        curr_time_steps += len(episodes[-1]['reward'])

    return episodes, curr_time_steps

def sample_n_trajectories(env, trajectory_num, max_env_steps):

    episodes = []
    for n in range(trajectory_num):
        episodes.append(sample_trajectory(env, max_env_steps))

    return episodes

def Episode(obs, actions, rewards, next_obs, terminals):

    return {"observations" : np.array(obs, dtype=np.float32),
            "rewards" : np.array(rewards, dtype=np.float32),
            "actions" : np.array(actions, dtype=np.float32),
            "next_observations": np.array(next_obs, dtype=np.float32),
            "terminals": np.array(terminals, dtype=np.float32)}