import gym
# import torch
import numpy as np

from vae_storage import VAEStorage
from utils import sample_trajectory

def main():

    seed = 12345
    # torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('MountainCar-v0')
    env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    vae_buffer = VAEStorage(obs_dim=obs_dim,
                            action_dim=action_dim)

    trajectory_num = 5

    # Collect Trajectories
    for trajectory_idx in range(trajectory_num):

        prev_states, actions, next_states, rewards, dones = sample_trajectory(env=env, max_env_steps=100)

        vae_buffer.insert(prev_states=prev_states,
                          actions=actions,
                          next_states=next_states,
                          rewards=rewards,
                          dones=dones)

    # Train Simple VAE
    

if __name__ == '__main__':
    main()