import argparse
import datetime
import gym
import numpy as np
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter

from src.algos.sac import SAC
from src.environments.wrappers.non_stationary_cheetah_windvel_wrapper import NonStationaryCheetahWindVelEnv
from src.environments.wrappers.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from src.utils.replay_memory import ReplayMemory


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v3",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='run on CUDA (default: False)')
    parser.add_argument('--save_episodes', type=int, default=2, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    args0 = parser.parse_args()
    return args0


def collect_rollouts(env, arguments):

    # Training Loop
    total_steps = 0

    while total_steps < arguments.num_steps:

        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            action = env.action_space.sample()  # Sample random action

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            replay_memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            state = next_state



def main(arguments):

    # Tesnorboard
    writer = SummaryWriter(f'runs/'f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_VAE')

    # Environment
    max_episode_steps = 100
    writer.add_scalar(tag='env/max_env_steps', scalar_value=max_episode_steps)

    env = gym.make('HalfCheetah-v3')
    env._max_episode_steps = max_episode_steps

    # env = NonStationaryCheetahWindVelEnv(env=env, change_freq=20000, renewal=True, summary_writer=writer)
    env = StationaryCheetahWindVelEnv(env=env, summary_writer=writer)
    env._max_episode_steps = max_episode_steps

    env.seed(arguments.seed)
    env.action_space.seed(arguments.seed)

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    # Memory
    replay_memory = ReplayMemory(arguments.replay_size, arguments.seed)






    env.close()



if __name__ == "__main__":
    args = parse_args()
    main(args)