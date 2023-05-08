from datetime import datetime
import itertools

import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig, AgentConfig
from simplified_vae.config.envs_config import StationaryCheetahWindvelEnvConfig, StationaryHopperWindvelEnvConfig
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.models.sac import SAC
from simplified_vae.utils.env_utils import set_seed
from simplified_vae.utils.logging_utils import write_config
from src.utils.replay_memory import ReplayMemory


def run_agent_and_environment():

    config = BaseConfig(env=StationaryHopperWindvelEnvConfig(),
                        agent=AgentConfig(start_steps=-1))

    logger = SummaryWriter(f'runs/SAC_{config.env.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    write_config(config=config, logdir=logger.log_dir)
    set_seed(config.seed)

    # wandb.init(project="cusum_exps",
    #            config=config.__dict__)

    # Init Env
    env = env_factory(config=config, logger=logger)
    # env.set_task(task=np.array([1.8980728, 5.800347])) # SAC_Hopper-v3_2023-05-07_16-10-41
    env.set_task(task=np.array([0.16308017, 19.30782])) # SAC_Hopper-v3_2023-05-08_08-51-56
    # Agent
    agent = SAC(config=config,
                num_inputs=env.obs_dim,
                action_space=env.action_space)

    # Memory
    replay_memory = ReplayMemory(config.agent.replay_size, config.seed)

    # Training Loop
    total_steps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if config.agent.start_steps > total_steps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(replay_memory) > config.agent.batch_size:
                # Number of updates per step in environment
                for i in range(config.agent.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(replay_memory,
                                                                                                         config.training.batch_size,
                                                                                                         updates)

                    logger.add_scalar('loss/critic_1', critic_1_loss, updates)
                    logger.add_scalar('loss/critic_2', critic_2_loss, updates)
                    logger.add_scalar('loss/policy', policy_loss, updates)
                    logger.add_scalar('loss/entropy_loss', ent_loss, updates)
                    logger.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            replay_memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        if total_steps > config.agent.num_steps:
            break

        logger.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_steps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))

        if i_episode % 10 == 0:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            logger.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close()


if __name__ == "__main__":
    run_agent_and_environment()
