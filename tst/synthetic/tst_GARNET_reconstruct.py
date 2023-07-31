import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime

from src.synthetic.GARNET import GARNET
from src.synthetic.simple_stats_agent import MDPStats
from definitions import ROOT_DIR


def tst_simple_stats(
        num_episodes=3,
        num_states=3,
        num_actions=2,
        length=int(3 * 3 * 2 * 1.5)):
    mdp_stats = MDPStats(num_states=num_states,
                         num_actions=num_actions,
                         length=length)
    np.set_printoptions(precision=3, suppress=True)
    t = 0
    for episode in range(num_episodes):
        for action in range(num_actions):
            for state in range(num_states):
                for state_next in range(num_states):
                    sample_to_remove = mdp_stats.add_sample(action, state, state_next)
                    print("---------")
                    print(f"episode={episode}, t={t}, sample_to_remove={sample_to_remove}")
                    print(f"action={action} state={state} state_next={state_next}")
                    print(f"memory len/max_len = {len(mdp_stats.memory)}/{mdp_stats.memory.maxlen}")
                    print(f"memory =\n {mdp_stats.memory}")
                    print(f"stats =\n {mdp_stats.stats}")
                    print(f"transition=\n{mdp_stats.get_mdp()}")
                    t += 1


def tst_GARNET_reconstruct(
        num_states=10,
        num_actions=3,
        branching_factor=4,
        reward_sparsity=0.5,
        contrast=1,
        length=int(10 * 10 * 3 * 1000),
        reconstruct_freq=500,
        seed=0):
    rnd = np.random.RandomState(seed=seed)
    writer = SummaryWriter(f'{ROOT_DIR}/tensorboard/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    mdp = GARNET(num_states=num_states,
                 num_actions=num_actions,
                 branching_factor=branching_factor,
                 reward_sparsity=reward_sparsity,
                 rnd=rnd,
                 contrast=contrast)

    mdp_stats = MDPStats(num_states=num_states,
                         num_actions=num_actions,
                         length=length)
    state = mdp.reset()
    tuples = list()
    for t in range(length):
        action = rnd.choice(num_actions)
        state_next, reward, done, info = mdp.step(action)
        tuples.append([state, action, reward, state_next])
        mdp_stats.add_sample(action=action, state=state, state_next=state_next)
        state = state_next
        if t % reconstruct_freq == 0 or t == length - 1:
            print(f"t={t}")
            mdp_hat = mdp_stats.get_mdp()
            error = np.sum(np.abs(mdp.P - mdp_hat))
            writer.add_scalar("error", error, t)
    print(f"mdp=\n{mdp.P}\nmdp_hat=\n{mdp_hat}\ndel=\n{np.abs(mdp.P - mdp_hat)}")


if __name__ == "__main__":
    # tst_simple_stats()
    tst_GARNET_reconstruct()
