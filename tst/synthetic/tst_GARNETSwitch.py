import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.synthetic.GARNET import GARNETSwitch
from src.synthetic.simple_stats_agent import MDPStatsTransition
from definitions import ROOT_DIR


def tst_show_GARNETSwitch(num_env=2,
                          switch_average_time=20,
                          num_states=3,
                          num_actions=2,
                          branching_factor=3,
                          reward_sparsity=0.5,
                          contrast=1,
                          maximal_num_switches=10,
                          num_trajectories=2,
                          trajectory_length=100):
    # Creating the GARNETSwitch
    rnd = np.random.RandomState(seed=1)

    garnet_switch = GARNETSwitch(num_env,
                                 switch_average_time,
                                 maximal_num_switches=maximal_num_switches,
                                 num_states=num_states,
                                 num_actions=num_actions,
                                 branching_factor=branching_factor,
                                 reward_sparsity=reward_sparsity,
                                 rnd=rnd,
                                 contrast=contrast)
    print(f"GARNET Switch MDP:\n{garnet_switch}\n------")
    trajectories = list()
    for e in range(num_trajectories):
        state = garnet_switch.reset()
        trajectory = list()
        for t in range(trajectory_length):
            # Random policy
            action = rnd.choice(garnet_switch.num_actions)
            state_next, reward, done, info = garnet_switch.step(action)
            trajectory.append([t, state, action, reward, state_next, info["previous"], info["next"]])
            state = state_next
        trajectories.append(trajectory)

    for i, trajectory in enumerate(trajectories):
        print(f"trajectory {i}\n{np.array(trajectory)}")


def tst_GARNETSwitch(num_env=10,
                     switch_average_time=10000,
                     num_states=3,
                     num_actions=2,
                     branching_factor=3,
                     reward_sparsity=0.5,
                     contrast=1,
                     maximal_num_switches=1000,
                     trajectory_length=500000,
                     print_freq=10000,
                     check_freq=100):
    writer = SummaryWriter(f'{ROOT_DIR}/tensorboard/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    rnd = np.random.RandomState(seed=1)
    garnet_switch = GARNETSwitch(num_env,
                                 switch_average_time,
                                 maximal_num_switches=maximal_num_switches,
                                 num_states=num_states,
                                 num_actions=num_actions,
                                 branching_factor=branching_factor,
                                 reward_sparsity=reward_sparsity,
                                 rnd=rnd,
                                 contrast=contrast)
    lengths = [100, 500, 1000, 5000, 10000]
    mdp_state_transitions = list()
    for length in lengths:
        mdp_state_transition = MDPStatsTransition(num_states,
                                                  num_actions,
                                                  length)
        mdp_state_transitions.append(mdp_state_transition)
    print(f"GARNET Switch MDP:\n{garnet_switch}\n------")
    state = garnet_switch.reset()
    for t in range(trajectory_length):
        # Random policy
        action = rnd.choice(garnet_switch.num_actions)
        state_next, reward, done, info = garnet_switch.step(action)
        mdp_true_previous, mdp_true_next = info["previous"], info["next"]
        for mdp_state_transition in mdp_state_transitions:
            mdp_state_transition.add_sample(action, state, state_next)
        state = state_next
        if t % print_freq == 0:
            print(f"t={t}")
        if t % check_freq == 0:
            d = dict()
            d["true_mdp"] = mdp_true_previous / (num_env - 1)
            for mdp_state_transition in mdp_state_transitions:
                signal = mdp_state_transition.get_corr_signal()
                d[f"delta{mdp_state_transition.length}"] = signal
            writer.add_scalars("runs", d, t)


if __name__ == "__main__":
    # tst_show_GARNETSwitch()
    tst_GARNETSwitch()
