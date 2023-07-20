import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.synthetic.GARNETContinuous import GARNETContinuousSwitch
from src.synthetic.simple_stats_agent import MDPStatsTransition
from definitions import ROOT_DIR


def tst_GARNETContinuousSwitch(num_env=2,
                               switch_average_time=10000,
                               states_dim=3,
                               actions_dim=2,
                               dt=0.01,
                               maximal_num_switches=1000,
                               trajectory_length=500000,
                               print_freq=10000,
                               check_freq=100):
    writer = SummaryWriter(
        f'{ROOT_DIR}/tensorboard/GARNETContinuousSwitch_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    rnd = np.random.RandomState(seed=1)
    garnet_continuous_switch = GARNETContinuousSwitch(num_env,
                                                      switch_average_time,
                                                      maximal_num_switches=maximal_num_switches,
                                                      states_dim=states_dim,
                                                      actions_dim=actions_dim,
                                                      dt=dt,
                                                      rnd=rnd)
    print(f"GARNET Switch MDP:\n{garnet_continuous_switch}\n------")
    x = garnet_continuous_switch.reset()
    for t in range(trajectory_length):
        # Random policy
        u = rnd.randn(actions_dim, 1)
        x_next, reward, done, info = garnet_continuous_switch.step(u)
        mdp_true_previous, mdp_true_next = info["previous"], info["next"]
        # for mdp_state_transition in mdp_state_transitions:
        #     mdp_state_transition.add_sample(action, state, state_next)
        x = x_next
        if t % print_freq == 0:
            print(f"t={t}")
        if t % check_freq == 0:
            d = dict()
            d["true_mdp"] = mdp_true_previous / (num_env - 1)
            writer.add_scalars("runs", d, t)


if __name__ == "__main__":
    tst_GARNETContinuousSwitch()
