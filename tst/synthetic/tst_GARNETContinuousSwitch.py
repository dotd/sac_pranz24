import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.synthetic.GARNETContinuous import GARNETContinuousSwitch
from src.synthetic.transformers_continuous_to_discrete import TransformerContinuousToDiscreteRandom
from src.synthetic.simple_stats_agent import MDPStatsTransition
from src.synthetic.detectors import DetectorByCrossing
from definitions import ROOT_DIR


def tst_GARNETContinuousSwitch(num_env=2,
                               switch_average_time=10000,
                               states_dim=10,
                               actions_dim=2,
                               num_states=10,
                               num_actions=5,
                               dt=0.01,
                               maximal_num_switches=1000,
                               trajectory_length=100000,
                               print_freq=1000000,
                               check_freq=100,
                               threshold=0.2,
                               threshold_derivative=None,
                               length=2,
                               minimal_between_crossing_times=1000
                               ):
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(
        f'{ROOT_DIR}/tensorboard/GARNETContinuousSwitch_{date}')
    rnd = np.random.RandomState(seed=0)
    garnet_continuous_switch = GARNETContinuousSwitch(num_env,
                                                      switch_average_time,
                                                      maximal_num_switches=maximal_num_switches,
                                                      states_dim=states_dim,
                                                      actions_dim=actions_dim,
                                                      dt=dt,
                                                      rnd=rnd)
    detector_crossing = DetectorByCrossing(threshold=threshold,
                                           threshold_derivative=threshold_derivative,
                                           length=length,
                                           minimal_between_crossing_times=minimal_between_crossing_times
                                           )
    lengths = [1000]
    mdp_state_transitions = list()
    for length in lengths:
        mdp_state_transition = MDPStatsTransition(num_states,
                                                  num_actions,
                                                  length)
        mdp_state_transitions.append(mdp_state_transition)

    print(f"GARNET Switch MDP:\n{garnet_continuous_switch}\n------")
    x = garnet_continuous_switch.reset()
    trans_state = TransformerContinuousToDiscreteRandom(states_dim, num_states, rnd)
    trans_action = TransformerContinuousToDiscreteRandom(actions_dim, num_actions, rnd)
    stats_precision_recall = list()
    for t in range(trajectory_length):
        # Random policy
        u = 0.01 * rnd.randn(actions_dim, 1)
        x_next, reward, done, info = garnet_continuous_switch.step(u)
        if info["switch"]:
            stats_precision_recall.append((True, t))
        state = trans_state.transform(x)
        action = trans_action.transform(u)
        state_next = trans_state.transform(x_next)
        mdp_true_previous, mdp_true_next = info["previous"], info["next"]
        for mdp_state_transition in mdp_state_transitions:
            mdp_state_transition.add_sample(action, state, state_next)
        x = x_next
        if t % print_freq == 0:
            print(f"t={t}")
        if t % check_freq == 0:
            d = dict()
            d["true_mdp"] = mdp_true_previous / (num_env - 1)
            for mdp_state_transition in mdp_state_transitions:
                signal = mdp_state_transition.get_corr_signal()
                crossing = detector_crossing.add_sample(time=t, value=signal)
                if crossing:
                    stats_precision_recall.append((False, t, signal))
                d[f"delta{mdp_state_transition.length}"] = signal
                d[f"crossing{mdp_state_transition.length}"] = crossing
            writer.add_scalars("runs", d, t)
    print(stats_precision_recall)


if __name__ == "__main__":
    tst_GARNETContinuousSwitch()
