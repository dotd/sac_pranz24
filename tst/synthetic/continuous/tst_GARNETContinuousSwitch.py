import numpy as np
import os
import time
import shutil

import datetime
from torch.utils.tensorboard import SummaryWriter

from src.synthetic.GARNETContinuous import GARNETContinuousSwitch
from src.synthetic.transformers_continuous_to_discrete import TransformerContinuousToDiscreteRandom
from src.synthetic.simple_stats_agent import MDPStatsTransition, process_stats
from src.synthetic.detectors import DetectorByCrossing
from src.utils.utils import SmoothSimple
from definitions import ROOT_DIR


def tst_GARNETContinuousSwitch(num_env=2,
                               switch_average_time=10000,
                               states_dim=10,
                               actions_dim=2,
                               num_states=10,
                               num_actions=2,
                               dt=0.01,
                               maximal_num_switches=1000,
                               trajectory_length=1000000,
                               print_freq=10000,
                               check_freq=100,
                               threshold=0.2,
                               threshold_derivative=None,
                               length=2,
                               minimal_between_crossing_times=1000
                               ):
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # folder = f'{ROOT_DIR}/tensorboard/GARNETContinuousSwitch_{date}'
    folder = f'{ROOT_DIR}/tensorboard/GARNETContinuousSwitch_0123/'
    shutil.rmtree(folder)
    time.sleep(1.5)
    print("Finished rmtree")

    writer = SummaryWriter(folder)
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
    lengths = [5000]
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
    # stats_precision_recall:
    #   0th field: True=real transition, False=estimated transition
    #   1st field: time
    smooth = SmoothSimple(0.1)

    for t in range(trajectory_length):
        if t == 66180:
            pass
        # Random policy
        u = 0.01 * rnd.randn(actions_dim, 1)
        x_next, reward, done, info = garnet_continuous_switch.step(u)
        """
        if t < 50000:
            trans_state.add_sample(x_next.squeeze().tolist())
            trans_action.add_sample(u.squeeze().tolist())
        if t == 50000:
            trans_state.do_kmeans()
            trans_action.do_kmeans()
        """
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
                signal = smooth.add_samples(signal)
                crossing = detector_crossing.add_sample(time=t, value=signal)
                if crossing:
                    stats_precision_recall.append((False, t, signal))
                d[f"delta{mdp_state_transition.length}"] = signal
                d[f"crossing{mdp_state_transition.length}"] = crossing

                mdp_stats = mdp_state_transition.get_stats_mdps()
                d[f"mdp_stats{mdp_state_transition.length}_0"] = mdp_stats["stat_perc0"]
                d[f"mdp_stats{mdp_state_transition.length}_1"] = mdp_stats["stat_perc1"]

                d[f"x_next"] = np.linalg.norm(x_next)
            writer.add_scalars("runs", d, t)
    print(stats_precision_recall)
    processed_stats_precision_recall = process_stats(stats_precision_recall)
    print(processed_stats_precision_recall)


if __name__ == "__main__":
    tst_GARNETContinuousSwitch()
