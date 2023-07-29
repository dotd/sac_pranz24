import numpy as np
import time
import os
import shutil

import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

from src.synthetic.GARNETContinuous import GARNETContinuousSwitch
from src.synthetic.simple_stats_agent import MDPStatsTransition, process_stats
from definitions import ROOT_DIR


class LSTMPred(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def add_sample(self, sample):
        pass



def tst_GARNETContinuousSwitchLSTM(num_env=2,
                                   switch_average_time=10000,
                                   states_dim=10,
                                   actions_dim=2,
                                   dt=0.01,
                                   maximal_num_switches=1000,
                                   trajectory_length=1000000,
                                   print_freq=10000,
                                   check_freq=100
                                   ):
    folder = f'{ROOT_DIR}/tensorboard/GARNETContinuousSwitch_0123/'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    time.sleep(0.5)
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

    print(f"GARNET Switch MDP:\n{garnet_continuous_switch}\n------")
    x = garnet_continuous_switch.reset()
    stats_precision_recall = list()

    for t in range(trajectory_length):
        if t == 66180:
            pass
        # Random policy
        u = 0.01 * rnd.randn(actions_dim, 1)
        x_next, reward, done, info = garnet_continuous_switch.step(u)
        if info["switch"]:
            stats_precision_recall.append((True, t))
        mdp_true_previous, mdp_true_next = info["previous"], info["next"]
        x = x_next
        if t % print_freq == 0:
            print(f"t={t}")
        if t % check_freq == 0:
            d = dict()
            d["true_mdp"] = mdp_true_previous / (num_env - 1)

            d[f"x_next"] = np.linalg.norm(x_next)
            writer.add_scalars("runs", d, t)
    print(stats_precision_recall)
    processed_stats_precision_recall = process_stats(stats_precision_recall)
    print(processed_stats_precision_recall)


if __name__ == "__main__":
    tst_GARNETContinuousSwitchLSTM()
