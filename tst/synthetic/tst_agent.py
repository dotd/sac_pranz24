import numpy as np
from src.synthetic.mdp import GARNETSwitch


def tst_agent(num_env=2,
              switch_average_time=200,
              num_states=5,
              num_actions=3,
              branching_factor=3,
              rnd=np.random.RandomState(seed=1),
              reward_sparsity=0.5,
              contrast=1):
    garnet_switch = GARNETSwitch(num_env,
                                 switch_average_time,
                                 num_states=num_states,
                                 num_actions=num_actions,
                                 branching_factor=branching_factor,
                                 reward_sparsity=reward_sparsity,
                                 rnd=rnd,
                                 contrast=contrast)


if __name__ == "__main__":
    tst_agent()
