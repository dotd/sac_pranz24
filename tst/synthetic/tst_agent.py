import numpy as np
from src.synthetic.GARNET import GARNETSwitch
from src.synthetic.simple_stats_agent import MDPStatsTransition


def tst_agent(num_env=2,
              switch_average_time=100,
              maximal_num_switches=1,
              num_states=5,
              num_actions=3,
              branching_factor=3,
              rnd=np.random.RandomState(seed=1),
              reward_sparsity=0.5,
              contrast=1):
    garnet_switch = GARNETSwitch(num_env=num_env,
                                 switch_average_time=switch_average_time,
                                 maximal_num_switches=maximal_num_switches,
                                 num_states=num_states,
                                 num_actions=num_actions,
                                 branching_factor=branching_factor,
                                 reward_sparsity=reward_sparsity,
                                 rnd=rnd,
                                 contrast=contrast)

    print(f"GARNET Switch MDP:\n{garnet_switch}\n------")
    num_trajectories = 2
    trajectory_length = 1000
    cusum_length = 100
    trajectories = list()
    for e in range(num_trajectories):
        state = garnet_switch.reset()
        trajectory = list()

        simple_stats_transition = MDPStatsTransition(
            num_states=num_states,
            num_actions=num_actions,
            length=cusum_length)

        for t in range(trajectory_length):
            print(f"t={t}")
            # Random policy
            action = rnd.choice(garnet_switch.num_actions)
            state_next, reward, done, info = garnet_switch.step(action)
            sample = [state, action, reward, state_next, info["previous"], info["next"]]
            trajectory.append(sample)
            state = state_next
            simple_stats_transition.add_sample(sample)
            if t > cusum_length:
                loglikelihood_vec = simple_stats_transition.get_log_likelihood()
                print(loglikelihood_vec)

        # print(f"trajectory {e}\n{np.array(trajectory)}")


if __name__ == "__main__":
    tst_agent()
