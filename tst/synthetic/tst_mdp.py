import numpy as np
from src.synthetic.mdp import GARNET, GARNETSwitch


def tst_GARNET(
        X=5,
        A=3,
        B=3,
        rnd=np.random.RandomState(seed=1),
        reward_sparsity=0.5,
        contrast=1):
    mdp = GARNET(X, A, B, reward_sparsity, rnd=rnd, contrast=contrast)
    print(f"MDP is:\n{mdp}\n------")
    num_trajectories = 2
    trajectory_length = 10
    trajectories = list()
    for e in range(num_trajectories):
        state = mdp.reset()
        trajectory = list()
        for t in range(trajectory_length):
            action = rnd.choice(mdp.num_actions)
            state_next, reward, done, info = mdp.step(action)
            trajectory.append([state, action, reward, state_next])
            state = state_next
        trajectories.append(trajectory)

    print(np.array(trajectories))


def tst_switch(num_env=2,
               switch_average_time=20,
               num_states=5,
               num_actions=3,
               branching_factor=3,
               rnd=np.random.RandomState(seed=1),
               reward_sparsity=0.5,
               contrast=1
               ):
    garnet_switch = GARNETSwitch(num_env,
                                 switch_average_time,
                                 num_states=num_states,
                                 num_actions=num_actions,
                                 branching_factor=branching_factor,
                                 reward_sparsity=reward_sparsity,
                                 rnd=rnd,
                                 contrast=contrast)
    print(f"GARNET Switch MDP:\n{garnet_switch}\n------")
    num_trajectories = 2
    trajectory_length = 100
    trajectories = list()
    for e in range(num_trajectories):
        state = garnet_switch.reset()
        trajectory = list()
        for t in range(trajectory_length):
            # Random policy
            action = rnd.choice(garnet_switch.num_actions)
            state_next, reward, done, info = garnet_switch.step(action)
            trajectory.append([state, action, reward, state_next, info["previous"], info["next"]])
            state = state_next
        trajectories.append(trajectory)

    for i, trajectory in enumerate(trajectories):
        print(f"trajectory {i}\n{np.array(trajectory)}")


if __name__ == "__main__":
    tst_switch()
