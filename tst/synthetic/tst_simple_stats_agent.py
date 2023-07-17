from src.synthetic.simple_stats_agent import SimpleStats


def tst_simple_stats(num_states=3,
                     num_actions=2,
                     length=3):
    simple_stats = SimpleStats(num_states=num_states,
                               num_actions=num_actions,
                               length=length)
    t = 0
    for episode in range(3):
        for state in range(num_states):
            for action in range(num_actions):
                for state_next in range(num_states):
                    sample_to_remove = simple_stats.add_sample(state, action, state_next)
                    print("---------")
                    print(f"episode={episode}, t={t}, sample_to_remove={sample_to_remove is None} ({sample_to_remove}), state={state}, state_next={state_next}, action={action}")
                    print(f"memory =\n {simple_stats.memory}")
                    print(f"stats =\n {simple_stats.stats}")
                    print(f"transition=\n{simple_stats.get_mdp()}")
                    t += 1


if __name__ == "__main__":
    tst_simple_stats()
