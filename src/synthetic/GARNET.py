import numpy as np


class GARNET:

    def __init__(self,
                 num_states,
                 num_actions,
                 branching_factor,
                 reward_sparsity,
                 rnd,
                 contrast):
        """
        :param num_states:
        :param num_actions:
        :param branching_factor: how many zeros are not zero for each state-action pair.
        :param reward_sparsity: between 0 to 1, expresses how many rewards are not 0
        :param rnd:
        :param contrast: positive number. high contrast makes the rewards close to 0 or 1, while low contrast make
                it all rewards equal
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.branching_factor = branching_factor
        self.reward_sparsity = reward_sparsity
        self.random = rnd
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))
        self.contrast = contrast
        self.create_new_mdp()
        self.state = None

    def create_new_mdp(self):
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states))
        self.R = self.random.randn(self.num_states, self.num_actions)
        self.R *= self.random.choice(a=2, size=(self.num_states, self.num_actions),
                                     p=[1 - self.reward_sparsity, self.reward_sparsity])
        for a in range(self.num_actions):
            for s in range(self.num_states):
                prob = self.random.uniform(size=(self.branching_factor,))
                prob = prob / np.sum(prob)
                prob = np.power(prob, self.contrast)
                prob = prob / np.sum(prob)
                indices = self.random.choice(self.num_states, size=(self.branching_factor,), replace=False)
                self.P[a, s, indices] = prob

    def reset(self, state=None):
        if state is None:
            self.state = self.random.choice(self.num_states)
        else:
            self.state = state
        return self.state

    def step(self, action):
        state_next = self.random.choice(self.num_states, p=self.P[action, self.state])
        reward = self.R[self.state, action]
        self.state = state_next
        return self.state, reward, False, dict()

    def __str__(self):
        np.set_printoptions(precision=3, suppress=True)
        return f"num_states={str(self.num_states)}\nnum_actions={str(self.num_actions)}\nP=\n{str(self.P)}\nR=\n{str(self.R)}"


class GARNETSwitch:

    def __init__(self,
                 num_env,
                 switch_average_time,
                 maximal_num_switches,
                 num_states,
                 num_actions,
                 branching_factor,
                 reward_sparsity,
                 rnd,
                 contrast):
        self.num_env = num_env
        self.switch_average_time = switch_average_time
        self.maximal_num_switches = maximal_num_switches
        self.switches_counter = 0
        self.num_states = num_states
        self.num_actions = num_actions
        self.random = rnd
        self.mdps = list()
        for k in range(self.num_env):
            self.mdps.append(GARNET(num_states,
                                    num_actions,
                                    branching_factor,
                                    reward_sparsity,
                                    rnd=rnd,
                                    contrast=contrast))
        self.current_mdp = None
        self.state = None

    def reset(self, mdp_start=None, state=None):
        self.current_mdp = mdp_start if mdp_start is not None else self.random.choice(self.num_env)
        self.state = self.mdps[self.current_mdp].reset(state)
        return self.state

    def step(self, action):
        state_next, reward, done, info = self.mdps[self.current_mdp].step(action)
        info = {"previous": self.current_mdp}
        switch_mdp_flag = self.random.uniform() <= 1 / self.switch_average_time \
                          and self.switches_counter < self.maximal_num_switches
        if switch_mdp_flag:
            self.current_mdp = self.random.choice(self.num_env)
            self.mdps[self.current_mdp].reset(state_next)
            self.switches_counter += 1
        info["next"] = self.current_mdp
        self.state = self.mdps[self.current_mdp].state
        return self.state, reward, None, info

    def __str__(self):
        s = list()
        for k in range(self.num_env):
            s.append(f"MDP no. {k}\n{self.mdps[k].__str__()}")
        return "\n".join(s)
