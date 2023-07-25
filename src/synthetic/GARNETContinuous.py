import numpy as np


class GARNETContinuous:

    def __init__(self,
                 states_dim,
                 actions_dim,
                 dt,
                 rnd):

        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.dt = dt
        self.random = rnd
        self.A = None  # state x dynamics
        self.B = None  # state u dynamics
        self.create_new_ode()
        self.x = None

    def create_new_ode(self):
        self.A = self.random.randn(self.states_dim, self.states_dim)
        eigenvalues, _ = np.linalg.eig(self.A)
        normalization_factor = np.amax(np.real(eigenvalues))
        self.A = self.A - normalization_factor * np.eye(self.states_dim)
        self.B = self.random.randn(self.states_dim, self.actions_dim)

    def reset(self, state=None):
        if state is None:
            self.x = self.random.randn(self.states_dim, 1)
        else:
            self.x = state
        return self.x

    def step(self, action):
        self.x = self.x + self.dt * (self.A @ self.x + self.B @ action)
        return self.x, 0, False, dict()

    def __str__(self):
        np.set_printoptions(precision=3, suppress=True)
        return f"states_dim={str(self.states_dim)}\nactions_dim={str(self.actions_dim)}\nA=\n{str(self.A)}\nB=\n{str(self.B)}"


class GARNETContinuousSwitch:

    def __init__(self,
                 num_env,
                 switch_average_time,
                 maximal_num_switches,
                 states_dim,
                 actions_dim,
                 dt,
                 rnd):
        self.num_env = num_env
        self.switch_average_time = switch_average_time
        self.maximal_num_switches = maximal_num_switches
        self.switches_counter = 0
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.dt = dt
        self.random = rnd
        self.mdps = list()
        for k in range(self.num_env):
            self.mdps.append(GARNETContinuous(states_dim=states_dim,
                                              actions_dim=actions_dim,
                                              dt=dt,
                                              rnd=rnd))
        self.current_mdp = None
        self.x = None

    def reset(self, mdp_start=None, x=None):
        self.current_mdp = mdp_start if mdp_start is not None else self.random.choice(self.num_env)
        self.x = self.mdps[self.current_mdp].reset(x)
        return self.x

    def step(self, action):
        x_next, reward, done, info = self.mdps[self.current_mdp].step(action)
        info = {"previous": self.current_mdp}
        switch_mdp_flag = self.random.uniform() <= (1 / self.switch_average_time) \
                          and self.switches_counter < self.maximal_num_switches
        if switch_mdp_flag:
            self.current_mdp = self.random.choice(self.num_env)
            self.mdps[self.current_mdp].reset(x_next)
            self.switches_counter += 1
        info["next"] = self.current_mdp
        info["switch"] = info["next"] != info["previous"]
        self.x = self.mdps[self.current_mdp].x
        return self.x, reward, None, info

    def __str__(self):
        s = list()
        for k in range(self.num_env):
            s.append(f"MDP no. {k}\n{self.mdps[k].__str__()}")
        return "\n".join(s)
