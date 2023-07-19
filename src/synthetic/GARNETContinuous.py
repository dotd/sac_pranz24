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
        return f"states_dim={str(self.states_dim)}\nactions_dim={str(self.actions_dim)}\nA=\n{str(self.P)}\nB=\n{str(self.B)}"
