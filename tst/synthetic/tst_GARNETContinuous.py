import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from definitions import ROOT_DIR
from src.synthetic.GARNETContinuous import GARNETContinuous


def tst_GARNETContinous():
    state_dim = 10
    actions_dim = 5
    rnd = np.random.RandomState(0)
    writer = SummaryWriter(f'{ROOT_DIR}/tensorboard/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    env = GARNETContinuous(states_dim=state_dim, actions_dim=actions_dim, dt=0.01, rnd=rnd)
    x = env.reset()
    for t in range(10000):
        # Choosing a random action (or control)
        u = rnd.randn(actions_dim, 1)
        x_next, r, done, info = env.step(u)
        x_dict = dict((str(x), y) for x, y in enumerate(x_next.flatten(), 1))
        writer.add_scalars("dyn_system", x_dict, t)
    writer.flush()


if __name__ == "__main__":
    tst_GARNETContinous()
