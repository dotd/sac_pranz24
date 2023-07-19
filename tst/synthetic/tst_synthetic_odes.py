import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from definitions import ROOT_DIR
from src.synthetic.GARNETContinuous import GARNETContinuous


def tst1():
    d = 10
    rnd = np.random.RandomState(0)
    mat = rnd.randn(d, d)
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    print(eigenvalues)
    print(np.abs(eigenvalues))
    normalization_factor = np.amax(np.abs(eigenvalues))
    mat = mat / (normalization_factor * 1.5)
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    print(eigenvalues)
    print(np.abs(eigenvalues))

    writer = SummaryWriter(f'{ROOT_DIR}/tensorboard/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    x = rnd.randn(d, 1)
    dt = 0.01
    for t in range(10000):
        x = x + dt * mat @ x
        x_dict = dict((str(x), y) for x, y in enumerate(x.flatten(), 1))
        writer.add_scalars("dyn_system", x_dict, t)


def tst2():
    # np.set_printoptions(precision=3, suppress=True)
    d = 10
    rnd = np.random.RandomState(0)
    mat = rnd.randn(d, d)
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    print(eigenvalues)
    print(np.real(eigenvalues))
    normalization_factor = np.amax(np.real(eigenvalues))
    mat = mat - normalization_factor * np.eye(d)
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    print(mat)
    print(eigenvalues)
    print(np.real(eigenvalues))

    writer = SummaryWriter(f'{ROOT_DIR}/tensorboard/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    x = rnd.randn(d, 1)
    dt = 0.01
    for t in range(10000):
        u = rnd.randn(d, 1)
        x = x + dt * (mat @ x + u)
        x_dict = dict((str(x), y) for x, y in enumerate(x.flatten(), 1))
        writer.add_scalars("dyn_system", x_dict, t)
    writer.flush()



if __name__ == "__main__":
    # tst1()
    # tst2()
    print()
