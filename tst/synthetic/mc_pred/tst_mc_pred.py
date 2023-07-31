import numpy as np
import datetime
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from src.synthetic.GARNET import GARNETSwitch
from definitions import ROOT_DIR


def tst_record_MC_data(num_env=2,
                       switch_average_time=1000,
                       num_states=10,
                       num_actions=3,
                       branching_factor=3,
                       reward_sparsity=0.5,
                       contrast=1,
                       maximal_num_switches=1000,
                       trajectory_length=100000,
                       print_freq=10000,
                       check_freq=100):
    rnd = np.random.RandomState(seed=1)
    writer = SummaryWriter(f'{ROOT_DIR}/tensorboard/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    garnet_switch = GARNETSwitch(num_env,
                                 switch_average_time,
                                 maximal_num_switches=maximal_num_switches,
                                 num_states=num_states,
                                 num_actions=num_actions,
                                 branching_factor=branching_factor,
                                 reward_sparsity=reward_sparsity,
                                 rnd=rnd,
                                 contrast=contrast)
    print(f"GARNET Switch MDP:\n{garnet_switch}\n------")
    state = garnet_switch.reset()
    trajectory = list()
    for t in range(trajectory_length):
        # Random policy
        action = rnd.choice(garnet_switch.num_actions)
        state_next, reward, done, info = garnet_switch.step(action)
        trajectory.append([state, action, state_next, reward, info["previous"]])
        writer.add_scalar("mdp", info["previous"], t)
        state = state_next
        if t % print_freq == 0:
            print(f"t={t}")
    trajectory_pd = pd.DataFrame(trajectory, columns=["state", "action", "state_next", "reward", "mdp"])
    trajectory_pd.to_csv(f"{ROOT_DIR}/data/mdps.csv")


def create_dataset(data_x, data_y, lookback):
    """Transform a time series into a prediction dataset
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(data_x) - lookback):
        feature = data_x[i:i + lookback]
        target = data_y[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))


class MCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


def tst_classify_trajectories():
    trajectory = pd.read_csv(f"{ROOT_DIR}/data/mdps.csv")
    x = np.expand_dims(trajectory["state"].values, axis=1)
    y = np.expand_dims(trajectory["mdp"].values, axis=1)
    train_size = int(len(x) * 0.67)
    test_size = len(x) - train_size
    print(f"train_size={train_size}, test_size={test_size}")
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    lookback = 100
    x_train_torch, y_train_torch = create_dataset(x_train, y_train, lookback=lookback)
    x_test_torch, y_test_torch = create_dataset(x_test, y_test, lookback=lookback)
    print(f"X_train.shape={x_train_torch.shape}, y_train.shape={y_train_torch.shape}")
    print(f"X_test.shape={x_test_torch.shape}, y_test.shape={y_test_torch.shape}")
    model = MCClassifier()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(x_train_torch, y_train_torch), shuffle=True, batch_size=1024)

    n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == 0:
            print(f"X_batch={X_batch.shape} y_batch={y_batch.shape}")

        # Validation
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(x_train_torch)
                train_rmse = np.sqrt(loss_fn(y_pred, y_train_torch))
                y_pred = model(x_test_torch)
                test_rmse = np.sqrt(loss_fn(y_pred, y_test_torch))
            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


if __name__ == "__main__":
    # tst_record_MC_data()
    tst_classify_trajectories()
