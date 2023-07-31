from collections import deque
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, sizes):
        super(Net, self).__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = torch.nn.Linear(sizes[0], sizes[1])
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = torch.nn.Linear(sizes[1], sizes[2])

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.sigmoid(x)
        return output


class NNDetector:

    def __init__(self,
                 state_dim,
                 action_dim,
                 buffer_size,
                 window_size,
                 batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.sasn = (self.state_dim + self.action_dim)

    def add_samples(self, state, action, state_next):
        self.buffer.append([state, action])

    def train(self):
        # build batch
        pass
