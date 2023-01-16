import numpy as np
import scipy.signal
from sortedcontainers import SortedList
from collections import deque


class RunningMedian:

    def __init__(self, window):

        self.window = window
        self._queue = deque(maxlen=window)
        self._sortedlist = SortedList(self._queue)
        self.half = len(self._queue) // 2

    def update(self, curr_sample):

        if len(self._queue) < self.window:

            self._queue.append(curr_sample)
            self._sortedlist.add(curr_sample)
            self.half = len(self._queue) // 2

            return self._sortedlist[self.half]

        self.half = len(self._queue) // 2

        last = self._queue.popleft()
        self._sortedlist.remove(last)

        self._queue.append(curr_sample)
        self._sortedlist.add(curr_sample)

        return self._sortedlist[self.half]

    @property
    def median(self):
        self.half = len(self._queue) // 2
        return self._sortedlist[self.half]

def main():
    np.random.seed(0)
    data = np.random.randn(100)
    width = 5
    res_0 = scipy.signal.medfilt(data, width)
    median_filtering = RunningMedian(window=width)

    medians = []
    for val in data:
        medians.append(median_filtering.update(val))


if __name__ == '__main__':
    main()