from collections import deque


class DetectorByCrossing:

    def __init__(self,
                 threshold,
                 threshold_derivative,
                 length,
                 minimal_between_crossing_times):
        self.threshold = threshold
        self.threshold_derivative = threshold_derivative
        self.signal = deque(maxlen=length)
        self.derivative = deque(maxlen=length)
        self.crossing_times = list()
        self.minimal_between_crossing_times = minimal_between_crossing_times
        self.last_time_crossing = None

    def add_sample(self, time, value):
        # crossing_flag = False
        self.signal.append((time, value))
        # if len(self.signal) >= 2:
        #     self.derivative.append(self.signal[-1][1] - self.signal[-2][1])
        return self.check_crossings()

    def check_crossings(self):
        if len(self.signal) < 2:
            return False
        if self.signal[-1][1] > self.threshold:
            if self.signal[-2][1] < self.threshold:
                if self.last_time_crossing is None:
                    self.last_time_crossing = self.signal[-1][0]
                    return True
                if self.signal[-1][0] - self.last_time_crossing >= self.minimal_between_crossing_times:
                    self.last_time_crossing = self.signal[-1][0]
                    return True
        return False
