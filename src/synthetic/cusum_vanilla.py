class CUSUMVanilla:

    def __init__(self):
        self.k = None
        self.s = None
        self.S = None
        self.S_min = None
        self.G = None

    def reset(self):
        self.k = 0
        self.s = list()
        self.S = list()
        self.S_min = list()
        self.G = list()

    def add_sample(self, log_likelihhod_ratio):
        self.s.append(log_likelihhod_ratio)
        if self.k == 0:
            self.S.append(log_likelihhod_ratio)
            self.S_min.append(log_likelihhod_ratio)
            self.G.append(0)
        else:
            self.S.append(self.S[-1] + log_likelihhod_ratio)
            self.S_min.append(self.S[-1] if self.S[-1] < self.S_min[-1] else self.S_min[-1])
            self.G.append(self.S[-1] - self.S_min[-2])
        self.k += 1

    def __str__(self):
        s = list()
        s.append(f"k={self.k}")
        s.append(f"s={self.s}")
        s.append(f"S={self.S}")
        s.append(f"S_min={self.S_min}")
        s.append(f"G={self.G}")
        return "\n".join(s)

