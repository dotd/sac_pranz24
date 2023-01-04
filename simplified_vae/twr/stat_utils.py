import numpy as np


def update_shiryaev(self, s, llr, rho=0):
    return np.exp(llr) * (1 + s) / (1 + rho)


def update_cusum(self, s, llr):
    return max(0, s + llr)


def cpd(self, llr_flat, kl_avg):
    s_shiryaev = [0]
    s_cusum = [0]
    step = 0

    for llr in llr_flat - (self.args.annealing_coef / np.array(kl_avg)):
        s_shiryaev.append(self.update_shiryaev(s_shiryaev[-1], llr))
        s_cusum.append(self.update_cusum(s_cusum[-1], llr))
        # wandb.log({'shiryaev': s_shiryaev[-1], 'cusum': s_cusum[-1]}, step=step)
        step += 1

    return s_shiryaev, s_cusum