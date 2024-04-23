import numpy as np


class Loss:
    def _init_(self):
        pass

    def getLoss(traj, traj_oc):
        return np.linalg.norm(traj-traj_oc, ord=2)**2





