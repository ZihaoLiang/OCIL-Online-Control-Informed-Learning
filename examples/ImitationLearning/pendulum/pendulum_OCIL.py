import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from OCIL import OCIL
from JinEnv import JinEnv



# ------------------------------ Set up dynamic system ------------------------------
project = "Pendulum"
mode = "Imitation Learning"
saveFlag = False
dynsys = JinEnv.SinglePendulum()

dynsys.initDyn(l=1, m=1, damping_ratio=0.1)
dynsys.initCost()

dir = 'examples/ImitationLearning/pendulum/data/'
demoFile = 'pendulum_demos.mat'


system = OCIL(project, mode, dynsys, dir, demoFile, saveFlag)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(2) * 0.0000001
Q = np.eye(2) * 0.
R = np.eye(3) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()
