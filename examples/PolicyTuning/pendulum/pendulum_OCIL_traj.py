import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
import OCIL
from JinEnv import JinEnv


# ------------------------------ Set up dynamic system ------------------------------
project = "Pendulum"
mode = "Policy Tuning"
case = "traj"
saveFlag = False
dynsys = JinEnv.SinglePendulum()
dynsys.initDyn(l=1, m=1, damping_ratio=0.1)
dynsys.initCost(wq=10, wdq=1)

dir = 'examples/PolicyTuning/pendulum/data/'
demoFile = 'pendulum_demos.mat'

# --------------------------- initilize ----------------------------------------
P = np.eye(151) * 0.000000001
Q = np.eye(151) * 0.
R = np.eye(3) * 0.0000000001

nnFactor = 5

system = OCIL.PolicyTuning(project, mode, case, dynsys, nnFactor, dir, demoFile, saveFlag)
system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()
