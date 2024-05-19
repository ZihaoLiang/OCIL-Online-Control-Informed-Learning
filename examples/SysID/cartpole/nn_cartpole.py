import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
import OCIL
import JinEnv_NN

# ------------------------------ Set up dynamic system ------------------------------
project = "CartPole"
mode = "SysID"
saveFlag = False
dynsys = JinEnv_NN.CartPole()
dynsys.initDyn()
dynsys.initCost(wx=1,wq=6,wdx=1,wdq=1,wu = 0.1)
dt = 0.05

#initialize neural synamic
nnFactor = 5
n_state = dynsys.X.size()[0]
n_control = dynsys.U.size()[0]
dynsys.initNeuralDyn(hidden_layers=[nnFactor*(n_state+n_control), nnFactor*(n_state+n_control)])


dir = 'examples/SysID/cartpole/data/'
demoFile = 'cartpole_iodata.mat'

system = OCIL.SysID(project, mode, dynsys, dt, dir, demoFile, saveFlag)
system.initialize_nn_parameter()
system.set_sigma(2)
system.set_iteration(10)

# --------------------------- initialize EKF ----------------------------------------
P = np.eye(dynsys.n_auxvar) * 0.0000001
Q = np.eye(dynsys.n_auxvar) * 0.
R = np.eye(4) * 0.0000000001

system.initialize_EKF(P, Q, R)

system.solve()

