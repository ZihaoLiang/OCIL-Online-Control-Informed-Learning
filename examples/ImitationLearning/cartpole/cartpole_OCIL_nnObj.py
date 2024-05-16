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
project = "CartPole"
mode = "Objective"
saveFlag = False
dynsys = JinEnv.CartPole()
dynsys.initDyn(mc=0.5, mp=0.5, l=1)
# dynsys.initCost(wu = 0.1)

#initialize neural cost
nnFactor = 5
n_state = dynsys.X.size()[0]
dynsys.initNeuralCost(hidden_layers=[nnFactor*n_state, nnFactor*n_state],\
                       wu=0.0001)  #neural objective

dir = 'examples/ImitationLearning/cartpole/data/'
demoFile = 'cartpole_demos.mat'

system = OCIL.ImitationLearning(project, mode, dynsys, dir, demoFile, saveFlag)

# --------------------------- initialize EKF ----------------------------------------
# P = np.eye(4) * 0.0000001
# Q = np.eye(4) * 0.
# R = np.eye(5) * 0.0000000001

# for neural objective
P = np.eye(dynsys.n_auxvar) * 0.0001
Q = np.eye(dynsys.n_auxvar) * 0.
R = np.eye(5) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()

