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
project = "CartPole"
mode = "Imitation Learning"
saveFlag = False
dynsys = JinEnv.CartPole()
dynsys.initDyn(mc=0.5, mp=0.5, l=1)
dynsys.initCost(wu = 0.1)

dir = 'examples/ImitationLearning/cartpole/data/'
demoFile = 'cartpole_demos.mat'

system = OCIL(project, mode, dynsys, dir, demoFile, saveFlag)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(4) * 0.0000001
Q = np.eye(4) * 0.
R = np.eye(5) * 0.0000000001

system.initialize_EKF(P, Q, R)

system.solveAllLoss()

# dataFile = 'examples/ImitationLearning/cartpole/data/results/iter_30.mat'
# system.load(dataFile)