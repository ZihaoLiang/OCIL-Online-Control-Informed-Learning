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
mode = "Policy Tuning"
case = "Objective"
saveFlag = False
dynsys = JinEnv.CartPole()
dynsys.initDyn(mc=0.1, mp=0.1, l=1)
dynsys.initCost(wx=0.1, wq=0.6, wdx=0.1, wdq=0.1, wu=0.3)

dir = 'examples/PolicyTuning/cartpole/data/'
demoFile = 'cartpole_demos.mat'
run = 1

# --------------------------- initilize ----------------------------------------
P = np.eye(541) * 0.00000000001
Q = np.eye(541) * 0.
R = np.eye(5) * 0.0000000001

nnFactor = 5
dt = 0.05
horizon = 25
ini_state = [0, 0, 0, 0]

system = OCIL.PolicyTuning(project, mode, case, dynsys, nnFactor, dir, demoFile, saveFlag)
# system.set_iteration(200)

if run == 1:
    ## single run ##
    system.initialize_EKF(P, Q, R)
    system.generate_traj(dt, horizon, ini_state)
    system.solve()
else: 
    ## multiple runs ##
    for idx in range(run):
        system.initialize_EKF(P, Q, R)
        system.generate_traj(dt, horizon, ini_state)
        Loss = system.solve()
        print('Test = ', idx, 'Loss = ', Loss[-1])
        sio.savemat(dir+"online/results_"+str(idx)+".mat", {'Loss': Loss})
