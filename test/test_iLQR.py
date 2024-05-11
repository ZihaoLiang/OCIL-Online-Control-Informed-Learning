import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming/ControlTool')
import OCIL
from JinEnv import JinEnv
from ControlTool import ControlTools


if __name__ == '__main__':
# ------------------------------ Set up dynamic system ------------------------------
    project = "Rocket"
    saveFlag = False
    rocket = JinEnv.Rocket()
    rocket.initDyn(Jx=0.5, Jy=1, Jz=1, mass=1, l=1)
    rocket.initCost(wr=1, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust=0.1)

    dir = 'examples/PolicyTuning/rocket/data/'
    demoFile = 'rocket_iLQR.mat'

# ------------------------------ Create iLQR object ------------------------------
    # initialize iLQR solver
    Rocket_iLQR = ControlTools.iLQR(project_name=project)
    Rocket_iLQR.setStateVariable(rocket.X)
    Rocket_iLQR.setControlVariable(rocket.U)
    # initialize dynamics
    dt = 0.1
    dyn = rocket.X + dt * rocket.f
    Rocket_iLQR.setDyn(dyn)
    # initialize cost
    Rocket_iLQR.setPathCost(rocket.path_cost)
    Rocket_iLQR.setFinalCost(rocket.final_cost)

    # initialize LQR solver

# ------------------------------ iLQR iteration ------------------------------
    # maximum iteration
    iterMax = 5

    for idx in range(iterMax):

        # NOTE: not sure whether need to initialize a new LQR solver every iteration
        lqr_solver = ControlTools.LQR()


        lossNow, ctrlTrajNow = Rocket_iLQR.step(ini_state, control_traj, lqr_solver)
