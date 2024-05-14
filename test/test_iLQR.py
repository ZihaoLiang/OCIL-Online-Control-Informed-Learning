import numpy as np
from casadi import *
import scipy.io as sio
import copy
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming/ControlTool')
import OCIL
from JinEnv import JinEnv
from PDP import PDP
from ControlTool import ControlTools


if __name__ == '__main__':
# ------------------------------ Set up dynamic system ------------------------------
    project = "Rocket"
    saveFlag = False
    rocket = JinEnv.Rocket()
    rocket.initDyn(Jx=0.5, Jy=1, Jz=1, mass=1, l=1)
    rocket.initCost(wr=1, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust=0.1)
    # dimension
    inputDim = rocket.U.shape[0]
    print("control dim: ", inputDim)

    dir = 'examples/PolicyTuning/rocket/data/'
    demoFile = 'rocket_iLQR.mat'

    # initialize dynamics
    dt = 0.1
    horizon = 50
    dyn = rocket.X + dt * rocket.f
    # initial state
    ini_r_I = [10, -8, 5.]
    ini_v_I = [-.1, 0.0, -0.0]
    ini_q = JinEnv.toQuaternion(1.5, [0, 0, 1])
    ini_w = [0, -0.0, 0.0]
    ini_state = ini_r_I + ini_v_I + ini_q + ini_w

# ------------------------------ Create true OC object ------------------------------
    RocketOCTrue = PDP.OCSys()
    RocketOCTrue.setStateVariable(rocket.X)
    RocketOCTrue.setControlVariable(rocket.U)
    RocketOCTrue.setDyn(dyn)
    RocketOCTrue.setPathCost(rocket.path_cost)
    RocketOCTrue.setFinalCost(rocket.final_cost)
    solTrue = RocketOCTrue.ocSolver(ini_state=ini_state, horizon=horizon)
    print("True OC cost: ", solTrue['cost'].flatten()[0])
    # print("True OC control: ", solTrue['control_traj_opt'])
    print("control traj shape: ", solTrue['control_traj_opt'].shape)

# ------------------------------ Create iLQR object ------------------------------
    # initialize iLQR solver
    Rocket_iLQR = ControlTools.iLQR(project_name=project)
    Rocket_iLQR.setStateVariable(rocket.X)
    Rocket_iLQR.setControlVariable(rocket.U)
    Rocket_iLQR.setDyn(dyn)
    # initialize cost
    Rocket_iLQR.setPathCost(rocket.path_cost)
    Rocket_iLQR.setFinalCost(rocket.final_cost)

# ------------------------------ iLQR iteration ------------------------------
    # maximum iteration
    iterMax = 10

    # initialize the control trajectory
    # ctrlTrajNow = np.zeros((horizon, inputDim))
    ctrlTrajNow = copy.deepcopy(solTrue['control_traj_opt'])

    for idx in range(iterMax):
        print("iter: ", idx)

        # NOTE: not sure whether need to initialize a new LQR solver every iteration
        # initialize a new solver every iteration for safety
        lqr_solver = ControlTools.LQR()

        # iLQR iterate
        lossNow, ctrlTrajNow = Rocket_iLQR.step(ini_state, ctrlTrajNow, lqr_solver)
        # change the data structure as horizon times inputDim
        ctrlTrajNow = np.array(ctrlTrajNow).transpose(2,0,1).reshape(horizon,-1)

        print("loss: ", lossNow)

    # final solution
    solFinal = Rocket_iLQR.integrateSys(ini_state, ctrlTrajNow)
    stateTrajFinal = solFinal["state_traj"]
    ctrlTrajFinal = solFinal["control_traj"]
    costFinal = solFinal["cost"]
