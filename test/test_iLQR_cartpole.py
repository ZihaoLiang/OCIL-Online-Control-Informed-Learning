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
import matplotlib.pyplot as plt


if __name__ == '__main__':
# ------------------------------ Set up dynamic system ------------------------------
    project = "CartPole"
    saveFlag = False
    cartpole = JinEnv.CartPole()
    mc, mp, l = 0.1, 0.1, 1
    wx, wq, wdx, wdq, wu = 0.1, 0.6, 0.1, 0.1, 0.3
    cartpole.initDyn(mc=mc, mp=mp, l=l)
    cartpole.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu=wu)
    # dimension
    inputDim = cartpole.U.shape[0]
    print("control dim: ", inputDim)

    dir = 'examples/PolicyTuning/cartpole/data/'
    demoFile = 'cartpole_iLQR.mat'

    # initialize dynamics
    dt = 0.05
    horizon = 25
    dyn = cartpole.X + dt * cartpole.f
    # initial state
    ini_state = [0, 0, 0, 0]

# ------------------------------ Create true OC object ------------------------------
    CartPoleOCTrue = PDP.OCSys()
    CartPoleOCTrue.setStateVariable(cartpole.X)
    CartPoleOCTrue.setControlVariable(cartpole.U)
    CartPoleOCTrue.setDyn(dyn)
    CartPoleOCTrue.setPathCost(cartpole.path_cost)
    CartPoleOCTrue.setFinalCost(cartpole.final_cost)
    solTrue = CartPoleOCTrue.ocSolver(ini_state=ini_state, horizon=horizon)
    costTrue = solTrue['cost'].flatten()[0]
    print("True OC cost: ", costTrue)
    # print("True OC control: ", solTrue['control_traj_opt'])
    print("control traj shape: ", solTrue['control_traj_opt'].shape)

# ------------------------------ Create iLQR object ------------------------------
    # initialize iLQR solver
    CartPole_iLQR = ControlTools.iLQR(project_name=project)
    CartPole_iLQR.setStateVariable(cartpole.X)
    CartPole_iLQR.setControlVariable(cartpole.U)
    CartPole_iLQR.setDyn(dyn)
    # initialize cost
    CartPole_iLQR.setPathCost(cartpole.path_cost)
    CartPole_iLQR.setFinalCost(cartpole.final_cost)

# ------------------------------ iLQR iteration ------------------------------
    # maximum iteration
    iterMax = 300

    lossTraj = list()

    # initialize the control trajectory
    ctrlTrajNow = np.zeros((horizon, inputDim))
    # ctrlTrajNow = copy.deepcopy(solTrue['control_traj_opt'])

    for idx in range(iterMax):
        # NOTE: not sure whether need to initialize a new LQR solver every iteration
        # initialize a new solver every iteration for safety
        lqr_solver = ControlTools.LQR()

        # iLQR iterate
        lossNow, ctrlTrajNow = CartPole_iLQR.step(ini_state, ctrlTrajNow, lqr_solver)


        # # change the data structure as horizon times inputDim
        # ctrlTrajNow = np.array(ctrlTrajNow).transpose(2,0,1).reshape(horizon,-1)

        # change the data structure as horizon times inputDim
        _ctrlTrajNow = np.zeros((horizon, inputDim))
        for i in range(len(ctrlTrajNow)):
            _ctrlTrajNow[i, :] = ctrlTrajNow[i].flatten()
        ctrlTrajNow = copy.deepcopy(_ctrlTrajNow)


        lossTraj.append(lossNow)
        if idx % 25 == 0:
            print("iter: ", idx)
            print("loss: ", lossNow)

    # final solution
    solFinal = CartPole_iLQR.integrateSys(ini_state, ctrlTrajNow)
    stateTrajFinal = solFinal["state_traj"]
    ctrlTrajFinal = solFinal["control_traj"]
    costFinal = solFinal["cost"]

    print("control dim: ", inputDim)
    print("control traj shape: ", solTrue['control_traj_opt'].shape)
    print("True OC cost: ", costTrue)

    # visualization
    fig = plt.figure(1)
    ax = fig.subplots(1)
    ax.plot(list(range(iterMax)), lossTraj, "b")
    ax.plot([0, iterMax-1], [costTrue, costTrue], "--r")
    ax.legend(["iLQR", "Optimal Control"])
    ax.set_ylabel("Loss (Cost)")
    ax.set_xlabel('Iteration')
    ax.grid()
    plt.show()
