import numpy as np
from casadi import *
import scipy.io as sio
sys.path.append(os.getcwd() + '/src')
import PDP
import JinEnv
from EKF import EKF
from Loss_function import Loss

# --------------------------- Set up OC system ----------------------------------------

dynsys = JinEnv.CartPole()
dynsys.initDyn()
dynsys.initCost(wu = 0.1)

# --------------------------- load demos data ----------------------------------------
data = sio.loadmat('externals/Pontryagin-Differentiable-Programming/Examples/IRL/Cartpole/data/Cartpole_demos.mat')
trajectories = data['trajectories']
true_parameter = data['true_parameter'].flatten()
dt = data['dt']

# --------------------------- Initialize Classes ----------------------------------------
sysoc = PDP.OCSys()
sysoc.setAuxvarVariable(vertcat(dynsys.dyn_auxvar, dynsys.cost_auxvar)) #set which theta to learn
sysoc.setControlVariable(dynsys.U)
sysoc.setStateVariable(dynsys.X)
dyn = dynsys.X + dt * dynsys.f
sysoc.setDyn(dyn)
sysoc.setPathCost(dynsys.path_cost)
sysoc.setFinalCost(dynsys.final_cost)
sysoc.diffPMP()
lqr_solver = PDP.LQR()

# --------------------------- initilize tunable parameter ----------------------------------------
sigma = 0.9
initial_parameter = true_parameter + sigma * np.random.random(len(true_parameter)) - sigma / 2
current_parameter = initial_parameter

loss = 0
dp = np.zeros(current_parameter.shape)
demo_state_traj = trajectories[0, 1]['state_traj_opt'][0, 0]
demo_control_traj = trajectories[0, 1]['control_traj_opt'][0, 0]
demo_ini_state = demo_state_traj[0, :]
demo_horizon = demo_control_traj.shape[0]

# traj_true = sysoc.ocSolver(ini_state=demo_ini_state, horizon=demo_horizon, auxvar_value=true_parameter)

# --------------------------- initilize EKF ----------------------------------------
P_prev = np.eye(current_parameter.shape[0]) * 0.0001
Q_prev = np.eye(current_parameter.shape[0]) * 0.0001
R = 0.0001

# --------------------------- Online Control-Informed Learning ----------------------------------------
for idx in range(demo_horizon):
    
    # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
    traj = sysoc.ocSolver(ini_state=demo_ini_state, horizon=demo_horizon, auxvar_value = current_parameter)

    # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
    aux_sys = sysoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                    control_traj_opt=traj['control_traj_opt'],
                                    costate_traj_opt=traj['costate_traj_opt'],
                                    auxvar_value = current_parameter)
    lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
    lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                            Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
    lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
    aux_sol = lqr_solver.lqrSolver(numpy.zeros((sysoc.n_state, sysoc.n_auxvar)), demo_horizon)

    # take solution of the auxiliary control system
    dxdtheta_traj = aux_sol['state_traj_opt']
    dudtheta_traj = aux_sol['control_traj_opt']

    dxdtheta_t = dxdtheta_traj[idx]
    dudtheta_t = dudtheta_traj[idx]
    dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

    # --------------------------- Loss function, dLdXi ---------------------------------------- 
    state_traj = traj['state_traj_opt']
    control_traj = traj['control_traj_opt']
    # dldx_traj = state_traj - demo_state_traj
    # dldu_traj = control_traj - demo_control_traj

    xi = SX.sym("xi", 5) #n+m
    demo_traj = np.hstack((demo_state_traj[idx], demo_control_traj[idx]))
    current_traj = np.hstack((state_traj[idx], control_traj[idx]))

    loss = norm_2(xi - demo_traj)**2
    dLdXi = jacobian(loss, xi)
    lossFun = Function("lossFun", [xi], [loss])
    dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

    lossNow = lossFun(current_traj).full()
    dLdXiNow = dLdXiFun(current_traj).full()
    
    # --------------------------- Chain rule ----------------------------------------
    # for t in range(demo_horizon):
    #     dp = dp + np.matmul(dldx_traj[t, :], dxdtheta_traj[t]) + np.matmul(dldu_traj[t, :], dudtheta_traj[t])
    # dp = dp + numpy.dot(dldx_traj[-1, :], dxdtheta_traj[-1])
    dLdtheta = np.dot(dLdXiNow, dxidtheta_t)

    # --------------------------- EKF ----------------------------------------
    updateTheta = EKF()
    updateTheta.predict(current_parameter, P_prev, Q_prev)
    updateTheta.update(dLdtheta, R, lossNow)
    print(updateTheta.theta)

    P_prev = updateTheta.P
    current_parameter = updateTheta.theta