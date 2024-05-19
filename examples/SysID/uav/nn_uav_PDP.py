import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
import OCIL
from PDP import PDP
import JinEnv_NN

# ------------------------------ Set up dynamic system ------------------------------
project = "Quadrotor"
mode = "SysID"
saveFlag = False
dynsys = JinEnv_NN.Quadrotor()
dynsys.initDyn(c=0.01)
dynsys.initCost(wr=1, wv=1, wq=5, ww=1, wthrust=0.1)
dt = 0.1

#initialize neural synamic
nnFactor = 5
n_state = dynsys.X.size()[0]
n_control = dynsys.U.size()[0]
dynsys.initNeuralDyn(hidden_layers=[nnFactor*(n_state+n_control), nnFactor*(n_state+n_control)])
n_auxvar = dynsys.n_auxvar

dir = 'examples/SysID/uav/data/'
demoFile = 'uav_iodata.mat'

dynsys.sysid = PDP.SysID()
dynsys.sysid.setAuxvarVariable(dynsys.dyn_auxvar)
dynsys.sysid.setControlVariable(dynsys.U)
dynsys.sysid.setStateVariable(dynsys.X)
dynsys.dyn = dynsys.X + dt * dynsys.f
dynsys.sysid.setDyn(dynsys.dyn)
# ------------------------------ load demos data ------------------------------
data = sio.loadmat(dir+demoFile)
data = data[demoFile[:len(demoFile)-4]][0,0]

n_batch = len(data['batch_inputs'])
batch_inputs = []
batch_states = []
for idx in range(n_batch):
    batch_inputs += [data['batch_inputs'][idx]]
    batch_states += [data['batch_states'][idx]]

# ------------------------------ initialize nn parameter ------------------------------
theta = np.random.randn(n_auxvar)

batch_states_nn = []
for idx in range(n_batch):
    input_traj = batch_inputs[idx]
    ini_state = batch_states[idx][0, :]
    batch_states_nn += [dynsys.sysid.integrateDyn(ini_state=ini_state, inputs=input_traj, auxvar_value=theta)]

batch_states = batch_states_nn

true_theta = theta

# --------------------------- load the data ----------------------------------------
for j in range(1,5):
    # learning rate
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    sigma = 0.9
    initial_parameter = np.array(true_theta) + sigma * np.random.rand(len(true_theta)) - sigma / 2
    current_parameter = initial_parameter
    for k in range(int(2000)):
        # one iteration of PDP
        loss, dp = dynsys.sysid.step(batch_inputs, batch_states, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter]
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss, )

    # save
    save_data = {'trail_no': j,
                 'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'learning_rate': lr}
    sio.savemat('examples/SysID/uav/data/PDP_SysID_nn_results_trial_' + str(j) + '.mat', {'results': save_data})
