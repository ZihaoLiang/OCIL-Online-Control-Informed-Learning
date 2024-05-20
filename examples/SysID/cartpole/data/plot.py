import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes

params = {'axes.labelsize': 50,
          'axes.titlesize': 30,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize':24}
plt.rcParams.update(params)

fig = plt.figure(figsize=(11, 9))

h = [Size.Fixed(1.8), Size.Fixed(8.5)]
v = [Size.Fixed(1.2), Size.Fixed(7.5)]

divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
# the width and height of the rectangle is ignored.

ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

fig.add_axes(ax)

ax.set_xscale('symlog')
ax.set_yscale('log')
ax.set_xlim(0,10000)
ax.set_ylim(bottom=1e-8,top=1e4)
ax.set_xlabel('Number of Data Points')
ax.set_ylabel('SysID Loss')
ax.tick_params(axis='both', which='major')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([1,1,10,6])


# load OCIL results
OCIL_loss_list = []
for i in range(5):
    load = sio.loadmat('OCIL_results_' + str(i))
    loss_trace = load['Loss'][0]
    OCIL_loss_list += [loss_trace]
    horizon = int(load['horizon'])

OCIL_iter = list(range(0, len(OCIL_loss_list[0])))

# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('PDP_SysID_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace]

pdp_iter = list(range(0, len(pdp_loss_list[0])))
pdp_iter = [x*horizon for x in pdp_iter]

# load nn results
nn_loss_list = []
for i in range(5):
    load = sio.loadmat('NN_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    nn_loss_list += [loss_trace]

nn_iter = list(range(0, len(nn_loss_list[0])))
nn_iter = [x*horizon for x in nn_iter]

# plot results
for pdp_loss, nn_loss in zip(pdp_loss_list, nn_loss_list):
    ax.plot(pdp_iter, pdp_loss, color = [0.6350, 0.0780, 0.1840], linewidth=4)
    ax.plot(nn_iter, nn_loss[0:], color=[0.4660, 0.6740, 0.1880], linewidth=4)

# show legend
line_pdp,=ax.plot(pdp_iter, pdp_loss_list[0], color = [0.6350, 0.0780, 0.1840], linewidth=4)
line_nn,=ax.plot(nn_iter, nn_loss_list[0][0:], color=[0.4660, 0.6740, 0.1880], linewidth=4)

for OCIL_loss in OCIL_loss_list:
    ax.plot(OCIL_loss[:horizon], color='b', linewidth=4)
    ax.plot(OCIL_iter[horizon:], OCIL_loss[horizon:], color='b', linestyle='--', linewidth=4)

line_OCIL, = ax.plot(OCIL_loss_list[0][:horizon], color='b', linewidth=4)
line_OCIL_dashed, = ax.plot(OCIL_iter[horizon:], OCIL_loss_list[0][horizon:], color='b', linestyle='--', linewidth=4)

ax.axvline(horizon, color='r', linewidth=4)
ax.legend([line_OCIL,line_OCIL_dashed,line_pdp,line_nn],['OCIL (Proposed), Online','OCIL (Proposed), Offline','PDP','NN Dynamics'],
            facecolor='white',framealpha=0.5,loc=3, fontsize=24)

# load nn OCIL results
OCILnn_loss_list = []
for i in range(5):
    load = sio.loadmat('nn_dynamic_results_' + str(i))
    loss_trace = load['Loss'][0]
    OCILnn_loss_list += [loss_trace]
    horizon = int(load['horizon'])


# load nn pdp results
pdpnn_loss_list = []
for i in range(5):
    load = sio.loadmat('PDP_SysID_nn_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdpnn_loss_list += [loss_trace]

pdpnn_iter = list(range(0, len(pdpnn_loss_list[0])))
pdpnn_iter = [x*horizon for x in pdpnn_iter]

# load dmd results
dmd_loss_list = []
for i in range(5):
    load = sio.loadmat('DMD_results_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    index = np.argwhere(loss_trace > 1000)
    loss_trace[index] = loss_trace[
        index - 1]  # remove the spikes inside the data (only for kkt results, because it is too bad)
    dmd_loss_list += [loss_trace]

dmd_iter = list(range(0, len(dmd_loss_list[0])))
dmd_iter = [x*horizon for x in dmd_iter]

# load dkr results
dkr_loss_list = []
for i in range(5):
    load = sio.loadmat('cartpole' +  str(i) + '_DKRloss.mat')
    loss_trace = load['loss_trace'].flatten()
    dkr_loss_list += [loss_trace]

dkr_iter = list(range(0, len(dkr_loss_list[0])))
dkr_iter = [100*x*horizon for x in dkr_iter]


params = {'axes.labelsize': 40,
          'axes.titlesize': 30,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize':24}
plt.rcParams.update(params)

fig2 = plt.figure(figsize=(11, 9))

divider2 = Divider(fig2, (0.0, 0.0, 1., 1.), h, v, aspect=False)
# the width and height of the rectangle is ignored.

ax2 = Axes(fig2, divider2.get_position())
ax2.set_axes_locator(divider2.new_locator(nx=1, ny=1))

fig2.add_axes(ax2)

# ax2.set_xscale('symlog')
ax2.set_yscale('log')
ax2.set_xlim(-2400,120000)
ax2.set_ylim(bottom=1e-5,top=1e5)
ax2.ticklabel_format(axis='x', style='sci', scilimits=(4,4))
ax2.set_xlabel('Number of Data Points')
ax2.set_ylabel('SysID Loss')
ax2.tick_params(axis='both', which='major')
ax2.set_facecolor('#E6E6E6')
ax2.grid()
ax2.set_position([1,1,10,6])
# plot results
for dmd_loss in dmd_loss_list:
    ax2.plot(dmd_iter, dmd_loss, color = [0.9290, 0.6940, 0.1250], linewidth=4)

for pdp_loss in pdpnn_loss_list:
    ax2.plot(pdpnn_iter, pdp_loss, color = [0.6350, 0.0780, 0.1840], linewidth=4)

for dkr_loss in dkr_loss_list:
    ax2.plot(dkr_iter, dkr_loss, color = [0.4660, 0.6740, 0.1880], linewidth=4)


# show legend
line_dmd,=ax2.plot(pdpnn_iter, pdpnn_loss_list[0], color = [0.9290, 0.6940, 0.1250], linewidth=4)
line_pdpnn,=ax2.plot(pdpnn_iter, pdpnn_loss_list[0], color = [0.6350, 0.0780, 0.1840], linewidth=4)
line_dkr,=ax2.plot(dkr_iter, dkr_loss_list[0], color = [0.4660, 0.6740, 0.1880], linewidth=4)


for OCIL_loss in OCILnn_loss_list:
    ax2.plot(OCIL_loss, color='b', linewidth=4)

line_OCILnn, = ax2.plot(OCILnn_loss_list[0], color='b', linewidth=4)

ax2.legend([line_OCILnn,line_pdpnn,line_dmd,line_dkr],['OCIL (Proposed)','PDP','DMDc','DKR'],
            facecolor='white',framealpha=0.5,loc=1, fontsize=24)

plt.show()
