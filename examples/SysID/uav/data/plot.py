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

# # load dmd results
# dmd_loss_list = []
# for i in range(5):
#     load = sio.loadmat('DMD_results_trial_' + str(i))
#     loss_trace = load['results']['loss_trace'][0, 0].flatten()
#     index = np.argwhere(loss_trace > 1000)
#     loss_trace[index] = loss_trace[
#         index - 1]  # remove the spikes inside the data (only for kkt results, because it is too bad)
#     dmd_loss_list += [loss_trace]

# dmd_iter = list(range(0, len(dmd_loss_list[0])))
# dmd_iter = [x*horizon for x in dmd_iter]

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
plt.show()
