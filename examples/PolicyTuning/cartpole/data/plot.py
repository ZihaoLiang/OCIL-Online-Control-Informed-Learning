import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes

params = {'axes.labelsize': 50,
          'axes.titlesize': 30,
          'xtick.labelsize':25,
          'ytick.labelsize':25,
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

# ax.set_xscale('symlog')
# ax.set_yscale('log')
# ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
ax.set_xlim(-10,5000)
ax.set_ylim(bottom=-10,top=100)
ax.set_xlabel('Number of Data Points')
ax.set_ylabel('Policy Tuning Loss')
ax.tick_params(axis='both', which='major')
ax.set_facecolor('#E6E6E6')
ax.grid()
ax.set_position([1,1,10,6])

demo = sio.loadmat('cartpole_oc.mat')
horizon = demo['horizon'][0][0]
true_cost = demo['cost'][0][0]

# load OCIL results
OCIL_loss_list = []
for i in range(5):
    load = sio.loadmat('OCIL_results_' + str(i))
    loss_trace = load['Loss'][0]
    OCIL_loss_list += [loss_trace-true_cost]

OCIL_iter = list(range(0, len(OCIL_loss_list[0])))

OCIL_avg = np.mean(OCIL_loss_list, 0)
OCIL_std = np.std(OCIL_loss_list, 0)
OCIL_ub = OCIL_avg + 3*OCIL_std
OCIL_lb = OCIL_avg - 3*OCIL_std
for i in range(len(OCIL_lb)):
    if OCIL_lb[i] < 0:
        OCIL_lb[i] = 0

# load pdp results
pdp_loss_list = []
for i in range(5):
    load = sio.loadmat('PDP_Neural_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    pdp_loss_list += [loss_trace-true_cost]

pdp_iter = list(range(0, len(pdp_loss_list[0])))
pdp_iter = [x*horizon for x in pdp_iter]

pdp_avg = np.mean(pdp_loss_list, 0)
pdp_std = np.std(pdp_loss_list, 0)
pdp_ub = pdp_avg + 3*pdp_std
pdp_lb = pdp_avg - 3*pdp_std
for i in range(len(pdp_lb)):
    if pdp_lb[i] < 0:
        pdp_lb[i] = 0

# load nn results
gps_loss_list = []
for i in range(5):
    load = sio.loadmat('GPS_Neural_trial_' + str(i))
    loss_trace = load['results']['loss_trace'][0, 0].flatten()
    gps_loss_list += [loss_trace-true_cost]

gps_iter = list(range(0, len(gps_loss_list[0])))
gps_iter = [x*horizon for x in gps_iter]

gps_avg = np.mean(gps_loss_list, 0)
gps_std = np.std(gps_loss_list, 0)
gps_ub = gps_avg + 3*gps_std
gps_lb = gps_avg - 3*gps_std
for i in range(len(gps_lb)):
    if gps_lb[i] < 0:
        gps_lb[i] = 0

# plot results
line_OCIL, = ax.plot(OCIL_avg, color='b', linewidth=4)
line_pdp, = ax.plot(pdp_iter, pdp_avg, color = [0.6350, 0.0780, 0.1840], linewidth=4)
line_gps, = ax.plot(gps_iter, gps_avg, color=[0.4660, 0.6740, 0.1880], linewidth=4)
ax.fill_between(gps_iter, gps_lb, gps_ub, color='lightgreen')
ax.fill_between(pdp_iter, pdp_lb, pdp_ub, color='lightcoral', alpha=0.7)
ax.fill_between(OCIL_iter, OCIL_lb, OCIL_ub, color='lightskyblue')

ax.legend([line_OCIL,line_pdp,line_gps],['OCIL (Proposed)','PDP with Neural Policy','GPS with Neural Policy'],
            facecolor='white',framealpha=0.5,loc=1, fontsize=24)


fig2 = plt.figure(figsize=(11, 9))

divider2 = Divider(fig2, (0.0, 0.0, 1., 1.), h, v, aspect=False)
# the width and height of the rectangle is ignored.

ax2 = Axes(fig2, divider2.get_position())
ax2.set_axes_locator(divider2.new_locator(nx=1, ny=1))

fig2.add_axes(ax2)

# ax2.set_xscale('symlog')
# ax2.set_yscale('log')
# ax2.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
ax2.set_xlim(0,30)
ax2.set_ylim(bottom=0,top=1000)
ax2.set_xlabel('Number of Data Points')
ax2.set_ylabel('Policy Tuning Loss')
ax2.tick_params(axis='both', which='major')
ax2.set_facecolor('#E6E6E6')
ax2.grid()
ax2.set_position([1,1,10,6])

# for OCIL_loss in OCIL_loss_list:
#     ax2.plot(OCIL_loss[:horizon], color='b', linewidth=4)
ax2.plot(OCIL_avg[:horizon], color='b', linewidth=4)
ax2.fill_between(OCIL_iter[:horizon], OCIL_lb[:horizon], OCIL_ub[:horizon], color='lightskyblue')


plt.show()
