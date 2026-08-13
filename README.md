<div align="center">

# Online Control-Informed Learning

**Zihao Liang · Tianyu Zhou · Zehui Lu · Shaoshuai Mou**
Purdue University

Transactions on Machine Learning Research, 2025

[**Paper**](https://openreview.net/forum?id=LDzvZEVl5H) ·
[**Project page**](https://zihaoliang.github.io/OCIL-Online-Control-Informed-Learning/) ·
[**Code**](https://github.com/ZihaoLiang/OCIL-Online-Control-Informed-Learning)

[![TMLR 2025](https://img.shields.io/badge/TMLR-2025-b31b1b.svg)](https://openreview.net/forum?id=LDzvZEVl5H)
[![Project page](https://img.shields.io/badge/project%20page-live-e8490f.svg)](https://zihaoliang.github.io/OCIL-Online-Control-Informed-Learning/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

<img src="images/OPDP_1.png" width="92%" alt="The OCIL loop: the parameter estimate drives the system, the gradient generator differentiates its trajectory, and the chain rule feeds the result back to the estimator.">

</div>

We consider any robot as a **tunable optimal control system**, parameterised by
tunable parameters within its dynamics, its policy and its objective function.
OCIL learns those parameters from a stream of data, one measurement at a time.

Two parts do the work. A **gradient generator** obtains the exact derivative of
the trajectory with respect to the parameters, by differentiating through
Pontryagin's Maximum Principle. An **online parameter estimator** based on the
extended Kalman filter then corrects them as each measurement arrives.

There are no epochs, no replay buffer and no batch to wait for, and because the
filter carries an estimate of its own uncertainty it tolerates noisy
measurements rather than fitting them.

> **[Try the interactive project page →](https://zihaoliang.github.io/OCIL-Online-Control-Informed-Learning/)**
> Scrub through training and watch the pendulum, quadrotor, cart-pole and rocket
> landing converge onto their demonstrations.

## Three learning modes

One update rule, with a different part of the control problem left unknown.

| Mode | Unknown | Learned from | Example |
|---|---|---|---|
| **Online imitation learning** | cost weights, dynamics, or both | one demonstrated trajectory | [`examples/ImitationLearning`](examples/ImitationLearning) |
| **Online system identification** | dynamics, as parameters or a neural network | recorded inputs and states | [`examples/SysID`](examples/SysID) |
| **Policy tuning on the fly** | weights of a neural feedback policy | the control objective | [`examples/PolicyTuning`](examples/PolicyTuning) |

<div align="center">
<img src="images/IL_cartpole.png" width="32%" alt="Imitation learning on the cart-pole">
<img src="images/SysID_uav.png" width="32%" alt="System identification on the quadrotor">
<img src="images/PT_cartpole.png" width="32%" alt="Policy tuning on the cart-pole">

<em>Loss against the number of data points consumed, over five trials, one panel
per mode. The red line marks one pass over the data. The paler series is OCIL on
noisy measurements.</em>
</div>

## Installation

The submodule carries the optimal control and gradient machinery, so clone
recursively:

```bash
git clone --recursive https://github.com/ZihaoLiang/OCIL-Online-Control-Informed-Learning.git
cd OCIL-Online-Control-Informed-Learning
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Then install the dependencies. Tested on Python 3.10 with NumPy 1.26, SciPy 1.15,
Matplotlib 3.10 and CasADi 3.7:

```bash
conda create -n ocil python=3.10
conda activate ocil
pip install numpy scipy matplotlib casadi
```

Those four packages are all the examples below need. Some scripts inside the
Pontryagin Differentiable Programming submodule additionally import PyTorch, but
nothing in `examples/` does.

## Quick start

**Run every script from the repository root.** Each one builds its import paths
and its data paths from the working directory, so launching from inside an
example folder will not find `src/`.

```bash
python examples/ImitationLearning/pendulum/pendulum_OCIL.py
```

This recovers two cost weights from a single 20-step demonstration. The loss
falls by roughly three orders of magnitude within one pass over the demo. A
window with the loss curve and the learned trajectory opens when it finishes.

## Examples

Each script sets up a system, picks a learning mode, initialises the filter and
calls `solve()`. The covariances `P`, `Q` and `R` at the top of a script control
how aggressively the estimate moves.

### Online imitation learning &nbsp;·&nbsp; [`examples/ImitationLearning/`](examples/ImitationLearning)

Recover what the demonstrator was optimising, and the physics it was operating
under, from a single trajectory.

| System | Script | Learns |
|---|---|---|
| Pendulum | `pendulum/pendulum_OCIL.py` | 2 cost weights |
| Cart-pole | `cartpole/cartpole_OCIL.py` | 3 dynamics + 4 cost |
| Quadrotor | `uav/uav_OCIL.py` | 5 dynamics + 4 cost |
| Rocket | `rocket/rocket_OCIL.py` | 5 dynamics + 5 cost |

### Online system identification &nbsp;·&nbsp; [`examples/SysID/`](examples/SysID)

Recover the dynamics from recorded inputs and states. The `nn_` scripts replace
the physical model with a neural network written as a CasADi expression; the
`_PDP` scripts run Pontryagin Differentiable Programming on the same problem for
comparison.

| System | Physical parameters | Neural dynamics |
|---|---|---|
| Pendulum | `pendulum/pendulum_OCIL.py` | — |
| Cart-pole | `cartpole/cartpole_OCIL.py` | `cartpole/nn_cartpole.py` |
| Quadrotor | `uav/uav_OCIL.py` | `uav/nn_uav.py` |
| Rocket | `rocket/rocket_OCIL.py` | `rocket/nn_rocket.py` |

### Policy tuning on the fly &nbsp;·&nbsp; [`examples/PolicyTuning/`](examples/PolicyTuning)

Tune the weights of a neural feedback policy online. `_traj` matches both states
and controls against a reference, `_state` matches states only, and `_obj`
minimises the control cost directly.

| System | Track a trajectory | Minimise the objective |
|---|---|---|
| Pendulum | `pendulum/pendulum_OCIL_traj.py`<br>`pendulum/pendulum_OCIL_state.py` | — |
| Cart-pole | `cartpole/cartpole_OCIL_traj.py` | `cartpole/cartpole_OCIL_obj.py` |
| Quadrotor | `uav/uav_OCIL_traj.py` | `uav/uav_OCIL_obj.py` |
| Rocket | `rocket/rocket_OCIL_traj.py` | `rocket/rocket_OCIL_obj.py` |

## How the code is organised

```
src/
  OCIL.py           the three learning modes: ImitationLearning, SysID, PolicyTuning
  EKF.py            the extended Kalman filter predict and update steps
  JinEnv_NN.py      pendulum, cart-pole, quadrotor and rocket, with neural
                    dynamics and neural cost variants
externals/
  Pontryagin-Differentiable-Programming/
                    the optimal control solver, the auxiliary control system
                    that produces the gradient, and the original environments
examples/           one folder per learning mode, per system, with the
                    demonstration data and saved baseline results
test/               standalone iterative LQR scripts
images/             the figures above
```

Every mode runs the same five steps for each incoming data point. Solve the
optimal control problem with the current parameters. Build the auxiliary control
system and solve it for the derivative of the trajectory with respect to those
parameters. Take the residual between the prediction and the observation at one
time step. Chain the two together into the filter's measurement matrix. Correct
the parameters.

## Notes

- **The plots block.** Each `solve()` ends with a blocking Matplotlib window. For
  unattended runs set a non-interactive backend: `MPLBACKEND=Agg python ...`

- **Cost weights are identifiable only up to scale.** Multiplying every weight in
  the objective by the same constant leaves the optimal trajectory unchanged, so
  the parameter error can look large while the trajectory matches the
  demonstration exactly. Judge imitation results by the trajectory loss, and
  compare cost weights by their ratios.

- **Learning dynamics and cost together is sensitive to the starting guess.**
  Mode `"All"` converges from most random initialisations but not all. If a run
  flattens out at a high loss, re-run it, or set the spread of the initial guess
  with `set_sigma()`.

- **Cost per data point is high.** Every update re-solves the full optimal control
  problem and the full auxiliary system, then uses one time index of the result.
  This is what buys an exact gradient rather than an approximate one, but it means
  the cost of a pass grows with the square of the horizon.

## Reproducing the figures

Each example folder carries the saved loss traces for OCIL and for the baselines
it is compared against, along with the script that draws the figure:

```bash
cd examples/ImitationLearning/cartpole/data
python plot.py
```

Baselines included in the saved results are Pontryagin Differentiable
Programming, inverse KKT, dynamic mode decomposition with control, guided policy
search, iterative LQR, and neural networks trained with Adam.

## Citation

```bibtex
@article{liang2025online,
  title   = {Online Control-Informed Learning},
  author  = {Liang, Zihao and Zhou, Tianyu and Lu, Zehui and Mou, Shaoshuai},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2025},
  url     = {https://openreview.net/forum?id=LDzvZEVl5H}
}
```

## Acknowledgements

The optimal control solver, the auxiliary control system used to generate
gradients, and the simulated environments come from
[Pontryagin Differentiable Programming](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming)
by Wanxin Jin and colleagues.

## License

MIT. See [LICENSE](LICENSE).
