# # 1D Quantum Harmonic Oscillator

# %% import PyTorch
import torch
# impor numeric libraries
import numpy as np
import matplotlib.pyplot as plt

# import Custom Modules
import parameters as pm
import utilities as utils
import plots

# import Customs Classes
import models
from analysis import Dynamics
import stochastic_reconfiguration as SR
# import Custom Functions
from integrators import integrator
from scipy.interpolate import interp1d

from matplotlib.colors import TwoSlopeNorm

#import benchmark functions
import benchmarks 

import exact_nls

import random
import os
#%% Default type
torch.set_default_dtype(torch.float64)
#%% General parameters --------------------------------------------------

# Hardware (CPU or GPU)
dev = 'cpu' # can be changed to 'cuda' for GPU usage
device = torch.device(dev)

# Seed of the random number generator
seed = 2                                   
torch.manual_seed(seed)

# Model ID
file_name = lambda *args: "_".join(args)

# Create a spacial grid object
grid= utils.PointGrid(N=401, start=-24, end=24, device=device)
grid_col= utils.PointGrid(N=401, start=-24, end=24, device=device)
# sample from a distribution
# grid.sampler(lambda _: 1.0)

# Choose model
pm.architecture = 'NQS'
#overwrite
pm.overwrite = False
# Network architecture
input_size = 1
output_size = 1
hidden_layers = [5,5]

# Create Neural Quantum State
model = models.NQS(input_size, output_size, hidden_layers).to(device)

#%% Ground State Search
print('Ground State Search')

# Initial conditions
# trap
pm.x0 = 0
pm.w = 0.1
pm.k=0.0
pm.gauss_amplitude = 0.
pm.gauss_width = 1

pm.wall = 0

# Time parameters
pm.dt = 0.1
pm.t_max = 100

# Integrator parameters
pm.evolution = 'imag'
pm.lambda_reg = (1) * 1e-3
pm.e_error = 1e-8
pm.enforce_even_parity = False
den_list = []
g_list = []
w_list = []
psi_list = []
mesh_list = []
time_list = []
energy_list = []
params_list = []
#If g=0, we study de HO case
for pm.g, pm.w in zip([-1,-1], [0.1,0.0]):
    if np.isclose(pm.w, 0.0):
        pm.t_max = 100
        pm.dt = 0.1
    # Perform imaginary-time evolution
    file_path = utils.file_ID(
        pm.data_dir,
        file_name(pm.architecture, model.architecture, pm.evolution) + f'_g_{pm.g}_w_{pm.w}',
        pm.data_format
    )
    x = grid.x()
    with torch.no_grad():
        lnpsi = model(x)
        psi = torch.exp(lnpsi)
        print(f"lnpsi real: min={lnpsi.real.min():.2f}, max={lnpsi.real.max():.2f}")
        print(f"lnpsi imag: min={lnpsi.imag.min():.2f}, max={lnpsi.imag.max():.2f}")
        print(f"psi abs:    min={psi.abs().min():.2e}, max={psi.abs().max():.2e}")
        print(f"finito lnpsi: {torch.isfinite(lnpsi).all()}")
        print(f"finito psi:   {torch.isfinite(psi).all()}")
    integrator(model, grid, file_path=file_path)

    # Load the evolution
    imag_evo = Dynamics(file_path=file_path, x_grid=grid.x())

    # Compute density
    psi = imag_evo.psi
    den = np.abs(imag_evo.psi) ** 2

    # Final density normalized
    rho_num = den[-1, :]
    rho_num = rho_num / np.trapezoid(rho_num, grid.mesh)

    # Generic observables from the final density
    x_mean = np.trapezoid(grid.mesh * rho_num, grid.mesh)
    x2_mean = np.trapezoid((grid.mesh ** 2) * rho_num, grid.mesh)
    var_x = x2_mean - x_mean ** 2

    # Final energy
    E_gs = imag_evo.energy[-1]

    # Print basic diagnostics
    print(f"\nCase: g = {pm.g}, w = {pm.w}")
    print("norm initial:", np.trapezoid(den[0, :], grid.mesh))
    print("norm final:",   np.trapezoid(den[-1, :], grid.mesh))
    print("rho_num normalization:", np.trapezoid(rho_num, grid.mesh))
    print("max final density:", den[-1, :].max())
    print("t_final:", imag_evo.t_grid[-1])
    print("edge density:", den[-1, 0], den[-1, -1])
    print(f"<x> numerical = {x_mean:.6e}")
    print(f"Var(x) numerical = {var_x:.6e}")
    print(f"E_gs numerical = {E_gs:.6e}")

    # Get parameters
    params = imag_evo.get_params()

    # Plot full evolution
    fig_path = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, pm.evolution) + f'_g_{pm.g}',
        pm.fig_format
    )
    plots.evo_fig_params_poster(
        imag_evo.t_grid,
        grid.mesh,
        den.T,
        params,
        fig_path=fig_path
    )
    energy_fig_path = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'energy_imag')
        + f'_g_{pm.g}_w_{pm.w}',
        pm.fig_format
    )

    plots.energy_imag_fig(
        imag_evo.t_grid,
        imag_evo.energy,
        fig_path=energy_fig_path,
        title=rf'$g={pm.g},\ \omega={pm.w}$'
    )

    # ---- HO benchmark: compare final density with exact result ----
    if benchmarks.is_ho_linear_benchmark(
        g=pm.g,
        k=pm.k,
        gauss_amplitude=pm.gauss_amplitude,
        wall=pm.wall
    ):
        rho_exact = benchmarks.ho_ground_state_density(grid.mesh, omega=pm.w, x0=pm.x0)
        rho_exact = rho_exact / np.trapezoid(rho_exact, grid.mesh)

        err_den = benchmarks.relative_l2_error_density(rho_num, rho_exact, grid.mesh)

        x_mean_exact = pm.x0
        var_x_exact = 1.0 / (2.0 * pm.w)

        # Because hamiltonian() currently adds a constant +1 offset
        E_exact = pm.w / 2.0 + 1.0

        print(f'[HO benchmark] g={pm.g}, w={pm.w}')
        print(f'  Relative L2 density error = {err_den:.6e}')
        print(f'  <x> exact = {x_mean_exact:.6e}')
        print(f'  Var(x) exact = {var_x_exact:.6e}')
        print(f'  Error in <x> = {abs(x_mean - x_mean_exact):.6e}')
        print(f'  Error in Var(x) = {abs(var_x - var_x_exact):.6e}')
        print(f'  Error in E = {abs(E_gs - E_exact):.6e}')
        fig_path_benchmark = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'ho_benchmark') + f'_g_{pm.g}_w_{pm.w}',
        pm.fig_format
        )

        plots.ho_benchmark_density(
        grid.mesh,
        rho_num,
        rho_exact,
        pm.w,
        fig_path=fig_path_benchmark
        )
    if benchmarks.is_bright_soliton_benchmark(
    g=pm.g,
    w=pm.w,
    k=pm.k,
    gauss_amplitude=pm.gauss_amplitude,
    wall=pm.wall
    ):
        rho_max_num = np.max(rho_num)

        rho_exact = benchmarks.bright_soliton_density(
            grid.mesh,
            rho_max=rho_max_num,
            x0=pm.x0
            )

        rho_exact = rho_exact / np.trapezoid(rho_exact, grid.mesh)

        err_den = benchmarks.relative_l2_error_density(rho_num, rho_exact, grid.mesh)
        E_exact = benchmarks.bright_soliton_energy_code_rhomax(rho_max_num)

        print(f'[Bright soliton benchmark] g={pm.g}, w={pm.w}')
        print(f'  Relative L2 density error = {err_den:.6e}')
        print(f'  <x> exact = {pm.x0:.6e}')
        print(f'  Error in <x> = {abs(x_mean - pm.x0):.6e}')
        print(f'  E exact (current code) = {E_exact:.6e}')
        print(f'  Error in E = {abs(E_gs - E_exact):.6e}')

        fig_path_benchmark = utils.file_ID(
            pm.figs_dir,
            file_name(pm.architecture, model.architecture, 'bright_soliton_benchmark') + f'_g_{pm.g}',
            pm.fig_format
        )

        plots.bright_soliton_benchmark_density(
            grid.mesh,
            rho_num,
            rho_exact,
            rho_max_num,
            fig_path=fig_path_benchmark
        )
        summary_fig_path = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'bright_soliton_summary') + f'_g_{pm.g}_w_{pm.w}',
        pm.fig_format
        )

        plots.bright_soliton_summary_fig(
        mesh=grid.mesh,
        time=imag_evo.t_grid,
        den=den,
        energy=imag_evo.energy,
        rho_final=rho_num,
        rho_exact=rho_exact,
        fig_path=summary_fig_path,
        title=rf'$g={pm.g},\ \omega={pm.w}$'
        )

    # Save data
    g_list.append(pm.g)
    den_list.append(den)
    psi_list.append(psi)
    mesh_list.append(grid.mesh)
    w_list.append(pm.w)
    time_list.append(imag_evo.t_grid)
    energy_list.append(imag_evo.energy)
    params_list.append(params)
wi = pm.w
# save last ground state for the dynamics
gs_model_path = utils.file_ID(
    pm.data_dir,
    file_name(pm.architecture, model.architecture, 'ground_state') + f'_g_{pm.g}_w_{pm.w}',
    'pt'
)
torch.save(model.state_dict(), gs_model_path)

print(f"Last ground state saved in: {gs_model_path}")
E_gs = imag_evo.energy[-1]
print(f"Last ground state energy: {E_gs}")

#%%
import matplotlib.pyplot as plt 
#d01 = den_list[0][-1, :]   # caso w=0.01 (segundo)
#d00 = den_list[1][-1, :]   # caso w=0 (tercero)

#diff_rel = np.max(np.abs(d01 - d00)) / np.max(d00)
#print("max relative diff:", diff_rel)
for g, den, mesh, w in zip(g_list, den_list, mesh_list, w_list):
    norm = np.trapezoid(den[-1,:], mesh)
    plt.plot(mesh, den[-1,:] / norm, label=f'g={str(g)}, w={str(w)}')
x_max =grid.mesh[np.where(den[-1,:]==den[-1,:].max())]
plt.vlines(x_max,0,den[-1,:].max(),linestyles=':',label=f'{round(x_max.item(),3)}')
plt.legend()
plt.show()
# %%
for g, psi in zip(g_list, psi_list):
    plt.plot(grid.mesh, psi[-1,:].real, label=str(g))
    plt.plot(grid.mesh, psi[-1,:].imag, '--', label=str(g))
x_max =grid.mesh[np.where(den[-1,:]==den[-1,:].max())]
plt.vlines(x_max,0,0.3,label=f'{x_max}')
plt.legend()
plt.show()

# %%
print('Dynamics')
# load last ground state
model.load_state_dict(torch.load(gs_model_path, map_location=device))
model.to(device)
model.eval()
pm.pin_strength = 0
# trap
pm.x0 = 0
wf = 0.0
pm.w = wf

pm.gauss_amplitude = 0

pm.k = 0.3

pm.g=-1

# Time parameters
pm.dt = 0.01
pm.t_max = 40

# Integrator parameters
pm.evolution = 'real'
pm.lambda_reg =1e-5
print(pm.k,pm.w)

# Perform real time evolution
file_path = utils.file_ID(
    pm.data_dir,
    file_name(pm.architecture, model.architecture, pm.evolution)
    + f'_g_{pm.g}_w_{pm.w}_k_{pm.k}',
    pm.data_format
)
integrator(model, grid, file_path=file_path)

# Get the dynamics
real_evo = Dynamics(file_path=file_path, x_grid=grid.x())

# Compute density
psi = real_evo.psi
den = np.abs(real_evo.psi)**2
# Get parameters
params = real_evo.get_params()
# Time-dependent normalized density
norm_t = np.trapezoid(den, grid.mesh, axis=1)
rho_t = den / norm_t[:, None]

# Time-dependent moments
x_mean_t = np.trapezoid(grid.mesh[None, :] * rho_t, grid.mesh, axis=1)
x2_mean_t = np.trapezoid((grid.mesh[None, :] ** 2) * rho_t, grid.mesh, axis=1)
var_x_t = x2_mean_t - x_mean_t ** 2

# Energies
E_t = np.real(real_evo.energy)
E_0 = E_t[0]


# --- Peak observables for any real-time run ---
x_max_t = grid.mesh[np.argmax(rho_t, axis=1)]
rho_max_t = np.max(rho_t, axis=1)
rho_max_rel = (rho_max_t - rho_max_t[0]) / rho_max_t[0]

# Reference trajectory for the peak position
# If there is a kick, use ballistic reference. If not, keep it constant.
if np.isclose(pm.k, 0.0):
    x_max_ref = np.full_like(real_evo.t_grid, x_max_t[0], dtype=float)
else:
    x_max_ref = x_max_t[0] + pm.k * real_evo.t_grid

print("rho_max(0) =", rho_max_t[0])
print("rho_max(end) =", rho_max_t[-1])
print("max relative rho_max drift =", np.max(np.abs(rho_max_rel)))
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

# x_max(t)
axes[0].plot(real_evo.t_grid, x_max_t, label='NQS')
axes[0].plot(real_evo.t_grid, x_max_ref, '--', label='Reference')
axes[0].set_ylabel(r'$x_{\max}(t)$')
axes[0].legend()

# rho_max(t)
axes[1].plot(real_evo.t_grid, rho_max_t, label='NQS')
axes[1].axhline(rho_max_t[0], linestyle='--', label='Initial')
axes[1].set_ylabel(r'$\rho_{\max}(t)$')
axes[1].legend()

# relative drift
axes[2].plot(real_evo.t_grid, rho_max_rel, label='relative drift')
axes[2].axhline(0.0, linestyle='--')
axes[2].set_xlabel(r'$t$')
axes[2].set_ylabel(r'$\frac{\rho_{\max}(t)-\rho_{\max}(0)}{\rho_{\max}(0)}$')
axes[2].legend()

plt.tight_layout()
plt.show()
# Plot data
fig_path = utils.file_ID(pm.figs_dir,
                        file_name(pm.architecture, model.architecture, pm.evolution) + f'_g_{pm.g}',
                        pm.fig_format)
plots.evo_fig_params_poster(real_evo.t_grid, grid.mesh, den.T, params, fig_path=fig_path)
# Plot energy with GS reference
energy_fig_path = utils.file_ID(
    pm.figs_dir,
    file_name(pm.architecture, model.architecture, 'energy')
    + f'_g_{pm.g}_w_{pm.w}_k_{pm.k}',
    pm.fig_format
)

plots.energy_fig_with_gs(
    t_dyn=real_evo.t_grid,
    energy_dyn=real_evo.energy,
    energy_gs=E_gs,
    fig_path=energy_fig_path,
    ylabel=r'$E(t)$',
    relative=False
)

# Plot relative energy drift
energy_rel_fig_path = utils.file_ID(
    pm.figs_dir,
    file_name(pm.architecture, model.architecture, 'energy_rel')
    + f'_g_{pm.g}_w_{pm.w}_k_{pm.k}',
    pm.fig_format
)

plots.energy_fig_with_gs(
    t_dyn=real_evo.t_grid,
    energy_dyn=real_evo.energy,
    energy_gs=E_gs,
    fig_path=energy_rel_fig_path,
    ylabel=r'$\frac{E(t)-E(0)}{E(0)}$',
    relative=True
)
if benchmarks.is_ho_kick_benchmark(
    g=pm.g,
    w=pm.w,
    k=pm.k,
    gauss_amplitude=pm.gauss_amplitude,
    wall=pm.wall
):
    x_mean_exact = benchmarks.ho_kick_x_mean(real_evo.t_grid, pm.k, pm.w, x0=pm.x0)
    var_x_exact = benchmarks.ho_kick_variance(pm.w)
    E_exact = benchmarks.ho_kick_energy_code(pm.w, pm.k)

    err_x_mean = np.sqrt(np.trapezoid((x_mean_t - x_mean_exact) ** 2, real_evo.t_grid)) / \
                 np.sqrt(np.trapezoid(x_mean_exact ** 2, real_evo.t_grid))

    max_var_drift = np.max(np.abs(var_x_t - var_x_exact))
    max_energy_drift = np.max(np.abs(E_t - E_0))
    energy_error_0 = abs(E_0 - E_exact)
    max_norm_drift = np.max(np.abs(norm_t - 1.0))

    print('[HO kick benchmark]')
    print(f'  Relative L2 error in <x>(t) = {err_x_mean:.6e}')
    print(f'  Var(x) exact = {var_x_exact:.6e}')
    print(f'  Max |Var(x)-Var_exact| = {max_var_drift:.6e}')
    print(f'  E exact (current code) = {E_exact:.6e}')
    print(f'  Error in E(0) = {energy_error_0:.6e}')
    print(f'  Max |E(t)-E(0)| = {max_energy_drift:.6e}')
    print(f'  Max norm drift = {max_norm_drift:.6e}')

    fig_path_benchmark = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'ho_kick_benchmark')
        + f'_g_{pm.g}_w_{pm.w}_k_{pm.k}',
        pm.fig_format
    )

    plots.ho_kick_benchmark_fig(
        real_evo.t_grid,
        x_mean_t,
        x_mean_exact,
        fig_path=fig_path_benchmark
    )
if benchmarks.is_ho_quench_benchmark(
    g=pm.g,
    k=pm.k,
    wi=wi,
    wf=wf,
    gauss_amplitude=pm.gauss_amplitude,
    wall=pm.wall
):
    var_exact_t = benchmarks.ho_quench_variance(real_evo.t_grid, wi, wf)
    E_exact = benchmarks.ho_quench_energy_code(wi, wf)

    err_var = np.sqrt(np.trapezoid((var_x_t - var_exact_t) ** 2, real_evo.t_grid)) / \
              np.sqrt(np.trapezoid(var_exact_t ** 2, real_evo.t_grid))

    max_energy_drift = np.max(np.abs(E_t - E_0))
    energy_error_0 = abs(E_0 - E_exact)
    max_norm_drift = np.max(np.abs(norm_t - 1.0))
    max_x_mean = np.max(np.abs(x_mean_t))

    print('[HO quench benchmark]')
    print(f'  Relative L2 error in Var(x)(t) = {err_var:.6e}')
    print(f'  Max |<x>(t)| = {max_x_mean:.6e}')
    print(f'  E exact (current code) = {E_exact:.6e}')
    print(f'  Error in E(0) = {energy_error_0:.6e}')
    print(f'  Max |E(t)-E(0)| = {max_energy_drift:.6e}')
    print(f'  Max norm drift = {max_norm_drift:.6e}')

    omega_breath_exact = 2.0 * wf
    omega_breath_num, peak_times, periods = benchmarks.estimate_frequency_from_peaks(
        real_evo.t_grid, var_x_t
    )

    print(f'  omega_breath exact = {omega_breath_exact:.6e}')
    print(f'  omega_breath numerical = {omega_breath_num:.6e}')
    print(f'  Error in omega_breath = {abs(omega_breath_num - omega_breath_exact):.6e}')

    if len(periods) > 0:
        print(f'  Mean peak-to-peak period = {np.mean(periods):.6e}')
        print(f'  Std peak-to-peak period = {np.std(periods):.6e}')

    # --------------------------------------------------
    # Exact ground state used as initial state before the quench
    # --------------------------------------------------
    rho_gs_num = np.abs(imag_evo.psi[-1, :])**2
    rho_gs_num = rho_gs_num / np.trapezoid(rho_gs_num, grid.mesh)

    rho_gs_exact = benchmarks.ho_ground_state_density(
        grid.mesh,
        omega=wi,
        x0=pm.x0
    )
    rho_gs_exact = rho_gs_exact / np.trapezoid(rho_gs_exact, grid.mesh)

    # --------------------------------------------------
    # Old figures (optional: keep or comment out)
    # --------------------------------------------------
    fig_path_benchmark = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'ho_quench_benchmark')
        + f'_g_{pm.g}_wi_{wi}_wf_{wf}',
        pm.fig_format
    )

    plots.ho_quench_benchmark_fig(
        real_evo.t_grid,
        var_x_t,
        var_exact_t,
        fig_path=fig_path_benchmark
    )

    fig_path_freq = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'ho_quench_frequency')
        + f'_g_{pm.g}_wi_{wi}_wf_{wf}',
        pm.fig_format
    )

    plots.ho_quench_frequency_fig(
        real_evo.t_grid,
        var_x_t,
        var_exact_t,
        peak_times,
        fig_path=fig_path_freq
    )

    # --------------------------------------------------
    # New summary figure with 3 subplots
    # --------------------------------------------------
    fig_path_summary = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'ho_quench_summary')
        + f'_g_{pm.g}_wi_{wi}_wf_{wf}',
        pm.fig_format
    )

    plots.ho_quench_summary_fig(
        mesh=grid.mesh,
        rho_gs_num=rho_gs_num,
        rho_gs_exact=rho_gs_exact,
        time=real_evo.t_grid,
        den=den,
        var_num=var_x_t,
        var_exact=var_exact_t,
        fig_path=fig_path_summary,
        title=rf'$g={pm.g},\ \omega_i={wi},\ \omega_f={wf}$'
    )
if benchmarks.is_bright_soliton_kick_benchmark(
    g=pm.g,
    w=pm.w,
    k=pm.k,
    gauss_amplitude=pm.gauss_amplitude,
    wall=pm.wall
):
    x_max_t = benchmarks.density_peak_position(rho_t, grid.mesh)
    rho_max_t = benchmarks.density_peak_height(rho_t)

    # Use the numerical initial peak position as reference
    x_max_exact = benchmarks.bright_soliton_xmax(
        real_evo.t_grid,
        pm.k,
        x0=x_max_t[0]
    )

    err_x_max = benchmarks.relative_l2_error_signal(
        x_max_t,
        x_max_exact,
        real_evo.t_grid
    )

    max_norm_drift = benchmarks.max_relative_drift(norm_t)
    max_rho_drift = benchmarks.max_relative_drift(rho_max_t)

    print('[Bright soliton kick benchmark]')
    print(f'  Relative L2 error in x_max(t) = {err_x_max:.6e}')
    print(f'  Max relative norm drift = {max_norm_drift:.6e}')
    print(f'  Max relative rho_max drift = {max_rho_drift:.6e}')

    # --------------------------------------------------
    # Original benchmark figure
    # --------------------------------------------------
    fig_path_benchmark = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, 'bright_soliton_kick_benchmark')
        + f'_g_{pm.g}_w_{pm.w}_k_{pm.k}',
        pm.fig_format
    )

    plots.bright_soliton_kick_benchmark_fig(
        real_evo.t_grid,
        x_max_t,
        x_max_exact,
        rho_max_t,
        norm_t,
        fig_path=fig_path_benchmark
    )
# %%
norm = np.trapezoid(den[-1,:], mesh)
plt.plot(mesh, den[-1,:] / norm, label=f'g={str(g)}, w={str(w)}')
x_max =grid.mesh[np.where(den[-1,:]==den[-1,:].max())]
plt.vlines(x_max,0,den[-1,:].max(),linestyles=':',label=f'{round(x_max.item(),3)}')
plt.legend()
plt.show()
# %% NQS collision
print("NQS collision")

# ==================================================
# Single-seed setup
# ==================================================
NQS_SEED = 6

print("\n==================================================")
print(f"Running single NQS seed = {NQS_SEED}")
print("==================================================")

# Set random seeds
random.seed(NQS_SEED)
np.random.seed(NQS_SEED)
torch.manual_seed(NQS_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(NQS_SEED)

# Store outputs in a seed-specific folder
SEED_TAG = f"seed_{NQS_SEED}"

pm.data_dir = os.path.join(pm.data_dir, SEED_TAG)
pm.figs_dir = os.path.join(pm.figs_dir, SEED_TAG)

os.makedirs(pm.data_dir, exist_ok=True)
os.makedirs(pm.figs_dir, exist_ok=True)

print(f"Seed data_dir = {pm.data_dir}")
print(f"Seed figs_dir = {pm.figs_dir}")

# ==================================================
# 0) Master parameters
# ==================================================
scenario = "collision"

# Spatial grid for both NQS and exact blocks
x_grid = grid_col.x()
x_t = x_grid.squeeze().real

x_common = (
    grid_col.mesh.detach().cpu().numpy()
    if torch.is_tensor(grid_col.mesh)
    else np.asarray(grid_col.mesh, dtype=float)
)

# Physical parameters
g_col = -1.0
x0_sep = 12.0
phi_desired = 0.0

# Norms / masses
N_left = 1.0
N_right = 1.0
N_tot = N_left + N_right

# Velocities / kicks
k_left = +0.3
k_right = -0.3

# NQS real-time parameters
pm.evolution = "real"
pm.w = 0.0
pm.x0 = 0.0
pm.k = 0.0
pm.pin_strength = 0.0
pm.g = g_col
pm.dt = 0.005
pm.t_max = 3.0
pm.lambda_reg = 1e-3
pm.stopper = False
pm.target_norm = N_tot
pm.enforce_even_parity = False
pm.enforce_odd_parity = False

hidden_layers_c = [5, 5]

print("\n--- Master collision parameters ---")
print(f"N_left      = {N_left}")
print(f"N_right     = {N_right}")
print(f"N_tot       = {N_tot}")
print(f"x0_sep      = {x0_sep}")
print(f"k_left      = {k_left}")
print(f"k_right     = {k_right}")
print(f"phi_desired = {phi_desired}")
print(f"g_col       = {g_col}")
print(f"grid size   = {len(x_common)}")
print(f"t_max       = {pm.t_max}")

# ==================================================
# 1) Build initial target state on the same grid
# ==================================================
psi_target = utils.two_soliton_analytical(
    x_common,
    N_left,
    N_right,
    x0=x0_sep,
    k1=abs(k_left),
    k2=abs(k_right),
    phi=phi_desired,
    g=g_col,
)

if isinstance(psi_target, torch.Tensor):
    psi_target = psi_target.clone().detach().to(torch.complex128)
else:
    psi_target = torch.tensor(
        psi_target,
        dtype=torch.complex128,
        device=device,
    )

rho_target_phys_t0 = torch.abs(psi_target) ** 2

print(f"target norm    = {torch.trapz(rho_target_phys_t0, x_t).real.item():.6f}")
print(f"target rho_max = {torch.max(rho_target_phys_t0).item():.6f}")

# ==================================================
# 2) Create and fit NQS
# ==================================================
model_collision = models.NQS(
    input_size,
    output_size,
    hidden_layers_c,
).to(device)

model_collision, loss_history = utils.fit_model_to_target(
    model_collision,
    x_grid,
    psi_target,
    lr=1e-3,
    epochs=60000,
    verbose=True,
)

# ==================================================
# 3) Check initial fit
# ==================================================
with torch.no_grad():
    psi_fit = torch.exp(model_collision(x_grid)).squeeze()

    norm_fit = torch.trapz(
        torch.abs(psi_fit) ** 2,
        x_grid.squeeze(),
    ).real

    psi_fit = psi_fit / torch.sqrt(norm_fit) * np.sqrt(pm.target_norm)

psi_target_np = psi_target.detach().cpu().numpy()
psi_fit_np = psi_fit.detach().cpu().numpy()

rho_target_phys = np.abs(psi_target_np) ** 2
rho_fit_phys = np.abs(psi_fit_np) ** 2

psi_target_shape = benchmarks.normalize_wavefunction(
    psi_target_np,
    x_common,
)

psi_fit_shape = benchmarks.normalize_wavefunction(
    psi_fit_np,
    x_common,
)

rho_target_shape = np.abs(psi_target_shape) ** 2
rho_fit_shape = np.abs(psi_fit_shape) ** 2

err_rho_init = benchmarks.relative_l2_error_density(
    rho_fit_shape,
    rho_target_shape,
    x_common,
)

overlap_init = benchmarks.overlap_wavefunctions(
    psi_target_shape,
    psi_fit_shape,
    x_common,
)

fidelity_init = benchmarks.fidelity_wavefunctions(
    psi_target_shape,
    psi_fit_shape,
    x_common,
)

err_psi_init = benchmarks.relative_l2_error_wavefunction(
    psi_fit_shape,
    psi_target_shape,
    x_common,
)

err_psi_aligned_init = benchmarks.relative_l2_error_wavefunction_phase_aligned(
    psi_fit_shape,
    psi_target_shape,
    x_common,
)

print("\n--- NQS initial fit diagnostics ---")
print(f"norm target raw            = {np.trapezoid(rho_target_phys, x_common):.6f}")
print(f"norm fit raw               = {np.trapezoid(rho_fit_phys, x_common):.6f}")
print(f"Error L2 density shape     = {err_rho_init:.6e}")
print(f"Error L2 psi shape         = {err_psi_init:.6e}")
print(f"Error L2 psi aligned       = {err_psi_aligned_init:.6e}")
print(f"|<target|fit>|             = {np.abs(overlap_init):.6e}")
print(f"Fidelity                   = {fidelity_init:.6e}")
print(f"arg(<target|fit>)          = {np.angle(overlap_init):.6e} rad")

# ==================================================
# 4) Initial fit plots
# ==================================================
plt.figure(figsize=(8, 5))
plt.plot(
    x_common,
    rho_target_phys,
    label="target physical",
)
plt.plot(
    x_common,
    rho_fit_phys,
    "--",
    label="NQS fitted physical",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$|\psi|^2$")
plt.title("Initial NQS fit: physical density")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    x_common,
    rho_target_shape,
    label="target shape",
)
plt.plot(
    x_common,
    rho_fit_shape,
    "--",
    label="NQS fitted shape",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$|\psi|^2$ shape norm = 1")
plt.title("Initial NQS fit: shape-normalized density")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

phase_target = np.angle(psi_target_shape)
phase_fit = np.angle(psi_fit_shape)

plt.figure(figsize=(8, 5))
plt.plot(
    x_common,
    phase_target,
    label="target",
)
plt.plot(
    x_common,
    phase_fit,
    "--",
    label="NQS fitted",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$\arg(\psi)$")
plt.title("Initial phase: target vs NQS")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# ==================================================
# 5) Save fitted model and evolve
# ==================================================
collision_tag = (
    f"_g_{g_col}"
    f"_x0_{x0_sep}"
    f"_kL_{k_left}"
    f"_kR_{k_right}"
    f"_phi_{phi_desired}"
    f"_Ntot_{N_tot}"
)

fit_model_path = utils.file_ID(
    pm.data_dir,
    file_name(
        pm.architecture,
        model_collision.architecture,
        "collision_init_fit",
    )
    + collision_tag,
    "pt",
)

torch.save(
    model_collision.state_dict(),
    fit_model_path,
)

print(f"Fitted model saved in: {fit_model_path}")

file_path_col = utils.file_ID(
    pm.data_dir,
    file_name(
        pm.architecture,
        model_collision.architecture,
        "collision_free",
    )
    + collision_tag,
    pm.data_format,
)

integrator(
    model_collision,
    grid_col,
    file_path=file_path_col,
)

# ==================================================
# 6) Load NQS evolution
# ==================================================
col_evo = Dynamics(
    file_path=file_path_col,
    x_grid=grid_col.x(),
)

psi_nqs_tx = col_evo.psi
rho_nqs_phys_tx = np.abs(psi_nqs_tx) ** 2

norm_nqs_t = np.trapezoid(
    rho_nqs_phys_tx,
    x_common,
    axis=1,
)

rho_nqs_shape_tx = rho_nqs_phys_tx / norm_nqs_t[:, None]

print("\n--- NQS time-evolution diagnostics ---")
print(f"norm(t=0)            = {norm_nqs_t[0]:.6f}")
print(f"norm(t=end)          = {norm_nqs_t[-1]:.6f}")
print(f"rho_phys_max(t=0)    = {np.max(rho_nqs_phys_tx[0]):.6f}")
print(f"rho_phys_max(t=end)  = {np.max(rho_nqs_phys_tx[-1]):.6f}")

if hasattr(col_evo, "energy") and col_evo.energy is not None:
    print(f"E(t=0)               = {col_evo.energy[0]:.6f}")
    print(f"E(t=end)             = {col_evo.energy[-1]:.6f}")

# ==================================================
# 7) NQS evolution plots
# ==================================================
plt.figure(figsize=(8, 5))
plt.pcolormesh(
    col_evo.t_grid,
    x_common,
    rho_nqs_phys_tx.T,
    shading="auto",
)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.title(r"NQS evolution: $|\psi(x,t)|^2$")
plt.colorbar(label=r"$|\psi|^2$")
plt.tight_layout()
plt.show()

idx_list_nqs = [
    0,
    len(col_evo.t_grid) // 4,
    len(col_evo.t_grid) // 2,
    -1,
]

plt.figure(figsize=(8, 5))

for idx in idx_list_nqs:
    plt.plot(
        x_common,
        rho_nqs_phys_tx[idx],
        label=rf"$t={col_evo.t_grid[idx]:.2f}$",
    )

plt.xlabel(r"$x$")
plt.ylabel(r"$|\psi(x,t)|^2$")
plt.title("NQS density snapshots")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
# %% Exact analytical solution
print("Exact analytical solution")
# ==================================================
# 0) Parameters REUSED from the NQS collision block
# ==================================================

x01 = +x0_sep
x02 = -x0_sep

# Map norms to exact parameters:
# N_exact = 4 * eta
eta1 = N_left / 4.0
eta2 = N_right / 4.0

# Use the same physical kicks as the NQS block
k1 = -k_left
k2 = -k_right

# Same spatial grid as NQS
x_plot = x_common

# Exact time grid
t_min = 0.0
t_max = pm.t_max
n_t_exact = 6401

t_grid_exact = np.linspace(t_min, t_max, n_t_exact)
t_snap = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0])

progress_chunk = 50

# ==================================================
# 1) Automatic phase calibration and ABC
# ==================================================

exact_data = exact_nls.prepare_two_soliton_collision_abc(
    x=x_plot,
    eta1=eta1,
    eta2=eta2,
    k1=k1,
    k2=k2,
    x01=x01,
    x02=x02,
    phi_desired=phi_desired,
    calibration_window=2.0,
)

A = exact_data["A"]
B = exact_data["B"]
C = exact_data["C"]

lam1 = exact_data["lam1"]
lam2 = exact_data["lam2"]

phi_offset = exact_data["phi_offset"]
phi_input = exact_data["phi_input"]

print("\n--- Exact analytical parameters ---")
print(f"eta1                = {eta1:.6f}")
print(f"eta2                = {eta2:.6f}")
print(f"k1                  = {k1:.6f}")
print(f"k2                  = {k2:.6f}")
print(f"x01                 = {x01:.6f}")
print(f"x02                 = {x02:.6f}")
print(f"phi_desired         = {phi_desired:.6f}")
print(f"phi_offset          = {phi_offset:.6f}")
print(f"phi_input           = {phi_input:.6f}")
print(f"lambda_1            = {lam1}")
print(f"lambda_2            = {lam2}")

# ==================================================
# 2) Initial exact state on the SAME spatial grid
# ==================================================

psi_exact_t0, dbg = exact_nls.exact_nls_abc_general(
    x_plot,
    0.0,
    A,
    B,
    C,
    check_invertibility=True,
    return_debug=True,
)

rho_exact_t0 = exact_nls.density_from_psi(psi_exact_t0)
phase_exact_t0 = np.unwrap(np.angle(psi_exact_t0))
norm_exact_t0 = exact_nls.norm_from_density(rho_exact_t0, x_plot)

print("\n--- Exact initial state ---")
print(f"norm_exact_t0       = {norm_exact_t0:.12f}")

# ==================================================
# 3) Exact time evolution on the SAME spatial grid
# ==================================================

print("\nBuilding exact time evolution...")

psi_exact_tx = exact_nls.exact_nls_abc_general_on_time_grid_scaled(
    x=x_plot,
    t_grid=t_grid_exact,
    A=A,
    B=B,
    C=C,
    time_scale=0.5,
    check_invertibility=True,
    progress=True,
    progress_chunk=progress_chunk,
)

rho_exact_tx = exact_nls.density_from_psi(psi_exact_tx)

norm_exact_t = exact_nls.norms_on_time_grid(
    rho_exact_tx,
    x_plot,
)

print("\n--- Exact evolution diagnostics ---")
print(f"rho_exact_tx shape  = {rho_exact_tx.shape}")
print(f"norm min            = {np.min(norm_exact_t):.12f}")
print(f"norm max            = {np.max(norm_exact_t):.12f}")
print(f"norm drift          = {np.max(np.abs(norm_exact_t - norm_exact_t[0])):.3e}")

# ==================================================
# 4) Exact plots
# ==================================================

plt.figure(figsize=(8, 5))
plt.plot(
    x_plot,
    rho_exact_t0,
    linewidth=2.0,
)
plt.xlabel(r"$x$")
plt.ylabel(r"$|u(x,0)|^2$")
plt.title("Exact analytical initial density")
plt.grid(alpha=0.2)
plt.show()

plt.figure(figsize=(8, 5))
plt.pcolormesh(
    t_grid_exact,
    x_plot,
    rho_exact_tx.T,
    shading="auto",
)
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.title(r"Exact analytical evolution: $|u(x,t)|^2$")
plt.colorbar(label=r"$|u|^2$")
plt.tight_layout()
plt.show()

# %% Comparison: NQS vs exact
print('Comparison: NQS vs exact')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# ==================================================
# 0) Same spatial grid by construction, interpolate only in time
# ==================================================
t_nqs = np.asarray(col_evo.t_grid, dtype=float)
rho_nqs = np.asarray(rho_nqs_phys_tx, dtype=float)

t_exact = np.asarray(t_grid_exact, dtype=float)
rho_exact = np.asarray(rho_exact_tx, dtype=float)

print("\n--- Shapes before comparison ---")
print("x_common:", x_common.shape)
print("t_nqs:", t_nqs.shape)
print("t_exact:", t_exact.shape)
print("rho_nqs:", rho_nqs.shape)
print("rho_exact:", rho_exact.shape)

t_min_common = max(t_nqs.min(), t_exact.min())
t_max_common = min(t_nqs.max(), t_exact.max())

mask_exact = (t_exact >= t_min_common) & (t_exact <= t_max_common)
t_common = t_exact[mask_exact]
rho_exact_common = rho_exact[mask_exact]

rho_nqs_common = np.zeros((len(t_common), len(x_common)))
for ix in range(len(x_common)):
    rho_nqs_common[:, ix] = np.interp(
        t_common,
        t_nqs,
        rho_nqs[:, ix]
    )

# ==================================================
# 1) Normalize for shape comparison
# ==================================================
def normalize_density(rho, x):
    norm = np.trapezoid(rho, x, axis=1)
    rho_norm = rho / norm[:, None]
    return rho_norm, norm

rho_exact_shape, norm_exact_cmp = normalize_density(rho_exact_common, x_common)
rho_nqs_shape, norm_nqs_cmp = normalize_density(rho_nqs_common, x_common)

# ==================================================
# 2) Errors
# ==================================================
diff_raw = rho_nqs_common - rho_exact_common
diff_shape = rho_nqs_shape - rho_exact_shape

err_l2_raw_t = np.sqrt(np.trapezoid(diff_raw**2, x_common, axis=1))
err_l2_shape_t = np.sqrt(np.trapezoid(diff_shape**2, x_common, axis=1))

denom_shape_t = np.sqrt(np.trapezoid(rho_exact_shape**2, x_common, axis=1))
err_rel_shape_t = err_l2_shape_t / denom_shape_t

print("\n--- Comparison metrics ---")
print(f"Mean raw L2 density error     = {np.mean(err_l2_raw_t):.6e}")
print(f"Max raw L2 density error      = {np.max(err_l2_raw_t):.6e}")
print(f"Final raw L2 density error    = {err_l2_raw_t[-1]:.6e}")
print(f"Mean shape L2 density error   = {np.mean(err_l2_shape_t):.6e}")
print(f"Max shape L2 density error    = {np.max(err_l2_shape_t):.6e}")
print(f"Final shape L2 density error  = {err_l2_shape_t[-1]:.6e}")
print(f"Mean relative shape error     = {np.mean(err_rel_shape_t):.6e}")
print(f"Max relative shape error      = {np.max(err_rel_shape_t):.6e}")
print(f"Final relative shape error    = {err_rel_shape_t[-1]:.6e}")

# ==================================================
# 3) Symmetry diagnostics
# ==================================================
def center_of_mass(rho, x):
    norm = np.trapezoid(rho, x, axis=1)
    return np.trapezoid(rho * x[None, :], x, axis=1) / norm

def density_asymmetry(rho, x):
    rho_flip = rho[:, ::-1]
    return np.trapezoid((rho - rho_flip)**2, x, axis=1)

xcm_exact = center_of_mass(rho_exact_shape, x_common)
xcm_nqs = center_of_mass(rho_nqs_shape, x_common)


print("\n--- Symmetry diagnostics ---")
print(f"max |<x>| exact      = {np.max(np.abs(xcm_exact)):.6e}")
print(f"max |<x>| NQS        = {np.max(np.abs(xcm_nqs)):.6e}")
print(f"final <x> exact      = {xcm_exact[-1]:.6e}")
print(f"final <x> NQS        = {xcm_nqs[-1]:.6e}")


# ==================================================
# 4) Peak trajectories
# ==================================================
def detect_two_main_peaks(rho, x):
    peaks, props = find_peaks(
        rho,
        height=0.08 * np.max(rho),
        distance=max(10, len(x)//30)
    )
    if len(peaks) < 2:
        return None
    heights = props["peak_heights"]
    idx_sorted = np.argsort(heights)[::-1][:2]
    main_peaks = peaks[idx_sorted]
    order = np.argsort(x[main_peaks])
    return main_peaks[order]

x_left_exact = np.full(len(t_common), np.nan)
x_right_exact = np.full(len(t_common), np.nan)
x_left_nqs = np.full(len(t_common), np.nan)
x_right_nqs = np.full(len(t_common), np.nan)

rho_left_exact = np.full(len(t_common), np.nan)
rho_right_exact = np.full(len(t_common), np.nan)
rho_left_nqs = np.full(len(t_common), np.nan)
rho_right_nqs = np.full(len(t_common), np.nan)

for it in range(len(t_common)):
    p_ex = detect_two_main_peaks(rho_exact_shape[it], x_common)
    p_nq = detect_two_main_peaks(rho_nqs_shape[it], x_common)

    if p_ex is not None:
        x_left_exact[it] = x_common[p_ex[0]]
        x_right_exact[it] = x_common[p_ex[1]]
        rho_left_exact[it] = rho_exact_shape[it, p_ex[0]]
        rho_right_exact[it] = rho_exact_shape[it, p_ex[1]]

    if p_nq is not None:
        x_left_nqs[it] = x_common[p_nq[0]]
        x_right_nqs[it] = x_common[p_nq[1]]
        rho_left_nqs[it] = rho_nqs_shape[it, p_nq[0]]
        rho_right_nqs[it] = rho_nqs_shape[it, p_nq[1]]

# ==================================================
# 5) Output directory
# ==================================================
out_dir = Path(pm.figs_dir) / (
    f"comparison_NQS_exact"
    f"_NL_{N_left}_NR_{N_right}"
    f"_kL_{k_left}_kR_{k_right}"
    f"_phi_{phi_desired}"
)
out_dir.mkdir(parents=True, exist_ok=True)
# ==================================================
# Combined poster figure:
# Top    = physical NQS density evolution
# Bottom = physical NQS - exact density difference
# ==================================================


x_plot_1d = np.asarray(x_common, dtype=float).squeeze()
t_plot_1d = np.asarray(t_common, dtype=float).squeeze()

# Physical densities
rho_nqs_plot = np.asarray(rho_nqs_common, dtype=float)
rho_diff_plot = np.asarray(diff_raw, dtype=float)

if rho_nqs_plot.shape != (len(t_plot_1d), len(x_plot_1d)):
    raise ValueError(
        f"rho_nqs_plot has shape {rho_nqs_plot.shape}, "
        f"but expected {(len(t_plot_1d), len(x_plot_1d))}"
    )

if rho_diff_plot.shape != (len(t_plot_1d), len(x_plot_1d)):
    raise ValueError(
        f"rho_diff_plot has shape {rho_diff_plot.shape}, "
        f"but expected {(len(t_plot_1d), len(x_plot_1d))}"
    )

vmax_diff = np.nanmax(np.abs(rho_diff_plot))
if not np.isfinite(vmax_diff) or vmax_diff == 0:
    vmax_diff = 1.0

norm_diff = TwoSlopeNorm(
    vmin=-vmax_diff,
    vcenter=0.0,
    vmax=vmax_diff
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 9),
    sharex=True,
    constrained_layout=True
)

im0 = axes[0].pcolormesh(
    t_plot_1d,
    x_plot_1d,
    rho_nqs_plot.T,
    shading="auto"
)

axes[0].set_ylabel("x")
axes[0].set_title(r"NQS physical density evolution: $|\psi_{\mathrm{NQS}}(x,t)|^2$")

cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label(r"$|\psi_{\mathrm{NQS}}|^2$")

im1 = axes[1].pcolormesh(
    t_plot_1d,
    x_plot_1d,
    rho_diff_plot.T,
    shading="auto",
    cmap="seismic",
    norm=norm_diff
)

axes[1].set_xlabel("t")
axes[1].set_ylabel("x")
axes[1].set_title(r"Physical density difference: $\rho_{\mathrm{NQS}}-\rho_{\mathrm{exact}}$")

cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label(r"$\rho_{\mathrm{NQS}}-\rho_{\mathrm{exact}}$")

fig.savefig(out_dir / "poster_NQS_physical_evolution_and_difference.png", dpi=300)
fig.savefig(out_dir / "poster_NQS_physical_evolution_and_difference.pdf")
plt.show()

print("Saved physical combined poster figure in:")
print(out_dir / "poster_NQS_physical_evolution_and_difference.png")
print(out_dir / "poster_NQS_physical_evolution_and_difference.pdf")
# ==================================================
# 6) Plots
# ==================================================
fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
for frac in fractions:
    i = int(frac * (len(t_common) - 1))

    plt.figure(figsize=(8, 4))
    plt.plot(x_common, rho_exact_shape[i], label=f"Exact t={t_common[i]:.2f}")
    plt.plot(x_common, rho_nqs_shape[i], "--", label=f"NQS t={t_common[i]:.2f}")
    plt.xlabel("x")
    plt.ylabel(r"$|\psi|^2$ normalized")
    plt.title(f"Density comparison, t={t_common[i]:.2f}, L2={err_l2_shape_t[i]:.3e}")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"density_snapshot_{frac:.2f}.png", dpi=200)
    plt.show()



plt.figure(figsize=(8, 4))
plt.plot(t_common, x_left_exact, label="Exact left peak")
plt.plot(t_common, x_left_nqs, "--", label="NQS left peak")
plt.plot(t_common, x_right_exact, label="Exact right peak")
plt.plot(t_common, x_right_nqs, "--", label="NQS right peak")
plt.xlabel("t")
plt.ylabel("peak position")
plt.title("Peak trajectory comparison")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "peak_trajectories_comparison.png", dpi=200)
plt.show()



print("\nPlots saved in:", out_dir)
pm.target_norm = 1.0
# %%