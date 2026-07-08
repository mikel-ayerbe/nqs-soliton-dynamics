# # 1D Quantum Harmonic Oscillator

# %% import PyTorch
import torch

# impor numeric libraries
import numpy as np
import pandas as pd
# import Custom Modules
import parameters as pm
import utilities as utils
import plots

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
# Only needed for 3D plots
from mpl_toolkits.mplot3d import Axes3D

# import Customs Classes
import models
from analysis import Dynamics

# improt Custom Functions
from integrators import integrator
import benchmarks
import exact_nls

import random
import os

from scipy.optimize import curve_fit
#%% Default type
torch.set_default_dtype(torch.float64)
#%% General parameters --------------------------------------------------

# Hardware (CPU or GPU)
dev = 'cpu' # can be changed to 'cuda' for GPU usage
device = torch.device(dev)

# Seed of the random number generator
seed = 1                                       
torch.manual_seed(seed)

# Model ID
file_name = lambda *args: "_".join(args)

# Create a spacial grid object
grid = utils.PointGrid(N=2**9+1, start=-24, end=24, device=device)
# sample from a distribution
# grid.sampler(lambda _: 1.0)

# Choose model
pm.architecture = 'NQS'

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
pm.x0 = -0
pm.w = 1

pm.gauss_amplitude = 0
pm.gauss_width = 8.0
pm.gauss_x0 = 0.0

pm.wall = 0


pm.mu = 1

# Time parameters
pm.dt = 0.1
pm.t_max = 150

# Integrator parameters
pm.evolution = 'imag'
pm.lambda_reg = (1) * 1e-3
pm.e_error = 1e-8

den_list = []
g_list = []
w_list = []
psi_list = []
mesh_list = []

for pm.g, pm.w, pm.mu in zip([1], [0.1], [1]):    

    # Perform imag time evolution
    file_path = utils.file_ID(pm.data_dir,
                            file_name(pm.architecture, model.architecture, pm.evolution) + f'g_{pm.g}',
                            pm.data_format)
    integrator(model, grid, file_path=file_path)

    # Get the dynamics
    imag_evo = Dynamics(file_path=file_path, x_grid=grid.x())
    # Compute density
    psi = imag_evo.psi
    den = np.abs(imag_evo.psi)**2
    # Get parameters
    params = imag_evo.get_params()

    # Plot data
    fig_path = utils.file_ID(pm.figs_dir,
                            file_name(pm.architecture, model.architecture, pm.evolution) + f'_g_{pm.g}',
                            pm.fig_format)
    plots.evo_fig_params_poster(imag_evo.t_grid, grid.mesh, den.T, params, fig_path=fig_path)
    # Plot norm
    norm_fig_path = utils.file_ID(
        pm.figs_dir,
        file_name(pm.architecture, model.architecture, pm.evolution) +
        f'_norm_g_{pm.g}_w_{pm.w}_A_{pm.gauss_amplitude}',
        pm.fig_format
    )

    plots.norm_fig(
        imag_evo.t_grid,
        imag_evo.norm,
        fig_path=norm_fig_path,
        title=rf'$g={pm.g},\ \omega={pm.w},\ A={pm.gauss_amplitude}$'
    )
    g_list.append(pm.g)
    den_list.append(den)
    psi_list.append(psi)
    mesh_list.append(grid.mesh)
    w_list.append(pm.w)
# guardar último ground state
gs_model_path = utils.file_ID(
    pm.data_dir,
    file_name(pm.architecture, model.architecture, 'ground_state') + f'_g_{pm.g}_w_{pm.w}',
    'pt'
)
torch.save(model.state_dict(), gs_model_path)
print(f'GS guardado en: {gs_model_path}')
#%%
import matplotlib.pyplot as plt 
for g, den, mesh, w in zip(g_list, den_list, mesh_list, w_list):
    norm = np.trapz(den[-1,:], mesh)
    plt.plot(mesh, den[-1,:], label=f'g={str(g)}, w={str(w)}')
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
print('Dark soliton via tanh-seeded target + NQS fit')

# --------------------------------------------------
# load GS
# --------------------------------------------------
model_gs = models.NQS(input_size, output_size, hidden_layers).to(device)
model_gs.load_state_dict(torch.load(gs_model_path, map_location=device))
model_gs.to(device)
model_gs.eval()


pm.use_phase_step = False
pm.phase = 0.0
pm.enforce_even_parity = False
pm.enforce_odd_parity = False
pm.k = 0.0

x_tensor = grid.x()
x_int = x_tensor.squeeze()
x = grid.mesh.detach().cpu().numpy() if torch.is_tensor(grid.mesh) else np.array(grid.mesh)

# --------------------------------------------------
# evaluate GS on the grid
# --------------------------------------------------
with torch.no_grad():
    psi_gs_torch = torch.exp(model_gs(x_tensor)).squeeze()

psi_gs = psi_gs_torch.detach().cpu().numpy()

# --------------------------------------------------
# psi_target = psi_gs * psi_D
# --------------------------------------------------
x0_dark = 1.0
ell_dark = 1.0

dark_factor = np.tanh((x - x0_dark) / ell_dark)
psi_target = psi_gs * dark_factor


model_dark_fit = models.NQS(input_size, output_size, hidden_layers).to(device)
model_dark_fit.load_state_dict(torch.load(gs_model_path, map_location=device))
model_dark_fit.to(device)
model_dark_fit.train()

psi_target_torch = torch.tensor(psi_target, dtype=torch.complex128, device=device)

# --------------------------------------------------
# transfer learning
# --------------------------------------------------
optimizer = torch.optim.Adam(model_dark_fit.parameters(), lr=1e-3)

n_epochs = 40000
print_every = 5000

loss_history = []

for epoch in range(n_epochs):
    optimizer.zero_grad()

    psi_pred = torch.exp(model_dark_fit(x_tensor)).squeeze()

    loss = torch.trapz(
        torch.abs(psi_pred - psi_target_torch)**2,
        x_int
    ).real

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % print_every == 0:
        print(f'Epoch {epoch:6d} | loss = {loss.item():.6e}')

print(f'Final fit loss = {loss_history[-1]:.6e}')

model_dark_fit.eval()

# --------------------------------------------------
# save adjusted model
# --------------------------------------------------
dark_fit_model_path = utils.file_ID(
    pm.data_dir,
    file_name(pm.architecture, model_dark_fit.architecture, 'dark_tanh_fit')
    + f'_g_{pm.g}_w_{pm.w}',
    'pt'
)
torch.save(model_dark_fit.state_dict(), dark_fit_model_path)
print(f'Dark tanh-fitted model saved to: {dark_fit_model_path}')

# %%
with torch.no_grad():
    psi_dark_fit_torch = torch.exp(model_dark_fit(x_tensor)).squeeze()

psi_dark_fit = psi_dark_fit_torch.detach().cpu().numpy()

den_gs = np.abs(psi_gs)**2
den_target = np.abs(psi_target)**2
den_fit = np.abs(psi_dark_fit)**2

phase_target = np.unwrap(np.angle(psi_target))
phase_fit = np.unwrap(np.angle(psi_dark_fit))

# --------------------------------------------------
# Summary figure for dark-soliton preparation
# --------------------------------------------------
dark_prep_fig_path = utils.file_ID(
    pm.figs_dir,
    file_name(
        pm.architecture,
        model_dark_fit.architecture,
        'dark_soliton_preparation_summary'
    ) + f'_g_{pm.g}_w_{pm.w}',
    pm.fig_format
)

plots.dark_soliton_preparation_summary_fig(
    x=x,
    den_gs=den_gs,
    den_target=den_target,
    den_fit=den_fit,
    time_gs=imag_evo.t_grid,
    energy_gs=imag_evo.energy,
    x0_dark=x0_dark,
    fig_path=dark_prep_fig_path,
    title=rf'$g={pm.g},\ \omega={pm.w}$',
    xlim=(-20, 20)
)

# 1) Gs density vs target vs fit
plt.figure(figsize=(8,5))
plt.plot(x, den_gs, label='GS density')
plt.plot(x, den_target, '--', label='dark target density')
plt.plot(x, den_fit, ':', label='fitted NQS density')
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.title('Ground state and dark-tanh target')
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# 2) phase target vs fit
plt.figure(figsize=(8,5))
plt.plot(x, phase_target, label='target phase')
plt.plot(x, phase_fit, '--', label='fitted NQS phase')
plt.axvline(0.0, color='k', linestyle=':', label='x=0')
plt.xlabel('x')
plt.ylabel(r'$\arg[\psi(x)]$')
plt.title('Dark-tanh target: phase profile')
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# 3) target wavefunction
plt.figure(figsize=(8,5))
plt.plot(x, psi_target.real, label='Re(target)')
plt.plot(x, psi_target.imag, '--', label='Im(target)')
plt.axvline(0.0, color='k', linestyle=':')
plt.xlabel('x')
plt.ylabel(r'$\psi_{\rm target}(x)$')
plt.title('Target wavefunction with tanh seed')
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# 4) fitted wavefunction
plt.figure(figsize=(8,5))
plt.plot(x, psi_dark_fit.real, label='Re(fit)')
plt.plot(x, psi_dark_fit.imag, '--', label='Im(fit)')
plt.axvline(0.0, color='k', linestyle=':')
plt.xlabel('x')
plt.ylabel(r'$\psi_{\rm fit}(x)$')
plt.title('Fitted NQS dark wavefunction')
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# 5) Loss
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('L2 loss')
plt.title('Dark target fit loss history')
plt.grid(alpha=0.2)
plt.show()

# 6) zoom 
plt.figure(figsize=(8, 5))

mask_center = np.abs(x - x0_dark) < 10.0

plt.plot(
    x[mask_center],
    den_fit[mask_center],
    linewidth=2.8,
    label='NQS black soliton'
)

plt.axvline(
    x0_dark,
    color='k',
    linestyle=':',
    linewidth=1.6,
    label=rf'$x_0={x0_dark:.1f}$'
)

plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$|\psi(x)|^2$', fontsize=18)
plt.title('Black soliton from transfer learning', fontsize=20)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.grid(alpha=0.25)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
# %%
print("Real-time evolution from fitted dark-tanh state")

from scipy.optimize import curve_fit

# ==================================================
# Load fitted model
# ==================================================
model_dyn = models.NQS(input_size, output_size, hidden_layers).to(device)
model_dyn.load_state_dict(torch.load(dark_fit_model_path, map_location=device))
model_dyn.to(device)
model_dyn.eval()

# ==================================================
# Physical parameters
# ==================================================
pm.g = 1.0
pm.x0 = 0.0
pm.w = 0.1
pm.mu = 1.0

pm.wall = 0.0

pm.gauss_amplitude = 0.0
pm.gauss_width = 1.0
pm.gauss_x0 = 0.0

pm.k = 0.0

pm.phase = 0.0
pm.phase_center = 0.0
pm.phase_width = 1.0
pm.use_phase_step = False

pm.enforce_even_parity = False
pm.enforce_odd_parity = False

# ==================================================
# Real-time evolution
# ==================================================
pm.dt = 0.01
pm.t_max = 50.0

pm.evolution = "real"
pm.lambda_reg = 1j * 1e-2

file_path_real = utils.file_ID(
    pm.data_dir,
    file_name(
        pm.architecture,
        model_dyn.architecture,
        "real_from_dark_tanh_fit",
    )
    + f"_g_{pm.g}_w_{pm.w}",
    pm.data_format,
)

integrator(
    model_dyn,
    grid,
    file_path=file_path_real,
)

# ==================================================
# Load dynamics
# ==================================================
real_evo = Dynamics(
    file_path=file_path_real,
    x_grid=grid.x(),
)

psi = real_evo.psi
den = np.abs(psi) ** 2
params = real_evo.get_params()

x = (
    grid.mesh.detach().cpu().numpy()
    if torch.is_tensor(grid.mesh)
    else np.asarray(grid.mesh)
)

t = np.asarray(real_evo.t_grid)

# ==================================================
# Full evolution figure
# ==================================================
fig_path = utils.file_ID(
    pm.figs_dir,
    file_name(
        pm.architecture,
        model_dyn.architecture,
        "real_from_dark_tanh_fit",
    )
    + f"_g_{pm.g}_w_{pm.w}",
    pm.fig_format,
)

plots.evo_fig_params_poster(
    real_evo.t_grid,
    grid.mesh,
    den.T,
    params,
    fig_path=fig_path,
)

# ==================================================
# Track dark-soliton minimum
# ==================================================
xmin_list = []
rhomin_list = []

x_prev = x0_dark
track_halfwidth = 4.0

for k in range(len(t)):
    denk = den[k, :]

    mask_track = np.abs(x - x_prev) < track_halfwidth
    idxs = np.where(mask_track)[0]

    if len(idxs) == 0:
        raise RuntimeError(
            f"No tracking points found at time index {k}. "
            f"Try increasing track_halfwidth."
        )

    den_loc = denk[idxs]
    i_loc = np.argmin(den_loc)
    i_glob = idxs[i_loc]

    # Parabolic refinement using the minimum and its two neighbors
    if 0 < i_glob < len(x) - 1:
        x1, x2, x3 = x[i_glob - 1], x[i_glob], x[i_glob + 1]
        y1, y2, y3 = denk[i_glob - 1], denk[i_glob], denk[i_glob + 1]

        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)

        if np.abs(denom) > 1e-14:
            A_par = (
                x3 * (y2 - y1)
                + x2 * (y1 - y3)
                + x1 * (y3 - y2)
            ) / denom

            B_par = (
                x3**2 * (y1 - y2)
                + x2**2 * (y3 - y1)
                + x1**2 * (y2 - y3)
            ) / denom

            C_par = y1 - A_par * x1**2 - B_par * x1

            if np.abs(A_par) > 1e-14:
                x_min_k = -B_par / (2.0 * A_par)
                rho_min_k = A_par * x_min_k**2 + B_par * x_min_k + C_par
            else:
                x_min_k = x[i_glob]
                rho_min_k = denk[i_glob]
        else:
            x_min_k = x[i_glob]
            rho_min_k = denk[i_glob]
    else:
        x_min_k = x[i_glob]
        rho_min_k = denk[i_glob]

    xmin_list.append(x_min_k)
    rhomin_list.append(rho_min_k)

    x_prev = x_min_k

xmin_arr = np.asarray(xmin_list)
rhomin_arr = np.asarray(rhomin_list)

# ==================================================
# Sinusoidal fit
# x_min(t) = x_c + A cos(omega t + phi)
# ==================================================
def xfit_func(t, x_c, A, omega, phi):
    return x_c + A * np.cos(omega * t + phi)


t_fit_min = 55.0
t_fit_max = 295.0

mask_fit_time = (t >= t_fit_min) & (t <= t_fit_max)

t_fit = t[mask_fit_time]
x_fit_data = xmin_arr[mask_fit_time]

if len(t_fit) < 10:
    raise RuntimeError(
        "Not enough points in the fit window. "
        "Adjust t_fit_min and t_fit_max."
    )

x_c0 = np.mean(x_fit_data)
A0 = 0.5 * (np.max(x_fit_data) - np.min(x_fit_data))
omega0 = pm.w / np.sqrt(2.0)
phi0 = 0.0

p0 = [x_c0, A0, omega0, phi0]

popt, pcov = curve_fit(
    xfit_func,
    t_fit,
    x_fit_data,
    p0=p0,
    maxfev=20000,
)

x_c_fit, A_fit, omega_fit, phi_fit = popt
perr = np.sqrt(np.diag(pcov))

omega_th = pm.w / np.sqrt(2.0)

print("\n=== Sinusoidal fit of x_min(t) ===")
print(f"x_c_fit   = {x_c_fit:.6e}")
print(f"A_fit     = {A_fit:.6e}")
print(f"omega_fit = {omega_fit:.6e}")
print(f"phi_fit   = {phi_fit:.6e}")
print(f"omega_th  = {omega_th:.6e}")
print(f"relative error omega = {abs(omega_fit - omega_th) / abs(omega_th):.6e}")

print("\n=== Fit uncertainties ===")
print(f"sigma_x_c   = {perr[0]:.6e}")
print(f"sigma_A     = {perr[1]:.6e}")
print(f"sigma_omega = {perr[2]:.6e}")
print(f"sigma_phi   = {perr[3]:.6e}")

T_fit = 2.0 * np.pi / omega_fit
T_th = 2.0 * np.pi / omega_th

print("\n=== Oscillation period ===")
print(f"T_fit = {T_fit:.6e}")
print(f"T_th  = {T_th:.6e}")
print(f"relative error T = {abs(T_fit - T_th) / abs(T_th):.6e}")

# ==================================================
# Plot sinusoidal fit
# ==================================================
x_ref = xfit_func(
    t,
    x_c_fit,
    A_fit,
    omega_fit,
    phi_fit,
)

x_fit_ref = xfit_func(
    t_fit,
    x_c_fit,
    A_fit,
    omega_fit,
    phi_fit,
)

plt.figure(figsize=(8, 5))
plt.plot(
    t,
    xmin_arr,
    label=r"$x_{\min}(t)$",
)
plt.plot(
    t_fit,
    x_fit_ref,
    "--",
    label="sinusoidal fit",
)
plt.xlabel(r"$t$")
plt.ylabel(r"$x_{\min}(t)$")
plt.title("Dark-soliton trajectory and sinusoidal fit")
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# ==================================================
# Norm conservation
# ==================================================
norm_t = np.array([
    np.trapezoid(den[k, :], x)
    for k in range(len(t))
])

plt.figure(figsize=(8, 5))
plt.plot(
    t,
    norm_t,
)
plt.xlabel(r"$t$")
plt.ylabel("Norm")
plt.title("Norm conservation")
plt.grid(alpha=0.2)
plt.show()

print("\n=== Norm conservation ===")
print(f"norm initial    = {norm_t[0]:.6e}")
print(f"norm final      = {norm_t[-1]:.6e}")
print(f"relative drift  = {abs(norm_t[-1] - norm_t[0]) / abs(norm_t[0]):.6e}")

# ==================================================
# Summary figure: dark-soliton density + parameter evolution
# ==================================================
dark_dyn_fig_path = utils.file_ID(
    pm.figs_dir,
    file_name(
        pm.architecture,
        model_dyn.architecture,
        "dark_soliton_density_params",
    )
    + f"_g_{pm.g}_w_{pm.w}",
    pm.fig_format,
)

plots.dark_soliton_density_params_side_by_side(
    time=t,
    mesh=x,
    den=den,
    params=params,
    x_ref=x_ref,
    fig_path=dark_dyn_fig_path,
    title=rf"$g={pm.g},\ \omega={pm.w}$",
)

# %%
print("Dark-dark collision: NQS vs analytical solution")

# ==================================================
# Single-seed setup
# ==================================================
NQS_SEED = 12

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
# NQS grid for simplified collision
# ==================================================
grid_scalar = utils.PointGrid(
    N=2**9 + 1,
    start=-45.0,
    end=45.0,
    device=device,
)

x_tensor = grid_scalar.x()

x_nqs = (
    grid_scalar.mesh.detach().cpu().numpy()
    if torch.is_tensor(grid_scalar.mesh)
    else np.asarray(grid_scalar.mesh)
)

# ==================================================
# Symmetric scalar analytical parameters
# ==================================================
kappa_abs = 0.60
x_sep = 5.0

darkdark_params = exact_nls.build_symmetric_dark_dark_parameters(
    kappa_abs=kappa_abs,
    x_sep=x_sep,
    q_plus_scalar=np.array([1.25 + 0.0j, 0.0 + 0.0j], dtype=complex),
)

q_plus_scalar = darkdark_params["q_plus_scalar"]
q0_scalar = darkdark_params["q0_scalar"]

kappa1 = darkdark_params["kappa1"]
kappa2 = darkdark_params["kappa2"]

nu1_scalar = darkdark_params["nu1_scalar"]
nu2_scalar = darkdark_params["nu2_scalar"]

zeta1_scalar = darkdark_params["zeta1_scalar"]
zeta2_scalar = darkdark_params["zeta2_scalar"]

gamma1 = darkdark_params["gamma1"]
gamma2 = darkdark_params["gamma2"]

x1_initial = darkdark_params["x1_initial"]
x2_initial = darkdark_params["x2_initial"]

t_collision_est = darkdark_params["t_collision_est"]

print("\n── Simplified symmetric collision parameters ──")
print(f"q0_scalar       = {q0_scalar:.8f}")
print(f"background rho  = {q0_scalar**2:.8f}")
print(f"kappa1          = {kappa1:.8f}")
print(f"kappa2          = {kappa2:.8f}")
print(f"nu1             = {nu1_scalar:.8f}")
print(f"nu2             = {nu2_scalar:.8f}")
print(f"zeta1           = {zeta1_scalar:.8f}")
print(f"zeta2           = {zeta2_scalar:.8f}")
print(f"gamma1          = {gamma1:.8e}")
print(f"gamma2          = {gamma2:.8e}")
print(f"approx x1(0)    = {x1_initial:.8f}")
print(f"approx x2(0)    = {x2_initial:.8f}")
print(f"v1 = 2*kappa1   = {2.0 * kappa1:.8f}")
print(f"v2 = 2*kappa2   = {2.0 * kappa2:.8f}")
print(f"expected rho_min= {kappa_abs**2:.8f}")

# ==================================================
# Time window
# ==================================================
t_exact_initial = 0.0
t_exact_final = 1.0

# Direct relation:
#     t_exact = t_exact_initial + 0.5 * tau_NQS
tau_max = 2.0 * (t_exact_final - t_exact_initial)

print("\n── Time setup ──")
print(f"t_exact_initial       = {t_exact_initial:.8f}")
print(f"t_exact_final         = {t_exact_final:.8f}")
print(f"estimated collision t = {t_collision_est:.8f}")
print(f"NQS tau_max           = {tau_max:.8f}")

# ==================================================
# Build initial analytical target for NQS
# ==================================================
psi_target_xt, D_target = exact_nls.scalar_dark_dark_solution_on_grid(
    x_nqs,
    np.array([t_exact_initial]),
    q_plus_scalar=q_plus_scalar,
    zeta1_scalar=zeta1_scalar,
    zeta2_scalar=zeta2_scalar,
    gamma1=gamma1,
    gamma2=gamma2,
    q0_scalar=q0_scalar,
)

psi_target = psi_target_xt[:, 0]
rho_target = np.abs(psi_target) ** 2

norm_target = np.trapezoid(rho_target, x_nqs)
pm.target_norm = float(norm_target)

print("\n── Initial analytical target ──")
print(f"t_exact_initial     = {t_exact_initial:.8f}")
print(f"target norm         = {norm_target:.8f}")
print(f"target rho min/max  = {rho_target.min():.8f}, {rho_target.max():.8f}")
print(f"max Im(D)           = {np.max(np.abs(np.imag(D_target))):.3e}")
print(f"min Re(D)           = {np.min(np.real(D_target)):.8e}")

plt.figure(figsize=(8, 5))
plt.plot(
    x_nqs,
    rho_target,
    label=rf"analytical target, $t={t_exact_initial}$",
)
plt.axvline(
    x1_initial,
    linestyle=":",
    color="gray",
    label="initial notch guesses",
)
plt.axvline(
    x2_initial,
    linestyle=":",
    color="gray",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$|\psi_{\rm target}(x)|^2$")
plt.title("Initial symmetric two-dark target for NQS")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    x_nqs,
    psi_target.real,
    label="Re target",
)
plt.plot(
    x_nqs,
    psi_target.imag,
    "--",
    label="Im target",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$\psi_{\rm target}(x)$")
plt.title("Complex initial target")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# ==================================================
# Fit NQS to analytical target
# ==================================================
print("\nFitting NQS to simplified symmetric analytical initial state")

hidden_layers_scalar = [20, 20]

model_scalar_fit = models.NQS(
    input_size,
    output_size,
    hidden_layers_scalar,
).to(device)

model_scalar_fit.train()

psi_target_torch = torch.tensor(
    psi_target,
    dtype=torch.complex128,
    device=device,
)

target_norm_torch = torch.tensor(
    norm_target,
    dtype=torch.float64,
    device=device,
)


def normalize_torch_to_target(psi, x_tensor, target_norm):
    x_real = x_tensor.squeeze().real
    norm = torch.trapz(torch.abs(psi) ** 2, x_real).real

    if norm <= 0 or not torch.isfinite(norm):
        raise ValueError(f"Invalid torch norm during normalization: {norm}")

    return psi / torch.sqrt(norm) * torch.sqrt(target_norm)


# ==================================================
# Stage 1: normalized-shape fit
# ==================================================
optimizer = torch.optim.Adam(
    model_scalar_fit.parameters(),
    lr=1e-3,
)

n_epochs_stage1 = 100000
n_epochs_stage2 = 40000
print_every = 10000

loss_history_scalar = []

print("\n── Stage 1: normalized-shape fit ──")

for epoch in range(n_epochs_stage1):
    optimizer.zero_grad()

    psi_pred_raw = torch.exp(model_scalar_fit(x_tensor)).squeeze()

    psi_pred = normalize_torch_to_target(
        psi_pred_raw,
        x_tensor,
        target_norm_torch,
    )

    loss_complex = torch.mean(torch.abs(psi_pred - psi_target_torch) ** 2)

    loss_density = torch.mean(
        (
            torch.abs(psi_pred) ** 2
            - torch.abs(psi_target_torch) ** 2
        )
        ** 2
    )

    loss = loss_complex + 0.5 * loss_density

    loss.backward()
    optimizer.step()

    loss_history_scalar.append(loss.item())

    if epoch % print_every == 0:
        print(
            f"Stage 1 | Epoch {epoch:6d} | "
            f"loss = {loss.item():.6e} | "
            f"L2 = {loss_complex.item():.6e} | "
            f"density = {loss_density.item():.6e}"
        )

print(f"Final Stage 1 loss = {loss_history_scalar[-1]:.6e}")

# ==================================================
# Stage 2: raw physical-scale fine tuning
# ==================================================
optimizer = torch.optim.Adam(
    model_scalar_fit.parameters(),
    lr=2e-4,
)

print("\n── Stage 2: raw physical-scale fine tuning ──")

for epoch in range(n_epochs_stage2):
    optimizer.zero_grad()

    psi_pred = torch.exp(model_scalar_fit(x_tensor)).squeeze()

    x_real = x_tensor.squeeze().real
    norm_pred = torch.trapz(torch.abs(psi_pred) ** 2, x_real).real

    loss_complex = torch.mean(torch.abs(psi_pred - psi_target_torch) ** 2)

    loss_density = torch.mean(
        (
            torch.abs(psi_pred) ** 2
            - torch.abs(psi_target_torch) ** 2
        )
        ** 2
    )

    loss_norm = ((norm_pred - target_norm_torch) / target_norm_torch) ** 2

    loss = loss_complex + 0.5 * loss_density + 10.0 * loss_norm

    loss.backward()
    optimizer.step()

    loss_history_scalar.append(loss.item())

    if epoch % print_every == 0:
        print(
            f"Stage 2 | Epoch {epoch:6d} | "
            f"loss = {loss.item():.6e} | "
            f"L2 = {loss_complex.item():.6e} | "
            f"density = {loss_density.item():.6e} | "
            f"norm = {norm_pred.item():.8f}"
        )

print(f"Final total fit loss = {loss_history_scalar[-1]:.6e}")

model_scalar_fit.eval()

# ==================================================
# Save fitted model
# ==================================================
scalar_arch_tag = "HL_" + "_".join(str(h) for h in hidden_layers_scalar)

scalar_fit_model_path = utils.file_ID(
    pm.data_dir,
    file_name(
        pm.architecture,
        model_scalar_fit.architecture,
        "simple_symmetric_two_dark_fit",
        scalar_arch_tag,
    )
    + f"_q0_{q0_scalar}_kappa_{kappa_abs}_xsep_{x_sep}",
    "pt",
)

torch.save(
    model_scalar_fit.state_dict(),
    scalar_fit_model_path,
)

print(f"Simplified symmetric fitted model saved to: {scalar_fit_model_path}")

# ==================================================
# Validate raw fit
# ==================================================
with torch.no_grad():
    psi_fit_torch = torch.exp(model_scalar_fit(x_tensor)).squeeze()

psi_fit = psi_fit_torch.detach().cpu().numpy()
rho_fit = np.abs(psi_fit) ** 2

norm_fit = np.trapezoid(rho_fit, x_nqs)

rho_l2 = np.sqrt(
    np.trapezoid(
        (rho_fit - rho_target) ** 2,
        x_nqs,
    )
)

psi_l2 = np.sqrt(
    np.trapezoid(
        np.abs(psi_fit - psi_target) ** 2,
        x_nqs,
    )
)

print("\n── Raw fit diagnostics ──")
print(f"norm target       = {norm_target:.8f}")
print(f"norm fit          = {norm_fit:.8f}")
print(f"L2 density error  = {rho_l2:.6e}")
print(f"L2 psi error      = {psi_l2:.6e}")
print(f"target rho min    = {rho_target.min():.8f}")
print(f"fit rho min       = {rho_fit.min():.8f}")
print(f"target rho max    = {rho_target.max():.8f}")
print(f"fit rho max       = {rho_fit.max():.8f}")

plt.figure(figsize=(8, 5))
plt.plot(
    x_nqs,
    rho_target,
    label="analytical target",
)
plt.plot(
    x_nqs,
    rho_fit,
    "--",
    label="NQS raw fit",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$|\psi|^2$")
plt.title("Initial density: analytical target vs raw NQS fit")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    x_nqs,
    psi_target.real,
    label="Re target",
)
plt.plot(
    x_nqs,
    psi_fit.real,
    "--",
    label="Re fit",
)
plt.plot(
    x_nqs,
    psi_target.imag,
    label="Im target",
)
plt.plot(
    x_nqs,
    psi_fit.imag,
    "--",
    label="Im fit",
)
plt.xlabel(r"$x$")
plt.ylabel(r"$\psi(x)$")
plt.title("Initial wavefunction: analytical target vs raw NQS fit")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(loss_history_scalar)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Simplified symmetric target fit loss")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# ==================================================
# Real-time NQS evolution
# ==================================================
print("\nReal-time NQS evolution from simplified symmetric target")

model_scalar_dyn = models.NQS(
    input_size,
    output_size,
    hidden_layers_scalar,
).to(device)

model_scalar_dyn.load_state_dict(
    torch.load(
        scalar_fit_model_path,
        map_location=device,
    )
)

model_scalar_dyn.to(device)
model_scalar_dyn.eval()

# ==================================================
# Physical parameters: original NQS Hamiltonian
# ==================================================
pm.kinetic_prefactor = -0.5
pm.g = 1.0
pm.w = 0.0
pm.x0 = 0.0
pm.mu = 2.0 * q0_scalar**2

pm.target_norm = float(norm_target)

pm.wall = 0.0
pm.gauss_amplitude = 0.0
pm.gauss_width = 1.0
pm.gauss_x0 = 0.0

pm.k = 0.0
pm.phase = 0.0
pm.phase_center = 0.0
pm.phase_width = 1.0
pm.use_phase_step = False

pm.enforce_even_parity = False
pm.enforce_odd_parity = False

pm.evolution = "real"
pm.integrator = "RK4"

pm.dt = 0.01
pm.t_max = tau_max
pm.t_size = 900

pm.lambda_reg = 1e-3
pm.pinv_rtol = 1e-7
pm.max_param_step = 0.02

pm.stopper = False

# ==================================================
# Preflight check
# ==================================================
with torch.no_grad():
    lnpsi_check = model_scalar_dyn(x_tensor)
    psi_check = torch.exp(lnpsi_check)

rho_check = torch.abs(psi_check.squeeze()) ** 2
norm_check = torch.trapz(
    rho_check,
    x_tensor.squeeze().real,
).real

print("\n── Preflight scalar NQS check ──")
print(f"finite lnpsi = {torch.isfinite(lnpsi_check).all().item()}")
print(f"finite psi   = {torch.isfinite(psi_check).all().item()}")
print(f"lnpsi real min/max = {lnpsi_check.real.min().item():.6e}, {lnpsi_check.real.max().item():.6e}")
print(f"lnpsi imag min/max = {lnpsi_check.imag.min().item():.6e}, {lnpsi_check.imag.max().item():.6e}")
print(f"|psi| min/max       = {torch.abs(psi_check).min().item():.6e}, {torch.abs(psi_check).max().item():.6e}")
print(f"raw NQS norm        = {norm_check.item():.8f}")
print(f"target norm         = {norm_target:.8f}")

if not torch.isfinite(lnpsi_check).all() or not torch.isfinite(psi_check).all():
    raise ValueError("Initial scalar NQS contains non-finite values.")

# ==================================================
# Run integrator
# ==================================================
file_path_scalar_real = utils.file_ID(
    pm.data_dir,
    file_name(
        pm.architecture,
        model_scalar_dyn.architecture,
        "real_simple_symmetric_two_dark",
        scalar_arch_tag,
    )
    + f"_q0_{q0_scalar}_kappa_{kappa_abs}_xsep_{x_sep}_dt_{pm.dt}",
    pm.data_format,
)

integrator(
    model_scalar_dyn,
    grid_scalar,
    file_path=file_path_scalar_real,
)

# ==================================================
# Load dynamics
# ==================================================
scalar_evo = Dynamics(
    file_path=file_path_scalar_real,
    x_grid=grid_scalar.x(),
)

psi_nqs = scalar_evo.psi
rho_nqs = np.abs(psi_nqs) ** 2
tau_nqs = scalar_evo.t_grid

# ==================================================
# Figure directories
# ==================================================
fig_dir_simple = os.path.join(
    pm.figs_dir,
    file_name(
        pm.architecture,
        model_scalar_dyn.architecture,
        "simple_symmetric_two_dark_figs",
        scalar_arch_tag,
    )
    + f"_q0_{q0_scalar}_kappa_{kappa_abs}_xsep_{x_sep}_dt_{pm.dt}",
)

snapshot_dir = os.path.join(fig_dir_simple, "snapshots")

os.makedirs(fig_dir_simple, exist_ok=True)
os.makedirs(snapshot_dir, exist_ok=True)

print(f"Figures will be saved in: {fig_dir_simple}")
print(f"Snapshots will be saved in: {snapshot_dir}")

# ==================================================
# NQS vs analytical comparison using fixed 0.5 scaling
# ==================================================
comparison = exact_nls.compare_density_nqs_vs_exact(
    tau_grid=tau_nqs,
    rho_nqs=rho_nqs,
    x_grid=x_nqs,
    t_exact_initial=t_exact_initial,
    t_exact_final=t_exact_final,
    q_plus_scalar=q_plus_scalar,
    zeta1_scalar=zeta1_scalar,
    zeta2_scalar=zeta2_scalar,
    gamma1=gamma1,
    gamma2=gamma2,
    q0_scalar=q0_scalar,
    inner_window=(-25.0, 25.0),
)

x_cmp = comparison["x"]
tau_cmp = comparison["tau"]
t_exact_cmp = comparison["t_exact"]

rho_nqs_cmp = comparison["rho_nqs"]
rho_exact_cmp = comparison["rho_exact"]
rho_error = comparison["rho_error"]

norm_nqs = comparison["norm_nqs"]
norm_exact = comparison["norm_exact"]

rel_l2_space_time = comparison["rel_l2"]
rel_l2_inner = comparison["rel_l2_inner"]
max_abs_error = comparison["max_abs_error"]
max_abs_inner_error = comparison["max_abs_inner_error"]
norm_rel_drift_nqs = comparison["norm_rel_drift_nqs"]

print("\n── NQS vs analytical density comparison ──")
print("time relation              = t_exact_initial + 0.5 * tau_NQS")
print(f"collision type             = symmetric simplified")
print(f"kappa1                    = {kappa1:.8f}")
print(f"kappa2                    = {kappa2:.8f}")
print(f"x_sep                     = {x_sep:.8f}")
print(f"NQS tau range compared    = [{tau_cmp[0]:.6f}, {tau_cmp[-1]:.6f}]")
print(f"t_exact range compared    = [{t_exact_cmp[0]:.6f}, {t_exact_cmp[-1]:.6f}]")
print(f"relative L2 spacetime err = {rel_l2_space_time:.6e}")
print(f"relative L2 inner error   = {rel_l2_inner:.6e}")
print(f"max absolute density err  = {max_abs_error:.6e}")
print(f"max absolute inner err    = {max_abs_inner_error:.6e}")
print(f"NQS norm initial          = {norm_nqs[0]:.8f}")
print(f"NQS norm final            = {norm_nqs[-1]:.8f}")
print(f"NQS max relative drift    = {norm_rel_drift_nqs:.6e}")
print(f"Exact norm initial        = {norm_exact[0]:.8f}")
print(f"Exact norm final          = {norm_exact[-1]:.8f}")

# ==================================================
# Cropped arrays for visualization only
# ==================================================
x_plot_max = 10

mask_plot = np.abs(x_cmp) <= x_plot_max

x_plot = x_cmp[mask_plot]
rho_exact_plot = rho_exact_cmp[:, mask_plot]
rho_nqs_plot = rho_nqs_cmp[:, mask_plot]
rho_error_plot = rho_error[:, mask_plot]

print("\n── Cropped visualization diagnostics ──")
print(f"x_plot range         = [{x_plot[0]:.6f}, {x_plot[-1]:.6f}]")
print(f"rho_exact_plot shape = {rho_exact_plot.shape}")
print(f"rho_nqs_plot shape   = {rho_nqs_plot.shape}")
print(f"rho_error_plot shape = {rho_error_plot.shape}")

crop_name = f"{x_plot_max:.1f}".replace(".", "d")

# ==================================================
# Density maps: analytical, NQS, and error
# ==================================================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.6),
    sharex=True,
    sharey=True,
)

im0 = axes[0].pcolormesh(
    t_exact_cmp,
    x_plot,
    rho_exact_plot.T,
    shading="auto",
    cmap="viridis",
)
axes[0].set_title(r"Analytical $|\psi_{\rm exact}|^2$")
axes[0].set_xlabel(r"$t$")
axes[0].set_ylabel(r"$x$")
axes[0].set_ylim(-x_plot_max, x_plot_max)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].pcolormesh(
    t_exact_cmp,
    x_plot,
    rho_nqs_plot.T,
    shading="auto",
    cmap="viridis",
)
axes[1].set_title(r"NQS $|\psi_{\rm NQS}|^2$")
axes[1].set_xlabel(r"$t$")
axes[1].set_ylim(-x_plot_max, x_plot_max)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

err_lim = np.max(np.abs(rho_error_plot))

if err_lim == 0 or not np.isfinite(err_lim):
    err_lim = 1e-12

im2 = axes[2].pcolormesh(
    t_exact_cmp,
    x_plot,
    rho_error_plot.T,
    shading="auto",
    cmap="coolwarm",
    vmin=-err_lim,
    vmax=+err_lim,
)
axes[2].set_title(r"NQS $-$ analytical")
axes[2].set_xlabel(r"$t$")
axes[2].set_ylim(-x_plot_max, x_plot_max)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()

density_maps_path_png = os.path.join(
    fig_dir_simple,
    f"density_maps_cropped_xmax_{crop_name}.png",
)

density_maps_path_pdf = os.path.join(
    fig_dir_simple,
    f"density_maps_cropped_xmax_{crop_name}.pdf",
)

plt.savefig(density_maps_path_png, dpi=300, bbox_inches="tight")
plt.savefig(density_maps_path_pdf, bbox_inches="tight")

print(f"Saved cropped density maps to: {density_maps_path_png}")
print(f"Saved cropped density maps to: {density_maps_path_pdf}")

plt.show()

# ==================================================
# Snapshot comparison
# ==================================================
snapshot_exact_times = [
    t_exact_initial,
    0.5 * (t_exact_initial + t_collision_est),
    t_collision_est,
    0.5 * (t_collision_est + t_exact_final),
    t_exact_final,
]

snapshot_exact_times = [
    ts for ts in snapshot_exact_times
    if t_exact_cmp[0] <= ts <= t_exact_cmp[-1]
]

for ts in snapshot_exact_times:
    idx = np.argmin(np.abs(t_exact_cmp - ts))
    t_plot = t_exact_cmp[idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x_plot,
        rho_exact_cmp[idx, mask_plot],
        label=rf"analytical, $t={t_plot:.2f}$",
        linewidth=2.0,
    )

    ax.plot(
        x_plot,
        rho_nqs_cmp[idx, mask_plot],
        "--",
        label="NQS",
        linewidth=2.0,
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title(rf"Density snapshot at $t={t_plot:.2f}$, central view")
    ax.set_xlim(-x_plot_max, x_plot_max)
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()

    t_name = (
        f"{t_plot:+.3f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "d")
    )

    snapshot_path_png = os.path.join(
        snapshot_dir,
        f"snapshot_cropped_xmax_{crop_name}_t_{t_name}.png",
    )

    snapshot_path_pdf = os.path.join(
        snapshot_dir,
        f"snapshot_cropped_xmax_{crop_name}_t_{t_name}.pdf",
    )

    plt.savefig(snapshot_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(snapshot_path_pdf, bbox_inches="tight")

    print(f"Saved cropped snapshot to: {snapshot_path_png}")
    print(f"Saved cropped snapshot to: {snapshot_path_pdf}")

    plt.show()

# ==================================================
# Poster plot: NQS dark-soliton collision density
# ==================================================
fig, ax = plt.subplots(figsize=(6.2, 3.4))

im = ax.pcolormesh(
    t_exact_cmp,
    x_plot,
    rho_nqs_plot.T,
    shading="auto",
    cmap="viridis",
)

ax.set_title(r"NQS dark-soliton collision", fontsize=18)
ax.set_xlabel(r"$t$", fontsize=16)
ax.set_ylabel(r"$x$", fontsize=16)
ax.tick_params(axis="both", labelsize=13)
ax.set_ylim(-x_plot_max, x_plot_max)

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label(r"$|\psi_{\rm NQS}(x,t)|^2$", fontsize=15)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

poster_nqs_path_png = os.path.join(
    fig_dir_simple,
    f"poster_nqs_dark_collision_xmax_{crop_name}.png",
)

poster_nqs_path_pdf = os.path.join(
    fig_dir_simple,
    f"poster_nqs_dark_collision_xmax_{crop_name}.pdf",
)

plt.savefig(poster_nqs_path_png, dpi=300, bbox_inches="tight")
plt.savefig(poster_nqs_path_pdf, bbox_inches="tight")

print(f"Saved poster NQS dark-collision plot to: {poster_nqs_path_png}")
print(f"Saved poster NQS dark-collision plot to: {poster_nqs_path_pdf}")

plt.show()

# ==================================================
# Poster plot: NQS dark collision + density difference
# ==================================================
err_lim = np.nanmax(np.abs(rho_error_plot))

if err_lim == 0 or not np.isfinite(err_lim):
    err_lim = 1e-12

norm_error = TwoSlopeNorm(
    vmin=-err_lim,
    vcenter=0.0,
    vmax=+err_lim,
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.0, 6.4),
    sharex=True,
    constrained_layout=True,
)

im0 = axes[0].pcolormesh(
    t_exact_cmp,
    x_plot,
    rho_nqs_plot.T,
    shading="auto",
    cmap="viridis",
)

axes[0].set_title(r"NQS dark-soliton collision", fontsize=17)
axes[0].set_ylabel(r"$x$", fontsize=15)
axes[0].set_ylim(-x_plot_max, x_plot_max)
axes[0].tick_params(axis="both", labelsize=12)

cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
cbar0.set_label(r"$|\psi_{\rm NQS}(x,t)|^2$", fontsize=13)
cbar0.ax.tick_params(labelsize=11)

im1 = axes[1].pcolormesh(
    t_exact_cmp,
    x_plot,
    rho_error_plot.T,
    shading="auto",
    cmap="coolwarm",
    norm=norm_error,
)

axes[1].set_title(
    r"Density difference: $\rho_{\rm NQS}-\rho_{\rm analytical}$",
    fontsize=17,
)
axes[1].set_xlabel(r"$t$", fontsize=15)
axes[1].set_ylabel(r"$x$", fontsize=15)
axes[1].set_ylim(-x_plot_max, x_plot_max)
axes[1].tick_params(axis="both", labelsize=12)

cbar1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
cbar1.set_label(r"$\rho_{\rm NQS}-\rho_{\rm analytical}$", fontsize=13)
cbar1.ax.tick_params(labelsize=11)

poster_combined_path_png = os.path.join(
    fig_dir_simple,
    f"poster_nqs_dark_collision_plus_difference_xmax_{crop_name}.png",
)

poster_combined_path_pdf = os.path.join(
    fig_dir_simple,
    f"poster_nqs_dark_collision_plus_difference_xmax_{crop_name}.pdf",
)

plt.savefig(poster_combined_path_png, dpi=300, bbox_inches="tight")
plt.savefig(poster_combined_path_pdf, bbox_inches="tight")

print(f"Saved combined dark poster plot to: {poster_combined_path_png}")
print(f"Saved combined dark poster plot to: {poster_combined_path_pdf}")

plt.show()
# %%