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
# Only needed for 3D plots
from mpl_toolkits.mplot3d import Axes3D

# import Customs Classes
import models
from analysis import Dynamics

# improt Custom Functions
from integrators import integrator
import benchmarks

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

for pm.g, pm.w, pm.mu in zip([1], [0.01], [1]):    

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
# cargar el ground state normal
# --------------------------------------------------
model_gs = models.NQS(input_size, output_size, hidden_layers).to(device)
model_gs.load_state_dict(torch.load(gs_model_path, map_location=device))
model_gs.to(device)
model_gs.eval()

# asegurar que no haya phase step ni paridad forzada
pm.use_phase_step = False
pm.phase = 0.0
pm.enforce_even_parity = False
pm.enforce_odd_parity = False
pm.k = 0.0

x_tensor = grid.x()
x = grid.mesh.detach().cpu().numpy() if torch.is_tensor(grid.mesh) else np.array(grid.mesh)

# --------------------------------------------------
# evaluar el GS normal en la malla
# --------------------------------------------------
with torch.no_grad():
    psi_gs_torch = torch.exp(model_gs(x_tensor)).squeeze()

psi_gs = psi_gs_torch.detach().cpu().numpy()

# --------------------------------------------------
# construir el estado objetivo dark
# psi_target = psi_gs * tanh((x-x0)/ell_dark)
# --------------------------------------------------
x0_dark = 1.0

# usa esto si ya tienes xi_fit del benchmark con pared:
# ell_dark = np.sqrt(2.0) * xi_fit

# si no, empieza con un valor razonable:
ell_dark = 1.0

dark_factor = np.tanh((x - x0_dark) / ell_dark)
psi_target = psi_gs * dark_factor

# --------------------------------------------------
# crear una nueva NQS para ajustar el estado target
# la inicializamos desde el GS para partir cerca
# --------------------------------------------------
model_dark_fit = models.NQS(input_size, output_size, hidden_layers).to(device)
model_dark_fit.load_state_dict(torch.load(gs_model_path, map_location=device))
model_dark_fit.to(device)
model_dark_fit.train()

psi_target_torch = torch.tensor(psi_target, dtype=torch.complex128, device=device)

# --------------------------------------------------
# ajuste supervisado
# --------------------------------------------------
optimizer = torch.optim.Adam(model_dark_fit.parameters(), lr=1e-3)

n_epochs = 20000
print_every = 5000

loss_history = []

for epoch in range(n_epochs):
    optimizer.zero_grad()

    psi_pred = torch.exp(model_dark_fit(x_tensor)).squeeze()

    # pérdida L2 compleja
    loss = torch.mean(torch.abs(psi_pred - psi_target_torch)**2)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % print_every == 0:
        print(f'Epoch {epoch:6d} | loss = {loss.item():.6e}')

print(f'Final fit loss = {loss_history[-1]:.6e}')

model_dark_fit.eval()

# --------------------------------------------------
# guardar el modelo ajustado
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

# 1) densidad GS vs target vs fit
plt.figure(figsize=(8,5))
plt.plot(x, den_gs, label='GS density')
plt.plot(x, den_target, '--', label='dark target density')
plt.plot(x, den_fit, ':', label='fitted NQS density')
plt.axvline(0.0, color='k', linestyle=':', label='x=0')
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.title('Ground state and dark-tanh target')
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# 2) fase target vs fit
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

# 5) pérdida
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('L2 loss')
plt.title('Dark target fit loss history')
plt.grid(alpha=0.2)
plt.show()

# 6) zoom central de la densidad ajustada
plt.figure(figsize=(8,5))
mask_center = np.abs(x) < 10.0
plt.plot(x[mask_center], den_fit[mask_center], label='fitted density')
plt.axvline(0.0, color='k', linestyle=':', label='x=0')
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.title('Central notch of fitted dark target')
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# %%
print('Real-time evolution from fitted dark-tanh state')

# cargar el modelo dark ajustado
model_dyn = models.NQS(input_size, output_size, hidden_layers).to(device)
model_dyn.load_state_dict(torch.load(dark_fit_model_path, map_location=device))
model_dyn.to(device)
model_dyn.eval()

# --------------------------------------------------
# parámetros físicos
# --------------------------------------------------
pm.g = 1.0
pm.x0 = 0.0
pm.w = 0.01
pm.mu = 1.0

pm.wall = 0.0

# sin barrera al principio
pm.gauss_amplitude = 0.0
pm.gauss_width = 1.0
pm.gauss_x0 = 0.0

# sin kick
pm.k = 0.0

# sin phase step
pm.phase = 0.0
pm.phase_center = 0.0
pm.phase_width = 1.0
pm.use_phase_step = False

pm.enforce_even_parity = False
pm.enforce_odd_parity = False

# --------------------------------------------------
# tiempo real
# --------------------------------------------------
pm.dt = 0.05
pm.t_max = 10.0

pm.evolution = 'real'
pm.lambda_reg = 1j * 1e-2

file_path_real = utils.file_ID(
    pm.data_dir,
    file_name(pm.architecture, model_dyn.architecture, 'real_from_dark_tanh_fit')
    + f'_g_{pm.g}_w_{pm.w}',
    pm.data_format
)

integrator(model_dyn, grid, file_path=file_path_real)

# leer dinámica
real_evo = Dynamics(file_path=file_path_real, x_grid=grid.x())
psi = real_evo.psi
den = np.abs(psi)**2
params = real_evo.get_params()

# figura evolución
fig_path = utils.file_ID(
    pm.figs_dir,
    file_name(pm.architecture, model_dyn.architecture, 'real_from_dark_tanh_fit')
    + f'_g_{pm.g}_w_{pm.w}',
    pm.fig_format
)
plots.evo_fig_params_poster(real_evo.t_grid, grid.mesh, den.T, params, fig_path=fig_path)


# %%
x = grid.mesh.detach().cpu().numpy() if torch.is_tensor(grid.mesh) else np.array(grid.mesh)
t = real_evo.t_grid

xmin_list = []
rhomin_list = []

# posición inicial esperada del notch
x_prev = x0_dark

# ventana local de seguimiento
track_halfwidth = 4.0

for k in range(len(t)):
    denk = den[k, :]

    mask = np.abs(x - x_prev) < track_halfwidth
    idxs = np.where(mask)[0]

    den_loc = denk[idxs]
    i_loc = np.argmin(den_loc)
    i_glob = idxs[i_loc]

    # refinamiento parabólico usando el mínimo y sus vecinos
    if 0 < i_glob < len(x) - 1:
        x1, x2, x3 = x[i_glob - 1], x[i_glob], x[i_glob + 1]
        y1, y2, y3 = denk[i_glob - 1], denk[i_glob], denk[i_glob + 1]

        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
        C = y1 - A * x1**2 - B * x1

        x_min_k = -B / (2 * A)
        rho_min_k = A * x_min_k**2 + B * x_min_k + C
    else:
        x_min_k = x[i_glob]
        rho_min_k = denk[i_glob]

    xmin_list.append(x_min_k)
    rhomin_list.append(rho_min_k)

    # actualizar centro de búsqueda
    x_prev = x_min_k

xmin_arr = np.array(xmin_list)
rhomin_arr = np.array(rhomin_list)

# --------------------------------------------------
# plots del tracking
# --------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(t, xmin_arr)
plt.xlabel('t')
plt.ylabel(r'$x_{\min}(t)$')
plt.title('Tracked dark soliton position vs time')
plt.grid(alpha=0.2)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(t, rhomin_arr)
plt.xlabel('t')
plt.ylabel(r'$\rho_{\min}(t)$')
plt.title('Tracked minimum density vs time')
plt.grid(alpha=0.2)
plt.show()

# # %%
# from scipy.optimize import curve_fit

# # --------------------------------------------------
# # ajuste sinusoidal simple:
# # x_min(t) = x_c + A cos(omega t + phi)
# # --------------------------------------------------
# def xfit_func(t, x_c, A, omega, phi):
#     return x_c + A * np.cos(omega * t + phi)

# # usa solo la parte temporal "limpia"
# # ajusta este corte si ves que al final la norma deriva demasiado
# t_fit_min = 55.0
# t_fit_max = 295.0
# mask_fit_time = (t >= t_fit_min) & (t <= t_fit_max)

# t_fit = t[mask_fit_time]
# x_fit_data = xmin_arr[mask_fit_time]

# # guesses iniciales razonables
# x_c0 = np.mean(x_fit_data)
# A0 = 0.5 * (np.max(x_fit_data) - np.min(x_fit_data))
# omega0 = pm.w / np.sqrt(2.0)
# phi0 = 0.0

# p0 = [x_c0, A0, omega0, phi0]

# popt, pcov = curve_fit(xfit_func, t_fit, x_fit_data, p0=p0)
# x_c_fit, A_fit, omega_fit, phi_fit = popt

# omega_th = pm.w / np.sqrt(2.0)

# print('=== Sinusoidal fit of x_min(t) ===')
# print(f'x_c_fit   = {x_c_fit:.6e}')
# print(f'A_fit     = {A_fit:.6e}')
# print(f'omega_fit = {omega_fit:.6e}')
# print(f'phi_fit   = {phi_fit:.6e}')
# print(f'omega_th  = {omega_th:.6e}')
# print(f'relative error = {abs(omega_fit - omega_th)/abs(omega_th):.6e}')

# # incertidumbres estimadas del ajuste
# perr = np.sqrt(np.diag(pcov))
# print('\n=== Fit uncertainties ===')
# print(f'sigma_x_c   = {perr[0]:.6e}')
# print(f'sigma_A     = {perr[1]:.6e}')
# print(f'sigma_omega = {perr[2]:.6e}')
# print(f'sigma_phi   = {perr[3]:.6e}')

# # plot del ajuste
# plt.figure(figsize=(8,5))
# plt.plot(t, xmin_arr, label=r'$x_{\min}(t)$')
# plt.plot(t_fit, xfit_func(t_fit, *popt), '--', label='sinusoidal fit')
# plt.xlabel('t')
# plt.ylabel(r'$x_{\min}(t)$')
# plt.title('Dark soliton trajectory and sinusoidal fit')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# T_fit = 2.0 * np.pi / omega_fit
# T_th = 2.0 * np.pi / omega_th

# print('\n=== Oscillation period ===')
# print(f'T_fit = {T_fit:.6e}')
# print(f'T_th  = {T_th:.6e}')
# print(f'relative error = {abs(T_fit - T_th)/abs(T_th):.6e}')

# # --------------------------------------------------
# # 3) FFT de la trayectoria
# # --------------------------------------------------
# dt_time = t[1] - t[0]

# signal = xmin_arr - np.mean(xmin_arr)

# fft_vals = np.fft.rfft(signal)
# fft_freqs = np.fft.rfftfreq(len(signal), d=dt_time)
# power = np.abs(fft_vals)**2

# if len(power) > 1:
#     idx_peak = np.argmax(power[1:]) + 1
#     omega_num = 2.0 * np.pi * fft_freqs[idx_peak]
# else:
#     idx_peak = 0
#     omega_num = np.nan

# omega_th = pm.w / np.sqrt(2.0)

# print('=== Soliton oscillation frequency ===')
# print(f'omega_theory = {omega_th:.6e}')
# print(f'omega_num    = {omega_num:.6e}')
# print(f'relative error = {abs(omega_num - omega_th)/abs(omega_th):.6e}')

# plt.figure(figsize=(8,5))
# plt.plot(2.0*np.pi*fft_freqs, power)
# plt.axvline(omega_th, color='r', linestyle='--', label=rf'$\omega/\sqrt{{2}}={omega_th:.4e}$')
# plt.axvline(omega_num, color='k', linestyle=':', label=rf'$\omega_{{num}}={omega_num:.4e}$')
# plt.xlabel(r'angular frequency $\omega$')
# plt.ylabel('FFT power')
# plt.title('FFT of dark soliton trajectory')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()
# norm_t = [np.trapz(den[k, :], x) for k in range(len(t))]

# plt.figure(figsize=(8,5))
# plt.plot(t, norm_t)
# plt.xlabel('t')
# plt.ylabel('Norm')
# plt.title('Norm conservation')
# plt.grid(alpha=0.2)
# plt.show()

# print(f'norm initial = {norm_t[0]:.6e}')
# print(f'norm final   = {norm_t[-1]:.6e}')
# print(f'relative drift = {abs(norm_t[-1]-norm_t[0])/abs(norm_t[0]):.6e}')
# %%
print('Step 1 — Analytical reproduction of Figure 3: two dark-dark solitons')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# Plot style
# --------------------------------------------------
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ==================================================
# Figure 3 parameters
# ==================================================

q_plus = np.array([1.0 + 0.0j, 0.75 + 0.0j], dtype=complex)

kappa1 = 1.0
kappa2 = 0.5

gamma1 = 1.0j
gamma2 = 2.0j

# q0 = ||q_+||
q0 = np.linalg.norm(q_plus)

# zeta_j = kappa_j + i nu_j, with |zeta_j| = q0
nu1 = np.sqrt(q0**2 - kappa1**2)
nu2 = np.sqrt(q0**2 - kappa2**2)

zeta1 = kappa1 + 1j * nu1
zeta2 = kappa2 + 1j * nu2

print('\n── Parameters ──')
print(f'q_plus = {q_plus}')
print(f'q0     = {q0:.8f}')
print(f'kappa1 = {kappa1:.8f}')
print(f'kappa2 = {kappa2:.8f}')
print(f'nu1    = {nu1:.8f}')
print(f'nu2    = {nu2:.8f}')
print(f'zeta1  = {zeta1:.8f}')
print(f'zeta2  = {zeta2:.8f}')
print(f'gamma1 = {gamma1}')
print(f'gamma2 = {gamma2}')

# ==================================================
# Space-time grid
# ==================================================

x_min, x_max = -10.0, 20.0
t_min, t_max = -6.0, 6.0

Nx = 600
Nt = 501

x = np.linspace(x_min, x_max, Nx)
t = np.linspace(t_min, t_max, Nt)

X, T = np.meshgrid(x, t, indexing='ij')

# ==================================================
# Analytical solution: Eq. (3.1a), Eq. (3.1b)
# ==================================================

def two_dark_dark_solution(X, T, q_plus, zeta1, zeta2, gamma1, gamma2, q0):
    """
    Analytical two dark-dark soliton solution.

    Returns
    -------
    q : array, shape (2, Nx, Nt)
        Two-component solution q = (q1, q2).
    D : array, shape (Nx, Nt)
        Denominator D(x,t).
    """

    kappa1 = np.real(zeta1)
    kappa2 = np.real(zeta2)

    nu1 = np.imag(zeta1)
    nu2 = np.imag(zeta2)

    xi1 = X - 2.0 * kappa1 * T
    xi2 = X - 2.0 * kappa2 * T

    E1 = np.exp(2.0 * nu1 * xi1)
    E2 = np.exp(2.0 * nu2 * xi2)
    E12 = E1 * E2

    z1 = zeta1
    z2 = zeta2

    z1c = np.conj(z1)
    z2c = np.conj(z2)

    # Denominator D(x,t), Eq. (3.1b)
    D = (
        1.0
        + gamma1 / (z1 - z1c) * E1
        + gamma2 / (z2 - z2c) * E2
        + (
            gamma1 * gamma2 * np.abs(z1 - z2)**2
            / (
                (z1 - z1c)
                * (z2 - z2c)
                * np.abs(z1c - z2)**2
            )
        ) * E12
    )

    # Correction term in Eq. (3.1a)
    cross_coeff = (
        gamma1 * gamma2
        * np.abs(z1 - z2)**2
        * (z1 * z2 - z1c * z2c)
        / (
            z1 * z2
            * (z1 - z1c)
            * (z2 - z2c)
            * np.abs(z1 - z2c)**2
        )
    )

    correction = (
        gamma1 / z1 * E1
        + gamma2 / z2 * E2
        + cross_coeff * E12
    )

    scalar_factor = np.exp(2.0j * q0**2 * T) * (1.0 - correction / D)

    q = q_plus[:, np.newaxis, np.newaxis] * scalar_factor[np.newaxis, :, :]

    return q, D


q_exact, D_exact = two_dark_dark_solution(
    X,
    T,
    q_plus=q_plus,
    zeta1=zeta1,
    zeta2=zeta2,
    gamma1=gamma1,
    gamma2=gamma2,
    q0=q0
)

q1 = q_exact[0]
q2 = q_exact[1]

rho1 = np.abs(q1)**2
rho2 = np.abs(q2)**2

# ==================================================
# Diagnostics
# ==================================================

print('\n── Diagnostics ──')
print(f'max Im(D)       = {np.max(np.abs(np.imag(D_exact))):.3e}')
print(f'min Re(D)       = {np.min(np.real(D_exact)):.8e}')
print(f'max Re(D)       = {np.max(np.real(D_exact)):.8e}')
print(f'rho1 min/max    = {rho1.min():.8f}, {rho1.max():.8f}')
print(f'rho2 min/max    = {rho2.min():.8f}, {rho2.max():.8f}')
print(f'rho2/rho1 bg    = {(0.75**2):.8f}')

# ==================================================
# Helper plotting functions
# ==================================================

def clean_3d_axis(ax, elev=28, azim=-138):
    ax.view_init(elev=elev, azim=azim)

    # transparent panes for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # softer grid
    ax.grid(True, alpha=0.25)

    # aspect ratio: wider than tall, closer to paper
    ax.set_box_aspect((1.35, 1.0, 0.6))


def plot_surface_paper_style(ax, Xp, Tp, Zp, zlabel, title, cmap='YlGnBu_r'):
    surf = ax.plot_surface(
        Xp,
        Tp,
        Zp,
        cmap=cmap,
        edgecolor='0.25',
        linewidth=0.35,
        antialiased=True,
        shade=True,
        alpha=0.98
    )

    ax.set_xlabel('x', labelpad=8)
    ax.set_ylabel('t', labelpad=8)
    ax.set_zlabel(zlabel, labelpad=8)
    ax.set_title(title, pad=12)

    clean_3d_axis(ax)

    return surf


# ==================================================
# 2D density maps
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True, sharey=True)

im1 = axes[0].pcolormesh(
    t, x, rho1,
    shading='auto',
    cmap='viridis'
)
axes[0].set_xlabel('t')
axes[0].set_ylabel('x')
axes[0].set_title(r'(a) $|q_1(x,t)|^2$')
cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.set_label(r'$|q_1|^2$')

im2 = axes[1].pcolormesh(
    t, x, rho2,
    shading='auto',
    cmap='viridis'
)
axes[1].set_xlabel('t')
axes[1].set_ylabel('x')
axes[1].set_title(r'(b) $|q_2(x,t)|^2$')
cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.set_label(r'$|q_2|^2$')

for ax in axes:
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(x_min, x_max)

plt.tight_layout()
plt.show()

# ==================================================
# 3D surface plots — improved readability
# ==================================================

# More aggressive downsampling so the mesh is visible and cleaner
stride_x = 6
stride_t = 6

X_plot = X[::stride_x, ::stride_t]
T_plot = T[::stride_x, ::stride_t]
rho1_plot = rho1[::stride_x, ::stride_t]
rho2_plot = rho2[::stride_x, ::stride_t]

fig = plt.figure(figsize=(13, 5.2))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = plot_surface_paper_style(
    ax1,
    X_plot,
    T_plot,
    rho1_plot,
    zlabel=r'$|q_1|^2$',
    title=r'(a) $|q_1(x,t)|^2$',
    cmap='YlGnBu_r'
)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(t_min, t_max)
ax1.set_zlim(rho1.min(), rho1.max() * 1.02)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = plot_surface_paper_style(
    ax2,
    X_plot,
    T_plot,
    rho2_plot,
    zlabel=r'$|q_2|^2$',
    title=r'(b) $|q_2(x,t)|^2$',
    cmap='YlGnBu_r'
)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(t_min, t_max)
ax2.set_zlim(rho2.min(), rho2.max() * 1.02)

# Optional colorbars
cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.70, pad=0.08)
cbar1.set_label(r'$|q_1|^2$')

cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.70, pad=0.08)
cbar2.set_label(r'$|q_2|^2$')

plt.tight_layout()
plt.show()

# ==================================================
# Snapshots at several times
# ==================================================

snapshot_times = [-5.0, -2.5, 0.0, 2.5, 5.0]
snapshot_indices = [np.argmin(np.abs(t - ts)) for ts in snapshot_times]

plt.figure(figsize=(8, 5))

for idx, ts in zip(snapshot_indices, snapshot_times):
    plt.plot(x, rho1[:, idx], linewidth=2.0, label=rf'$t={ts}$')

plt.xlabel('x')
plt.ylabel(r'$|q_1(x,t)|^2$')
plt.title(r'Snapshots of $|q_1|^2$')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

for idx, ts in zip(snapshot_indices, snapshot_times):
    plt.plot(x, rho2[:, idx], linewidth=2.0, label=rf'$t={ts}$')

plt.xlabel('x')
plt.ylabel(r'$|q_2(x,t)|^2$')
plt.title(r'Snapshots of $|q_2|^2$')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# %%
print('Step 2 — Scalar equivalent analytical solution')

# ==================================================
# Scalar-equivalent parameters
# ==================================================

# Original Figure 3 has q_plus = (1, 3/4)^T,
# whose norm is q0 = 1.25.
#
# Here we keep q0 = 1.25 but set the second component to zero.
# This gives a scalar-compatible version:
#     q1(x,t) != 0
#     q2(x,t) = 0
q_plus_scalar = np.array([1.25 + 0.0j, 0.0 + 0.0j], dtype=complex)

q0_scalar = np.linalg.norm(q_plus_scalar)

kappa1 = 1.0
kappa2 = 0.5

gamma1 = 1.0j
gamma2 = 2.0j

nu1_scalar = np.sqrt(q0_scalar**2 - kappa1**2)
nu2_scalar = np.sqrt(q0_scalar**2 - kappa2**2)

zeta1_scalar = kappa1 + 1j * nu1_scalar
zeta2_scalar = kappa2 + 1j * nu2_scalar

print('\n── Scalar-equivalent parameters ──')
print(f'q_plus_scalar = {q_plus_scalar}')
print(f'q0_scalar     = {q0_scalar:.8f}')
print(f'kappa1        = {kappa1:.8f}')
print(f'kappa2        = {kappa2:.8f}')
print(f'nu1_scalar    = {nu1_scalar:.8f}')
print(f'nu2_scalar    = {nu2_scalar:.8f}')
print(f'zeta1_scalar  = {zeta1_scalar:.8f}')
print(f'zeta2_scalar  = {zeta2_scalar:.8f}')
print(f'gamma1        = {gamma1}')
print(f'gamma2        = {gamma2}')

# ==================================================
# Use same grid as Step 1
# ==================================================

# If x, t, X, T already exist from Step 1, you can reuse them.
# Otherwise uncomment:
# x_min, x_max = -10.0, 20.0
# t_min, t_max = -6.0, 6.0
# Nx = 600
# Nt = 500
# x = np.linspace(x_min, x_max, Nx)
# t = np.linspace(t_min, t_max, Nt)
# X, T = np.meshgrid(x, t, indexing='ij')

# ==================================================
# Analytical scalar-compatible solution
# ==================================================

q_scalar_exact, D_scalar_exact = two_dark_dark_solution(
    X,
    T,
    q_plus=q_plus_scalar,
    zeta1=zeta1_scalar,
    zeta2=zeta2_scalar,
    gamma1=gamma1,
    gamma2=gamma2,
    q0=q0_scalar
)

psi_exact = q_scalar_exact[0]
q2_scalar = q_scalar_exact[1]

rho_exact = np.abs(psi_exact)**2
rho_q2_scalar = np.abs(q2_scalar)**2

print('\n── Scalar-equivalent diagnostics ──')
print(f'max Im(D)          = {np.max(np.abs(np.imag(D_scalar_exact))):.3e}')
print(f'min Re(D)          = {np.min(np.real(D_scalar_exact)):.8e}')
print(f'psi_exact rho min  = {rho_exact.min():.8f}')
print(f'psi_exact rho max  = {rho_exact.max():.8f}')
print(f'q2 max density     = {rho_q2_scalar.max():.8e}  (expected 0)')

# ==================================================
# 2D density map
# ==================================================

plt.figure(figsize=(7.5, 4.5))
plt.pcolormesh(t, x, rho_exact, shading='auto', cmap='viridis')
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'Scalar equivalent: $|\psi_{\rm exact}(x,t)|^2$')
plt.colorbar(label=r'$|\psi_{\rm exact}|^2$')
plt.tight_layout()
plt.show()

# ==================================================
# 3D surface plot
# ==================================================

stride_x = 6
stride_t = 6

X_plot = X[::stride_x, ::stride_t]
T_plot = T[::stride_x, ::stride_t]
rho_plot = rho_exact[::stride_x, ::stride_t]

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X_plot,
    T_plot,
    rho_plot,
    cmap='YlGnBu_r',
    edgecolor='0.25',
    linewidth=0.35,
    antialiased=True,
    alpha=0.98
)

ax.set_xlabel('x', labelpad=8)
ax.set_ylabel('t', labelpad=8)
ax.set_zlabel(r'$|\psi|^2$', labelpad=8)
ax.set_title(r'Scalar equivalent analytical solution')

ax.view_init(elev=28, azim=-138)
ax.set_box_aspect((1.35, 1.0, 0.6))

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08, label=r'$|\psi|^2$')

plt.tight_layout()
plt.show()

# ==================================================
# Snapshots
# ==================================================

snapshot_times = [-5.0, -2.5, 0.0, 2.5, 5.0]
snapshot_indices = [np.argmin(np.abs(t - ts)) for ts in snapshot_times]

plt.figure(figsize=(8, 5))

for idx, ts in zip(snapshot_indices, snapshot_times):
    plt.plot(x, rho_exact[:, idx], linewidth=2.0, label=rf'$t={ts}$')

plt.xlabel('x')
plt.ylabel(r'$|\psi_{\rm exact}(x,t)|^2$')
plt.title(r'Snapshots of scalar analytical solution')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# ==================================================
# Save t=0 target for future NQS fit
# ==================================================

idx_t0 = np.argmin(np.abs(t - 0.0))

psi_target_scalar_t0 = psi_exact[:, idx_t0]
rho_target_scalar_t0 = np.abs(psi_target_scalar_t0)**2

print('\n── NQS target candidate ──')
print(f't[idx_t0] = {t[idx_t0]:.8f}')
print(f'target norm at t=0 = {np.trapezoid(rho_target_scalar_t0, x):.8f}')
print(f'target rho min     = {rho_target_scalar_t0.min():.8f}')
print(f'target rho max     = {rho_target_scalar_t0.max():.8f}')

plt.figure(figsize=(8, 5))
plt.plot(x, rho_target_scalar_t0, label=r'$|\psi_{\rm target}(x,0)|^2$')
plt.xlabel('x')
plt.ylabel(r'$|\psi|^2$')
plt.title('Scalar analytical target at t=0 for NQS')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# print('Step 3 — NQS evolution vs scalar analytical two-dark-soliton solution')
# print('Using original NQS Hamiltonian convention + time rescaling')

# ==================================================
# Step 3 figure saving helpers
# ==================================================
# from pathlib import Path

# step3_figs_dir = Path("figs") / "step3_scalar_exact"
# step3_figs_dir.mkdir(parents=True, exist_ok=True)

# def time_to_tag(tval):
#     """
#     Convert time to a filename-safe tag.
#     Example:
#         -6.00 -> m6p00
#          0.00 -> 0p00
#          3.50 -> 3p50
#     """
#     s = f"{tval:.2f}"
#     return s.replace("-", "m").replace(".", "p")

# def save_step3_fig(label, arch_tag=None):
#     """
#     Save current matplotlib figure with a Step 3 name.
#     """
#     if arch_tag is None:
#         fname = f"step3_{label}.png"
#     else:
#         fname = f"step3_{label}_{arch_tag}.png"

#     fig_path = step3_figs_dir / fname
#     plt.savefig(fig_path, dpi=200, bbox_inches="tight")
#     print(f"Guardado: {fig_path}")
#     return fig_path


# ==================================================
# NQS grid for this analytical collision test
# ==================================================
# grid_scalar = utils.PointGrid(
#     N=2**10 + 1,
#     start=-40.0,
#     end=50.0,
#     device=device
# )

# x_tensor = grid_scalar.x()
# x_nqs = (
#     grid_scalar.mesh.detach().cpu().numpy()
#     if torch.is_tensor(grid_scalar.mesh)
#     else np.array(grid_scalar.mesh)
# )

# ==================================================
# Scalar analytical parameters from Step 2
# ==================================================
# q_plus_scalar = np.array([1.25 + 0.0j, 0.0 + 0.0j], dtype=complex)

# q0_scalar = np.linalg.norm(q_plus_scalar)

# kappa1 = 1.0
# kappa2 = 0.5

# gamma1 = 1.0j
# gamma2 = 2.0j

# nu1_scalar = np.sqrt(q0_scalar**2 - kappa1**2)
# nu2_scalar = np.sqrt(q0_scalar**2 - kappa2**2)

# zeta1_scalar = kappa1 + 1j * nu1_scalar
# zeta2_scalar = kappa2 + 1j * nu2_scalar

# ==================================================
# Time window
# ==================================================
# Analytical paper convention:

#     i psi_t = - psi_xx + 2 |psi|^2 psi - 4 q0^2 psi

# Original NQS Hamiltonian convention:

#     i psi_tau = -1/2 psi_xx + |psi|^2 psi - 2 q0^2 psi

# Therefore:

#     H_NQS = 0.5 * H_paper

# so the NQS time tau is twice slower. Compare using:

#     t_exact = t_exact_initial + 0.5 * tau_NQS

# t_exact_initial = -0.0
# t_exact_final   = +1.0

# time_scale = 0.5
# tau_max = (t_exact_final - t_exact_initial) / time_scale

# print('\n── Step 3 setup ──')
# print(f't_exact_initial = {t_exact_initial:.6f}')
# print(f't_exact_final   = {t_exact_final:.6f}')
# print(f'time_scale      = {time_scale:.6f}')
# print(f'NQS tau_max     = {tau_max:.6f}')
# print(f'q0_scalar       = {q0_scalar:.6f}')

# ==================================================
# Helper: evaluate scalar analytical solution on arbitrary x,t
# ==================================================
# def scalar_exact_solution_on_grid(x_grid, t_grid):
#     """
#     Returns psi_exact(x,t) = q1_exact(x,t) for the scalar-equivalent case.

#     Output shape:
#         psi_exact_xt: (Nx, Nt)
#     """
#     X_eval, T_eval = np.meshgrid(x_grid, t_grid, indexing='ij')

#     q_eval, D_eval = two_dark_dark_solution(
#         X_eval,
#         T_eval,
#         q_plus=q_plus_scalar,
#         zeta1=zeta1_scalar,
#         zeta2=zeta2_scalar,
#         gamma1=gamma1,
#         gamma2=gamma2,
#         q0=q0_scalar
#     )

#     psi_exact_xt = q_eval[0]

#     return psi_exact_xt, D_eval


# ==================================================
# PDE convention diagnosis for analytical solution
# ==================================================
# print('\nPDE convention diagnosis for scalar analytical solution')

# def diagnose_pde_convention(
#     x_diag,
#     t_diag,
#     psi_diag_xt,
#     kinetic_prefactor_current=-0.5,
#     g_current=1.0,
#     mu_current=1.0,
#     trim_x=8,
#     trim_t=8
# ):
#     """
#     Numerically fits the analytical solution to:

#         i psi_t = a psi_xx + b |psi|^2 psi + c psi

#     This helps identify the PDE convention satisfied by the analytical formula.
#     """

#     dx_diag = x_diag[1] - x_diag[0]
#     dt_diag = t_diag[1] - t_diag[0]

#     psi = psi_diag_xt

#     Central finite differences
#     psi_t = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dt_diag)

#     psi_xx = (
#         psi[2:, 1:-1]
#         - 2.0 * psi[1:-1, 1:-1]
#         + psi[:-2, 1:-1]
#     ) / dx_diag**2

#     psi_mid = psi[1:-1, 1:-1]
#     psi_t_mid = psi_t[1:-1, :]

#     if trim_x > 0:
#         psi_mid = psi_mid[trim_x:-trim_x, :]
#         psi_t_mid = psi_t_mid[trim_x:-trim_x, :]
#         psi_xx = psi_xx[trim_x:-trim_x, :]

#     if trim_t > 0:
#         psi_mid = psi_mid[:, trim_t:-trim_t]
#         psi_t_mid = psi_t_mid[:, trim_t:-trim_t]
#         psi_xx = psi_xx[:, trim_t:-trim_t]

#     rho_mid = np.abs(psi_mid)**2

#     Fit:
#         i psi_t = a psi_xx + b |psi|^2 psi + c psi
#     Y = (1j * psi_t_mid).reshape(-1)

#     A1 = psi_xx.reshape(-1)
#     A2 = (rho_mid * psi_mid).reshape(-1)
#     A3 = psi_mid.reshape(-1)

#     A = np.vstack([A1, A2, A3]).T

#     mask = np.isfinite(Y)
#     mask &= np.isfinite(A).all(axis=1)

#     A_fit = A[mask]
#     Y_fit = Y[mask]

#     coeffs, residuals, rank, svals = np.linalg.lstsq(A_fit, Y_fit, rcond=None)

#     a_fit, b_fit, c_fit = coeffs

#     Y_pred = A_fit @ coeffs
#     rel_res = np.linalg.norm(Y_fit - Y_pred) / np.linalg.norm(Y_fit)

#     print('\n── Fitted PDE coefficients ──')
#     print('Assumed fitted form:')
#     print('    i psi_t = a psi_xx + b |psi|^2 psi + c psi')
#     print(f'a_fit = {a_fit.real:+.10f} {a_fit.imag:+.3e}j')
#     print(f'b_fit = {b_fit.real:+.10f} {b_fit.imag:+.3e}j')
#     print(f'c_fit = {c_fit.real:+.10f} {c_fit.imag:+.3e}j')
#     print(f'relative residual = {rel_res:.3e}')

#     print('\nCurrent NQS/GPE convention used for comparison:')
#     print('    i psi_tau = kinetic_prefactor psi_xx + g |psi|^2 psi - mu psi')
#     print(f'a_NQS = {kinetic_prefactor_current:+.10f}')
#     print(f'b_NQS = {g_current:+.10f}')
#     print(f'c_NQS = {-mu_current:+.10f}')

#     print('\nComparison strategy:')
#     print('    analytical H ≈ 2 * H_NQS')
#     print('    t_exact = t_initial + 0.5 * tau_NQS')

#     return a_fit, b_fit, c_fit, rel_res


# x_diag = np.linspace(-20.0, 30.0, 1200)
# t_diag = np.linspace(-5.5, 5.5, 801)

# psi_diag_xt, _ = scalar_exact_solution_on_grid(x_diag, t_diag)

# Original NQS Hamiltonian parameters
# kinetic_prefactor_nqs = -0.5
# g_nqs = 1.0
# mu_nqs = 2.0 * q0_scalar**2

# a_fit, b_fit, c_fit, pde_rel_res = diagnose_pde_convention(
#     x_diag=x_diag,
#     t_diag=t_diag,
#     psi_diag_xt=psi_diag_xt,
#     kinetic_prefactor_current=kinetic_prefactor_nqs,
#     g_current=g_nqs,
#     mu_current=mu_nqs,
#     trim_x=10,
#     trim_t=10
# )

# print('\n── Original NQS Hamiltonian interpretation ──')
# print('Analytical solution approximately satisfies:')
# print('    i psi_t = -1 psi_xx + 2 |psi|^2 psi - 4 q0^2 psi')
# print('We use the original NQS Hamiltonian:')
# print('    i psi_tau = -1/2 psi_xx + |psi|^2 psi - 2 q0^2 psi')
# print('Therefore:')
# print('    H_NQS ≈ 0.5 * H_analytical')
# print('and')
# print('    t_exact = t_initial + 0.5 * tau_NQS')


# ==================================================
# Build initial target for NQS
# ==================================================
# psi_target_xt, _ = scalar_exact_solution_on_grid(
#     x_nqs,
#     np.array([t_exact_initial])
# )

# psi_target = psi_target_xt[:, 0]
# rho_target = np.abs(psi_target)**2

# norm_target = np.trapezoid(rho_target, x_nqs)
# pm.target_norm = float(norm_target)

# print('\n── Initial analytical target ──')
# print(f't_exact_initial     = {t_exact_initial:.8f}')
# print(f'target norm         = {norm_target:.8f}')
# print(f'target rho min/max  = {rho_target.min():.8f}, {rho_target.max():.8f}')


# ==================================================
# Fit NQS to analytical target
# ==================================================
# print('\nFitting NQS to scalar analytical initial state')

# hidden_layers_scalar = [25, 25]
# scalar_arch_tag = "HL_" + "_".join(str(h) for h in hidden_layers_scalar)

# print(f"Using hidden_layers_scalar = {hidden_layers_scalar}")
# print(f"scalar_arch_tag = {scalar_arch_tag}")


# ==================================================
# Initial target plots
# ==================================================
# plt.figure(figsize=(8, 5))
# plt.plot(x_nqs, rho_target, label=rf'analytical target, $t={t_exact_initial}$')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi_{\rm target}(x)|^2$')
# plt.title('Initial scalar analytical target for NQS')
# plt.grid(alpha=0.25)
# plt.legend()
# plt.tight_layout()
# save_step3_fig("target_density", scalar_arch_tag)
# plt.show()
# plt.close()

# plt.figure(figsize=(8, 5))
# plt.plot(x_nqs, psi_target.real, label='Re target')
# plt.plot(x_nqs, psi_target.imag, '--', label='Im target')
# plt.xlabel('x')
# plt.ylabel(r'$\psi_{\rm target}(x)$')
# plt.title('Complex initial target')
# plt.grid(alpha=0.25)
# plt.legend()
# plt.tight_layout()
# save_step3_fig("target_complex", scalar_arch_tag)
# plt.show()
# plt.close()


# model_scalar_fit = models.NQS(
#     input_size,
#     output_size,
#     hidden_layers_scalar
# ).to(device)

# model_scalar_fit.train()

# psi_target_torch = torch.tensor(
#     psi_target,
#     dtype=torch.complex128,
#     device=device
# )

# target_norm_torch = torch.tensor(
#     norm_target,
#     dtype=torch.float64,
#     device=device
# )

# def normalize_torch_to_target(psi, x_tensor, target_norm):
#     """
#     External normalization used only during Stage 1.
#     The final saved model is trained in Stage 2 without this external normalization.
#     """
#     x_real = x_tensor.squeeze().real
#     norm = torch.trapz(torch.abs(psi)**2, x_real).real

#     if norm <= 0 or not torch.isfinite(norm):
#         raise ValueError(f"Invalid torch norm during normalization: {norm}")

#     return psi / torch.sqrt(norm) * torch.sqrt(target_norm)


# ==================================================
# Supervised fit — Stage 1: normalized shape
# ==================================================
# optimizer = torch.optim.Adam(model_scalar_fit.parameters(), lr=1e-3)

# n_epochs_stage1 = 80000
# print_every = 10000

# loss_history_scalar = []

# print('\n── Stage 1: normalized-shape fit ──')

# for epoch in range(n_epochs_stage1):
#     optimizer.zero_grad()

#     psi_pred_raw = torch.exp(model_scalar_fit(x_tensor)).squeeze()

#     psi_pred = normalize_torch_to_target(
#         psi_pred_raw,
#         x_tensor,
#         target_norm_torch
#     )

#     loss_complex = torch.mean(torch.abs(psi_pred - psi_target_torch)**2)

#     loss_density = torch.mean(
#         (
#             torch.abs(psi_pred)**2
#             - torch.abs(psi_target_torch)**2
#         )**2
#     )

#     loss = loss_complex + 0.5 * loss_density

#     loss.backward()
#     optimizer.step()

#     loss_history_scalar.append(loss.item())

#     if epoch % print_every == 0:
#         print(
#             f'Stage 1 | Epoch {epoch:6d} | '
#             f'loss = {loss.item():.6e} | '
#             f'L2 = {loss_complex.item():.6e} | '
#             f'density = {loss_density.item():.6e}'
#         )

# print(f'Final Stage 1 loss = {loss_history_scalar[-1]:.6e}')


# ==================================================
# Supervised fit — Stage 2: raw physical scale
# ==================================================
# optimizer = torch.optim.Adam(model_scalar_fit.parameters(), lr=2e-4)

# n_epochs_stage2 = 50000

# print('\n── Stage 2: raw physical-scale fine tuning ──')

# for epoch in range(n_epochs_stage2):
#     optimizer.zero_grad()

#     psi_pred = torch.exp(model_scalar_fit(x_tensor)).squeeze()

#     x_real = x_tensor.squeeze().real
#     norm_pred = torch.trapz(torch.abs(psi_pred)**2, x_real).real

#     loss_complex = torch.mean(torch.abs(psi_pred - psi_target_torch)**2)

#     loss_density = torch.mean(
#         (
#             torch.abs(psi_pred)**2
#             - torch.abs(psi_target_torch)**2
#         )**2
#     )

#     loss_norm = ((norm_pred - target_norm_torch) / target_norm_torch)**2

#     loss = loss_complex + 0.5 * loss_density + 10.0 * loss_norm

#     loss.backward()
#     optimizer.step()

#     loss_history_scalar.append(loss.item())

#     if epoch % print_every == 0:
#         print(
#             f'Stage 2 | Epoch {epoch:6d} | '
#             f'loss = {loss.item():.6e} | '
#             f'L2 = {loss_complex.item():.6e} | '
#             f'density = {loss_density.item():.6e} | '
#             f'norm = {norm_pred.item():.8f}'
#         )

# print(f'Final total fit loss = {loss_history_scalar[-1]:.6e}')

# model_scalar_fit.eval()


# ==================================================
# Save fitted model
# ==================================================
# scalar_fit_model_path = utils.file_ID(
#     pm.data_dir,
#     file_name(
#         pm.architecture,
#         model_scalar_fit.architecture,
#         'scalar_exact_two_dark_fit_originalH',
#         scalar_arch_tag
#     ) + f'_q0_{q0_scalar}_tinit_{t_exact_initial}',
#     'pt'
# )

# torch.save(model_scalar_fit.state_dict(), scalar_fit_model_path)
# print(f'Scalar analytical fitted model saved to: {scalar_fit_model_path}')


# ==================================================
# Validate raw fit
# ==================================================
# with torch.no_grad():
#     psi_fit_torch = torch.exp(model_scalar_fit(x_tensor)).squeeze()

# psi_fit = psi_fit_torch.detach().cpu().numpy()
# rho_fit = np.abs(psi_fit)**2

# norm_fit = np.trapezoid(rho_fit, x_nqs)

# rho_l2 = np.sqrt(np.trapezoid((rho_fit - rho_target)**2, x_nqs))
# psi_l2 = np.sqrt(np.trapezoid(np.abs(psi_fit - psi_target)**2, x_nqs))

# print('\n── Raw fit diagnostics ──')
# print(f'norm target       = {norm_target:.8f}')
# print(f'norm fit          = {norm_fit:.8f}')
# print(f'L2 density error  = {rho_l2:.6e}')
# print(f'L2 psi error      = {psi_l2:.6e}')
# print(f'target rho min    = {rho_target.min():.8f}')
# print(f'fit rho min       = {rho_fit.min():.8f}')
# print(f'target rho max    = {rho_target.max():.8f}')
# print(f'fit rho max       = {rho_fit.max():.8f}')

# plt.figure(figsize=(8, 5))
# plt.plot(x_nqs, rho_target, label='analytical target')
# plt.plot(x_nqs, rho_fit, '--', label='NQS raw fit')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi|^2$')
# plt.title('Initial density: analytical target vs raw NQS fit')
# plt.grid(alpha=0.25)
# plt.legend()
# plt.tight_layout()
# save_step3_fig("fit_density", scalar_arch_tag)
# plt.show()
# plt.close()

# plt.figure(figsize=(8, 5))
# plt.plot(x_nqs, psi_target.real, label='Re target')
# plt.plot(x_nqs, psi_fit.real, '--', label='Re fit')
# plt.plot(x_nqs, psi_target.imag, label='Im target')
# plt.plot(x_nqs, psi_fit.imag, '--', label='Im fit')
# plt.xlabel('x')
# plt.ylabel(r'$\psi(x)$')
# plt.title('Initial wavefunction: analytical target vs raw NQS fit')
# plt.grid(alpha=0.25)
# plt.legend()
# plt.tight_layout()
# save_step3_fig("fit_wavefunction", scalar_arch_tag)
# plt.show()
# plt.close()

# plt.figure(figsize=(8, 5))
# plt.plot(loss_history_scalar)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Scalar analytical target fit loss')
# plt.grid(alpha=0.25)
# plt.tight_layout()
# save_step3_fig("fit_loss", scalar_arch_tag)
# plt.show()
# plt.close()


# ==================================================
# Real-time NQS evolution
# ==================================================
# print('\nReal-time NQS evolution from scalar analytical target')
# print('Using original NQS Hamiltonian convention + time rescaling')

# model_scalar_dyn = models.NQS(
#     input_size,
#     output_size,
#     hidden_layers_scalar
# ).to(device)

# model_scalar_dyn.load_state_dict(
#     torch.load(scalar_fit_model_path, map_location=device)
# )

# model_scalar_dyn.to(device)
# model_scalar_dyn.eval()

# --------------------------------------------------
# Physical parameters: original NQS Hamiltonian
# --------------------------------------------------
# pm.kinetic_prefactor = -0.5
# pm.g = 1.0
# pm.w = 0.0
# pm.x0 = 0.0
# pm.mu = 2.0 * q0_scalar**2

# pm.target_norm = float(norm_target)

# pm.wall = 0.0
# pm.gauss_amplitude = 0.0
# pm.gauss_width = 1.0
# pm.gauss_x0 = 0.0

# pm.k = 0.0
# pm.phase = 0.0
# pm.phase_center = 0.0
# pm.phase_width = 1.0
# pm.use_phase_step = False

# pm.enforce_even_parity = False
# pm.enforce_odd_parity = False

# --------------------------------------------------
# Time parameters
# --------------------------------------------------
# pm.evolution = 'real'
# pm.integrator = 'RK4'

# tau_max = 24 covers exact time [-6, 6] because time_scale = 0.5.
# pm.dt = 0.05
# pm.t_max = tau_max
# pm.t_size = 300

# Regularization for NQS dynamics
# pm.lambda_reg = 1e-3
# pm.pinv_rtol = 1e-7
# pm.max_param_step = 0.02

# pm.stopper = False

# Preflight check
# with torch.no_grad():
#     lnpsi_check = model_scalar_dyn(x_tensor)
#     psi_check = torch.exp(lnpsi_check)

# rho_check = torch.abs(psi_check.squeeze())**2
# norm_check = torch.trapz(rho_check, x_tensor.squeeze().real).real

# print('\n── Preflight scalar NQS check ──')
# print(f'finite lnpsi = {torch.isfinite(lnpsi_check).all().item()}')
# print(f'finite psi   = {torch.isfinite(psi_check).all().item()}')
# print(f'lnpsi real min/max = {lnpsi_check.real.min().item():.6e}, {lnpsi_check.real.max().item():.6e}')
# print(f'lnpsi imag min/max = {lnpsi_check.imag.min().item():.6e}, {lnpsi_check.imag.max().item():.6e}')
# print(f'|psi| min/max       = {torch.abs(psi_check).min().item():.6e}, {torch.abs(psi_check).max().item():.6e}')
# print(f'raw NQS norm        = {norm_check.item():.8f}')
# print(f'target norm         = {norm_target:.8f}')

# if not torch.isfinite(lnpsi_check).all() or not torch.isfinite(psi_check).all():
#     raise ValueError('Initial scalar NQS contains non-finite values.')


# --------------------------------------------------
# Run integrator
# --------------------------------------------------
# file_path_scalar_real = utils.file_ID(
#     pm.data_dir,
#     file_name(
#         pm.architecture,
#         model_scalar_dyn.architecture,
#         'real_scalar_exact_two_dark_originalH',
#         scalar_arch_tag
#     ) + f'_q0_{q0_scalar}_tinit_{t_exact_initial}_dt_{pm.dt}',
#     pm.data_format
# )

# integrator(
#     model_scalar_dyn,
#     grid_scalar,
#     file_path=file_path_scalar_real
# )


# --------------------------------------------------
# Load dynamics
# --------------------------------------------------
# scalar_evo = Dynamics(
#     file_path=file_path_scalar_real,
#     x_grid=grid_scalar.x()
# )

# psi_nqs = scalar_evo.psi
# rho_nqs = np.abs(psi_nqs)**2
# tau_nqs = scalar_evo.t_grid

# Original Hamiltonian: compare with rescaled analytical time
# t_exact_nqs = t_exact_initial + time_scale * tau_nqs

# psi_exact_xt_compare, _ = scalar_exact_solution_on_grid(
#     x_nqs,
#     t_exact_nqs
# )

# rho_exact_compare = np.abs(psi_exact_xt_compare)**2
# rho_exact_compare = rho_exact_compare.T

# Nt_compare = min(rho_nqs.shape[0], rho_exact_compare.shape[0])
# Nx_compare = min(rho_nqs.shape[1], rho_exact_compare.shape[1])

# rho_nqs_cmp = rho_nqs[:Nt_compare, :Nx_compare]
# rho_exact_cmp = rho_exact_compare[:Nt_compare, :Nx_compare]

# tau_cmp = tau_nqs[:Nt_compare]
# t_exact_cmp = t_exact_nqs[:Nt_compare]
# x_cmp = x_nqs[:Nx_compare]


# ==================================================
# Error metrics
# ==================================================
# rho_error = rho_nqs_cmp - rho_exact_cmp

# l2_space_time = np.sqrt(
#     np.trapezoid(
#         np.trapezoid(rho_error**2, x_cmp, axis=1),
#         tau_cmp
#     )
# )

# l2_exact_space_time = np.sqrt(
#     np.trapezoid(
#         np.trapezoid(rho_exact_cmp**2, x_cmp, axis=1),
#         tau_cmp
#     )
# )

# rel_l2_space_time = l2_space_time / l2_exact_space_time

# max_abs_error = np.max(np.abs(rho_error))

# norm_nqs = np.array([
#     np.trapezoid(rho_nqs_cmp[it, :], x_cmp)
#     for it in range(Nt_compare)
# ])

# norm_exact = np.array([
#     np.trapezoid(rho_exact_cmp[it, :], x_cmp)
#     for it in range(Nt_compare)
# ])

# norm_rel_drift_nqs = np.max(np.abs(norm_nqs - norm_nqs[0])) / abs(norm_nqs[0])

# mask_inner = (x_cmp > -15.0) & (x_cmp < 25.0)

# rho_error_inner = rho_nqs_cmp[:, mask_inner] - rho_exact_cmp[:, mask_inner]

# l2_inner = np.sqrt(
#     np.trapezoid(
#         np.trapezoid(rho_error_inner**2, x_cmp[mask_inner], axis=1),
#         tau_cmp
#     )
# )

# l2_exact_inner = np.sqrt(
#     np.trapezoid(
#         np.trapezoid(rho_exact_cmp[:, mask_inner]**2, x_cmp[mask_inner], axis=1),
#         tau_cmp
#     )
# )

# rel_l2_inner = l2_inner / l2_exact_inner

# print('\n── NQS vs analytical density comparison ──')
# print(f'Hamiltonian convention       = original NQS')
# print(f'time_scale used             = {time_scale:.6f}')
# print(f'NQS tau range compared      = [{tau_cmp[0]:.6f}, {tau_cmp[-1]:.6f}]')
# print(f't_exact range compared      = [{t_exact_cmp[0]:.6f}, {t_exact_cmp[-1]:.6f}]')
# print(f'relative L2 spacetime error = {rel_l2_space_time:.6e}')
# print(f'relative L2 inner error     = {rel_l2_inner:.6e}')
# print(f'max absolute density error  = {max_abs_error:.6e}')
# print(f'NQS norm initial            = {norm_nqs[0]:.8f}')
# print(f'NQS norm final              = {norm_nqs[-1]:.8f}')
# print(f'NQS max relative norm drift = {norm_rel_drift_nqs:.6e}')
# print(f'Exact norm initial          = {norm_exact[0]:.8f}')
# print(f'Exact norm final            = {norm_exact[-1]:.8f}')


# ==================================================
# Plots: analytical, NQS, and error
# ==================================================
# fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)

# im0 = axes[0].pcolormesh(
#     t_exact_cmp,
#     x_cmp,
#     rho_exact_cmp.T,
#     shading='auto',
#     cmap='viridis'
# )
# axes[0].set_title(r'Analytical $|\psi_{\rm exact}|^2$')
# axes[0].set_xlabel('exact t')
# axes[0].set_ylabel('x')
# plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# im1 = axes[1].pcolormesh(
#     t_exact_cmp,
#     x_cmp,
#     rho_nqs_cmp.T,
#     shading='auto',
#     cmap='viridis'
# )
# axes[1].set_title(r'NQS $|\psi_{\rm NQS}|^2$')
# axes[1].set_xlabel('exact t')
# plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# err_lim = np.max(np.abs(rho_error))

# im2 = axes[2].pcolormesh(
#     t_exact_cmp,
#     x_cmp,
#     rho_error.T,
#     shading='auto',
#     cmap='coolwarm',
#     vmin=-err_lim,
#     vmax=+err_lim
# )
# axes[2].set_title(r'NQS $-$ analytical')
# axes[2].set_xlabel('exact t')
# plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# plt.tight_layout()
# save_step3_fig("spacetime_compare", scalar_arch_tag)
# plt.show()
# plt.close()


# ==================================================
# Snapshot comparison
# ==================================================
# snapshot_exact_times = [0.0, 0.5, 1.0, 1.5, 2.0]
# snapshot_exact_times = [
#     ts for ts in snapshot_exact_times
#     if t_exact_cmp[0] <= ts <= t_exact_cmp[-1]
# ]

# for ts in snapshot_exact_times:
#     idx = np.argmin(np.abs(t_exact_cmp - ts))

#     plt.figure(figsize=(8, 5))
#     plt.plot(
#         x_cmp,
#         rho_exact_cmp[idx, :],
#         label=rf'analytical, $t={t_exact_cmp[idx]:.2f}$',
#         linewidth=2.0
#     )
#     plt.plot(
#         x_cmp,
#         rho_nqs_cmp[idx, :],
#         '--',
#         label='NQS',
#         linewidth=2.0
#     )

#     plt.xlabel('x')
#     plt.ylabel(r'$|\psi|^2$')
#     plt.title(rf'Density snapshot comparison at $t={t_exact_cmp[idx]:.2f}$')
#     plt.grid(alpha=0.25)
#     plt.legend()
#     plt.tight_layout()

#     t_tag = time_to_tag(t_exact_cmp[idx])
#     save_step3_fig(f"snapshot_t_{t_tag}", scalar_arch_tag)

#     plt.show()
#     plt.close()


# ==================================================
# Norm comparison
# ==================================================
# plt.figure(figsize=(8, 5))
# plt.plot(t_exact_cmp, norm_exact, label='exact norm')
# plt.plot(t_exact_cmp, norm_nqs, '--', label='NQS norm')
# plt.xlabel('exact t')
# plt.ylabel('norm')
# plt.title('Norm: NQS vs analytical')
# plt.grid(alpha=0.25)
# plt.legend()
# plt.tight_layout()
# save_step3_fig("norm", scalar_arch_tag)
# plt.show()
# plt.close()

# %%
print('Step 3 — Symmetric dark-dark collision: NQS vs analytical solution')
print('Background phase factorization active: psi = q0 * exp(i*q0^2*tau) * phi')

# ==================================================
# NQS grid
# ==================================================
grid_scalar = utils.PointGrid(
    N=2**9 + 1,        # 513 points — enough to resolve the notches
    start=-20.0,
    end=20.0,
    device=device
)

x_tensor = grid_scalar.x()
x_nqs = (
    grid_scalar.mesh.detach().cpu().numpy()
    if torch.is_tensor(grid_scalar.mesh)
    else np.array(grid_scalar.mesh)
)

# ==================================================
# Analytical parameters
# ==================================================
q_plus_scalar = np.array([1.25 + 0.0j, 0.0 + 0.0j], dtype=complex)
q0_scalar     = np.linalg.norm(q_plus_scalar)   # = 1.25

kappa_abs = 0.60
kappa1    = +kappa_abs
kappa2    = -kappa_abs

nu1_scalar = np.sqrt(q0_scalar**2 - kappa1**2)
nu2_scalar = np.sqrt(q0_scalar**2 - kappa2**2)

zeta1_scalar = kappa1 + 1j * nu1_scalar
zeta2_scalar = kappa2 + 1j * nu2_scalar

# Initial soliton positions
x_sep       = 4.0
x1_initial  = -x_sep
x2_initial  = +x_sep

gamma1 = 1j * 2.0 * nu1_scalar * np.exp(2.0 * nu1_scalar * x1_initial)
gamma2 = 1j * 2.0 * nu2_scalar * np.exp(2.0 * nu2_scalar * x2_initial)

print('\n── Analytical parameters ──')
print(f'q0          = {q0_scalar:.8f}')
print(f'kappa1/2    = ±{kappa_abs:.8f}')
print(f'nu1/2       = {nu1_scalar:.8f}')
print(f'v1/2        = ±{2.0*kappa_abs:.8f}')
print(f'rho_min     = kappa^2 = {kappa_abs**2:.8f}')
print(f'x1/2 init   = ±{x_sep:.8f}')

# ==================================================
# Time window
# ==================================================
# Analytical PDE:   i psi_t   = -     psi_xx + 2|psi|^2 psi - 4*q0^2 psi
# NQS Hamiltonian:  i psi_tau = - 0.5 psi_xx +  |psi|^2 psi - 2*q0^2 psi
#
# H_NQS = 0.5 * H_paper  =>  t_exact = t_init + 0.5 * tau_NQS

t_exact_initial   = 0.0
t_exact_final     = 2.0
time_scale        = 0.5
tau_max           = (t_exact_final - t_exact_initial) / time_scale
t_collision_est   = x_sep / (2.0 * kappa_abs)

print(f'\ntau_max = {tau_max:.4f}  (covers t_exact in [{t_exact_initial}, {t_exact_final}])')
print(f'estimated collision at t_exact = {t_collision_est:.4f}')

# ==================================================
# Helper: evaluate analytical solution
# ==================================================
def scalar_exact_solution_on_grid(x_grid, t_grid):
    X_eval, T_eval = np.meshgrid(x_grid, t_grid, indexing='ij')
    q_eval, D_eval = two_dark_dark_solution(
        X_eval, T_eval,
        q_plus=q_plus_scalar,
        zeta1=zeta1_scalar, zeta2=zeta2_scalar,
        gamma1=gamma1, gamma2=gamma2,
        q0=q0_scalar
    )
    return q_eval[0], D_eval   # shape (Nx, Nt)

# ==================================================
# Initial target
# ==================================================
psi_target_xt, _ = scalar_exact_solution_on_grid(
    x_nqs, np.array([t_exact_initial])
)

psi_target    = psi_target_xt[:, 0]
rho_target    = np.abs(psi_target)**2
norm_target   = np.trapezoid(rho_target, x_nqs)
pm.target_norm = float(norm_target)

print(f'\ntarget norm = {norm_target:.8f}')
print(f'target rho min/max = {rho_target.min():.6f}, {rho_target.max():.6f}')

plt.figure(figsize=(8, 4))
plt.plot(x_nqs, rho_target, label=rf'analytical, $t={t_exact_initial}$')
plt.axvline(x1_initial, ls=':', color='gray', label='initial notch positions')
plt.axvline(x2_initial, ls=':', color='gray')
plt.xlabel('x'); plt.ylabel(r'$|\psi|^2$')
plt.title('Initial target density'); plt.legend(); plt.grid(alpha=0.25)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
plt.plot(x_nqs, psi_target.real, label='Re target')
plt.plot(x_nqs, psi_target.imag, '--', label='Im target')
plt.xlabel('x'); plt.ylabel(r'$\psi$')
plt.title('Initial target wavefunction'); plt.legend(); plt.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# ==================================================
# Activate background phase BEFORE creating the model
# ==================================================
# The NQS models log(phi), where:
#   psi(x, tau) = q0 * exp(+i * q0^2 * tau) * phi(x, tau)
#   phi -> 1 as x -> ±inf  =>  log(phi) -> 0  (sigmoid-friendly)
#
# models.py forward() adds analytically:
#   log(psi) = log(phi_network(x)) + log(q0) + i * q0^2 * tau
#
# At tau=0: exp(forward()) = phi_network * q0
# So fitting exp(forward()) to psi_target is correct without changing
# psi_target_torch (it automatically fits phi_network to psi_target/q0).

pm.use_background_phase = True
pm.bg_q0    = float(q0_scalar)      # = 1.25
pm.bg_freq  = float(q0_scalar**2)   # = 1.5625
pm.current_tau = 0.0                # reset before fitting (tau=0)

# ==================================================
# Create and fit NQS
# ==================================================
hidden_layers_scalar = [20, 20]

model_scalar_fit = models.NQS(
    input_size, output_size, hidden_layers_scalar
).to(device)

model_scalar_fit.train()

psi_target_torch  = torch.tensor(psi_target,  dtype=torch.complex128, device=device)
target_norm_torch = torch.tensor(norm_target, dtype=torch.float64,    device=device)

def normalize_to_target(psi, x_t, target_norm):
    x_real = x_t.squeeze().real
    norm   = torch.trapz(torch.abs(psi)**2, x_real).real
    if norm <= 0 or not torch.isfinite(norm):
        raise ValueError(f'Invalid norm: {norm}')
    return psi / torch.sqrt(norm) * torch.sqrt(target_norm)

# --------------------------------------------------
# Stage 1: normalized-shape fit
# --------------------------------------------------
optimizer      = torch.optim.Adam(model_scalar_fit.parameters(), lr=1e-3)
n_epochs_s1    = 100000
print_every    = 10000
loss_history   = []

print('\n── Stage 1: normalized shape fit ──')
for epoch in range(n_epochs_s1):
    optimizer.zero_grad()

    psi_pred_raw = torch.exp(model_scalar_fit(x_tensor)).squeeze()
    psi_pred     = normalize_to_target(psi_pred_raw, x_tensor, target_norm_torch)

    loss_complex = torch.mean(torch.abs(psi_pred - psi_target_torch)**2)
    loss_density = torch.mean((torch.abs(psi_pred)**2 - torch.abs(psi_target_torch)**2)**2)
    loss         = loss_complex + 0.5 * loss_density

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % print_every == 0:
        print(f'  [{epoch:6d}] loss={loss.item():.4e}  L2={loss_complex.item():.4e}  rho={loss_density.item():.4e}')

print(f'Stage 1 final loss = {loss_history[-1]:.4e}')

# --------------------------------------------------
# Stage 2: raw physical-scale fine tuning
# --------------------------------------------------
optimizer   = torch.optim.Adam(model_scalar_fit.parameters(), lr=2e-4)
n_epochs_s2 = 40000

print('\n── Stage 2: physical-scale fine tuning ──')
for epoch in range(n_epochs_s2):
    optimizer.zero_grad()

    psi_pred  = torch.exp(model_scalar_fit(x_tensor)).squeeze()
    x_real    = x_tensor.squeeze().real
    norm_pred = torch.trapz(torch.abs(psi_pred)**2, x_real).real

    loss_complex = torch.mean(torch.abs(psi_pred - psi_target_torch)**2)
    loss_density = torch.mean((torch.abs(psi_pred)**2 - torch.abs(psi_target_torch)**2)**2)
    loss_norm    = ((norm_pred - target_norm_torch) / target_norm_torch)**2
    loss         = loss_complex + 0.5 * loss_density + 10.0 * loss_norm

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % print_every == 0:
        print(f'  [{epoch:6d}] loss={loss.item():.4e}  norm={norm_pred.item():.6f}')

print(f'Stage 2 final loss = {loss_history[-1]:.4e}')
model_scalar_fit.eval()

# --------------------------------------------------
# Validate fit
# --------------------------------------------------
with torch.no_grad():
    psi_fit = torch.exp(model_scalar_fit(x_tensor)).squeeze().cpu().numpy()

rho_fit  = np.abs(psi_fit)**2
norm_fit = np.trapezoid(rho_fit, x_nqs)
rho_l2   = np.sqrt(np.trapezoid((rho_fit - rho_target)**2, x_nqs))
psi_l2   = np.sqrt(np.trapezoid(np.abs(psi_fit - psi_target)**2, x_nqs))

print(f'\nnorm target/fit = {norm_target:.6f} / {norm_fit:.6f}')
print(f'L2 rho error   = {rho_l2:.4e}')
print(f'L2 psi error   = {psi_l2:.4e}')

# Check background: phi_network at boundaries should be ~1
# psi_fit at boundary should be ~q0 = 1.25
print(f'psi_fit at x=-20: {psi_fit[0]:.6f}  (should be ~{q0_scalar:.4f})')
print(f'psi_fit at x=+20: {psi_fit[-1]:.6f}  (should be ~{q0_scalar:.4f})')

plt.figure(figsize=(8, 4))
plt.plot(x_nqs, rho_target, label='analytical')
plt.plot(x_nqs, rho_fit, '--', label='NQS fit')
plt.xlabel('x'); plt.ylabel(r'$|\psi|^2$')
plt.title('Initial density: target vs NQS fit'); plt.legend(); plt.grid(alpha=0.25)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
plt.plot(x_nqs, psi_target.real, label='Re target')
plt.plot(x_nqs, psi_fit.real,   '--', label='Re fit')
plt.plot(x_nqs, psi_target.imag, label='Im target')
plt.plot(x_nqs, psi_fit.imag,   '--', label='Im fit')
plt.xlabel('x'); plt.ylabel(r'$\psi$')
plt.title('Wavefunction: target vs NQS fit'); plt.legend(); plt.grid(alpha=0.25)
plt.tight_layout(); plt.show()

plt.figure(figsize=(7, 3))
plt.semilogy(loss_history)
plt.xlabel('epoch'); plt.ylabel('loss')
plt.title('Fit loss history'); plt.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# ==================================================
# Save fitted model
# ==================================================
scalar_arch_tag = 'HL_' + '_'.join(str(h) for h in hidden_layers_scalar)

scalar_fit_model_path = utils.file_ID(
    pm.data_dir,
    file_name(
        pm.architecture,
        model_scalar_fit.architecture,
        'simple_symmetric_two_dark_fit_bgphase',
        scalar_arch_tag
    ) + f'_q0_{q0_scalar}_kappa_{kappa_abs}_xsep_{x_sep}',
    'pt'
)

torch.save(model_scalar_fit.state_dict(), scalar_fit_model_path)
print(f'\nFitted model saved to: {scalar_fit_model_path}')

# ==================================================
# Real-time NQS evolution
# ==================================================
print('\n── Real-time NQS evolution ──')

model_scalar_dyn = models.NQS(
    input_size, output_size, hidden_layers_scalar
).to(device)

model_scalar_dyn.load_state_dict(
    torch.load(scalar_fit_model_path, map_location=device)
)
model_scalar_dyn.eval()

# --------------------------------------------------
# Physical parameters
# --------------------------------------------------
pm.kinetic_prefactor = -0.5
pm.g   = 1.0
pm.w   = 0.0
pm.x0  = 0.0
pm.mu  = 2.0 * q0_scalar**2    # = 3.125

pm.wall             = 0.0
pm.gauss_amplitude  = 0.0
pm.gauss_width      = 1.0
pm.gauss_x0         = 0.0
pm.k                = 0.0
pm.phase            = 0.0
pm.phase_center     = 0.0
pm.phase_width      = 1.0
pm.use_phase_step   = False
pm.enforce_even_parity = False
pm.enforce_odd_parity  = False

# Background phase: stays ON during evolution.
# current_tau is reset to 0 here and updated by the integrator each micro-step.
pm.use_background_phase = True
pm.bg_q0    = float(q0_scalar)
pm.bg_freq  = float(q0_scalar**2)
pm.current_tau = 0.0            # <-- integrators.py increments this by pm.dt

# --------------------------------------------------
# Time parameters
# --------------------------------------------------
pm.evolution  = 'real'
pm.integrator = 'RK4'
pm.dt         = 0.005
pm.t_max      = tau_max
pm.t_size     = 900

pm.lambda_reg    = 1e-3
pm.pinv_rtol     = 1e-7
pm.max_param_step = 0.02
pm.stopper       = False

# --------------------------------------------------
# Preflight check
# --------------------------------------------------
with torch.no_grad():
    lnpsi_check = model_scalar_dyn(x_tensor)
    psi_check   = torch.exp(lnpsi_check)

norm_check = torch.trapz(
    torch.abs(psi_check.squeeze())**2,
    x_tensor.squeeze().real
).real

print(f'finite lnpsi  = {torch.isfinite(lnpsi_check).all().item()}')
print(f'finite psi    = {torch.isfinite(psi_check).all().item()}')
print(f'norm at tau=0 = {norm_check.item():.6f}  (target: {norm_target:.6f})')
print(f'psi boundary  = {psi_check.squeeze()[0].item():.4f} / {psi_check.squeeze()[-1].item():.4f}  (should be ~{q0_scalar:.4f})')

if not torch.isfinite(lnpsi_check).all() or not torch.isfinite(psi_check).all():
    raise ValueError('Initial NQS contains non-finite values — check fit quality.')

# --------------------------------------------------
# Run integrator
# --------------------------------------------------
file_path_scalar_real = utils.file_ID(
    pm.data_dir,
    file_name(
        pm.architecture,
        model_scalar_dyn.architecture,
        'real_simple_symmetric_two_dark_bgphase',
        scalar_arch_tag
    ) + f'_q0_{q0_scalar}_kappa_{kappa_abs}_xsep_{x_sep}_dt_{pm.dt}',
    pm.data_format
)

integrator(
    model_scalar_dyn,
    grid_scalar,
    file_path=file_path_scalar_real
)

# ==================================================
# Load and analyse dynamics
# ==================================================
scalar_evo = Dynamics(
    file_path=file_path_scalar_real,
    x_grid=grid_scalar.x()
)

psi_nqs  = scalar_evo.psi
rho_nqs  = np.abs(psi_nqs)**2
tau_nqs  = scalar_evo.t_grid

t_exact_nqs = t_exact_initial + time_scale * tau_nqs

psi_exact_xt, _ = scalar_exact_solution_on_grid(x_nqs, t_exact_nqs)

rho_exact = np.abs(psi_exact_xt)**2
rho_exact = rho_exact.T   # shape (Nt, Nx)

Nt = min(rho_nqs.shape[0], rho_exact.shape[0])
Nx = min(rho_nqs.shape[1], rho_exact.shape[1])

rho_nqs   = rho_nqs  [:Nt, :Nx]
rho_exact = rho_exact[:Nt, :Nx]
tau_cmp   = tau_nqs    [:Nt]
t_cmp     = t_exact_nqs[:Nt]
x_cmp     = x_nqs      [:Nx]

rho_err = rho_nqs - rho_exact

# Error metrics
rel_l2 = (
    np.sqrt(np.trapezoid(np.trapezoid(rho_err**2,   x_cmp, axis=1), tau_cmp))
  / np.sqrt(np.trapezoid(np.trapezoid(rho_exact**2, x_cmp, axis=1), tau_cmp))
)

norm_nqs_t   = np.array([np.trapezoid(rho_nqs[i],   x_cmp) for i in range(Nt)])
norm_exact_t = np.array([np.trapezoid(rho_exact[i], x_cmp) for i in range(Nt)])
norm_drift   = np.max(np.abs(norm_nqs_t - norm_nqs_t[0])) / abs(norm_nqs_t[0])

print('\n── Error metrics ──')
print(f'relative L2 (spacetime) = {rel_l2:.4e}')
print(f'max |rho_err|           = {np.max(np.abs(rho_err)):.4e}')
print(f'norm initial/final      = {norm_nqs_t[0]:.6f} / {norm_nqs_t[-1]:.6f}')
print(f'max relative norm drift = {norm_drift:.4e}')

# ==================================================
# Save directory
# ==================================================
import os

fig_dir = os.path.join(
    pm.figs_dir,
    file_name(
        pm.architecture,
        model_scalar_dyn.architecture,
        'simple_symmetric_two_dark_bgphase',
        scalar_arch_tag
    ) + f'_q0_{q0_scalar}_kappa_{kappa_abs}_xsep_{x_sep}_dt_{pm.dt}'
)
snapshot_dir = os.path.join(fig_dir, 'snapshots')
os.makedirs(snapshot_dir, exist_ok=True)
print(f'\nFigures in: {fig_dir}')

# ==================================================
# Spacetime density maps
# ==================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)

im0 = axes[0].pcolormesh(t_cmp, x_cmp, rho_exact.T, shading='auto', cmap='viridis')
axes[0].set_title(r'Analytical $|\psi_{\rm exact}|^2$')
axes[0].set_xlabel('exact t'); axes[0].set_ylabel('x')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].pcolormesh(t_cmp, x_cmp, rho_nqs.T, shading='auto', cmap='viridis')
axes[1].set_title(r'NQS $|\psi_{\rm NQS}|^2$')
axes[1].set_xlabel('exact t')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

err_lim = max(np.max(np.abs(rho_err)), 1e-12)
im2 = axes[2].pcolormesh(t_cmp, x_cmp, rho_err.T,
                          shading='auto', cmap='coolwarm',
                          vmin=-err_lim, vmax=+err_lim)
axes[2].set_title(r'NQS $-$ analytical')
axes[2].set_xlabel('exact t')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'density_maps.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, 'density_maps.pdf'), bbox_inches='tight')
plt.show()

# ==================================================
# Snapshots
# ==================================================
snapshot_times = [
    t_exact_initial,
    0.5 * (t_exact_initial + t_collision_est),
    t_collision_est,
    0.5 * (t_collision_est + t_exact_final),
    t_exact_final,
]
snapshot_times = [ts for ts in snapshot_times if t_cmp[0] <= ts <= t_cmp[-1]]

for ts in snapshot_times:
    idx    = np.argmin(np.abs(t_cmp - ts))
    t_plot = t_cmp[idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_cmp, rho_exact[idx], label=rf'analytical, $t={t_plot:.2f}$', lw=2)
    ax.plot(x_cmp, rho_nqs[idx],   '--', label='NQS', lw=2)
    ax.set_xlabel('x'); ax.set_ylabel(r'$|\psi|^2$')
    ax.set_title(rf'Density snapshot at $t = {t_plot:.2f}$')
    ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()

    t_name = f'{t_plot:+.3f}'.replace('+','p').replace('-','m').replace('.','d')
    plt.savefig(os.path.join(snapshot_dir, f'snapshot_t_{t_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(snapshot_dir, f'snapshot_t_{t_name}.pdf'), bbox_inches='tight')
    plt.show()

# ==================================================
# Norm conservation
# ==================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_cmp, norm_exact_t, label='exact norm')
ax.plot(t_cmp, norm_nqs_t,   '--', label='NQS norm')
ax.set_xlabel('exact t'); ax.set_ylabel('norm')
ax.set_title('Norm conservation'); ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'norm_conservation.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, 'norm_conservation.pdf'), bbox_inches='tight')
plt.show()
 # %%
# print('Two dark solitons OP mode via tanh-seeded NQS target')

# # --------------------------------------------------
# # OP-specific architecture
# # --------------------------------------------------
# # This architecture is used only for:
# #   1) fitting the two-dark OP target
# #   2) real-time OP evolution
# #
# # The ground state still uses the original hidden_layers.
# hidden_layers_OP = [15, 15]

# # --------------------------------------------------
# # cargar el ground state normal
# # --------------------------------------------------
# model_gs_OP = models.NQS(input_size, output_size, hidden_layers).to(device)
# model_gs_OP.load_state_dict(torch.load(gs_model_path, map_location=device))
# model_gs_OP.to(device)
# model_gs_OP.eval()

# # asegurar que no haya phase step ni paridad forzada
# pm.use_phase_step = False
# pm.phase = 0.0
# pm.enforce_even_parity = False
# pm.enforce_odd_parity = False
# pm.k = 0.0

# # parámetros físicos
# pm.g  = 1.0
# pm.w  = 0.01
# pm.x0 = 0.0
# pm.mu = 1.0

# Omega = pm.w
# q0    = 1.0

# x_tensor = grid.x()
# x = grid.mesh.detach().cpu().numpy() if torch.is_tensor(grid.mesh) else np.array(grid.mesh)

# # --------------------------------------------------
# # predicción analítica OP
# # --------------------------------------------------
# from scipy.special import lambertw

# omega_o = Omega / np.sqrt(2.0)

# z_arg = 32.0 * q0**4 / omega_o**2
# x_eq = (1.0 / (4.0 * q0)) * np.real(lambertw(z_arg))

# omega_OP_exact = np.sqrt(
#     omega_o**2
#     + 32.0 * q0**2 * np.exp(-4.0 * q0 * x_eq)
# )

# T_OP_exact = 2.0 * np.pi / omega_OP_exact

# print('\n── Analytical OP prediction ──')
# print(f'Omega          = {Omega:.8f}')
# print(f'omega_o        = {omega_o:.8f}')
# print(f'q0             = {q0:.8f}')
# print(f'x_eq           = {x_eq:.8f}')
# print(f'omega_OP exact = {omega_OP_exact:.8f}')
# print(f'T_OP exact     = {T_OP_exact:.8f}')

# # --------------------------------------------------
# # posición inicial de los dos notches
# # --------------------------------------------------
# delta_OP = 0.3
# x0_pair = x_eq + delta_OP

# print('\n── Initial OP notch positions ──')
# print(f'x_left(0)  = {-x0_pair:.8f}')
# print(f'x_right(0) = {+x0_pair:.8f}')
# print(f'delta_OP   = {delta_OP:.8f}')

# # --------------------------------------------------
# # evaluar GS normal en la malla
# # --------------------------------------------------
# with torch.no_grad():
#     psi_gs_torch_OP = torch.exp(model_gs_OP(x_tensor)).squeeze()

# psi_gs_OP = psi_gs_torch_OP.detach().cpu().numpy()

# # quitar fase global usando el centro
# center_idx = np.argmin(np.abs(x))
# global_phase = np.angle(psi_gs_OP[center_idx])
# psi_gs_OP = psi_gs_OP * np.exp(-1j * global_phase)

# # escalar el GS para que |psi(0)| = q0
# center_amp = np.abs(psi_gs_OP[center_idx])
# if center_amp <= 0:
#     raise ValueError("Ground-state amplitude at center is zero.")

# psi_bg_OP = psi_gs_OP * (q0 / center_amp)

# # --------------------------------------------------
# # construir target con dos solitones oscuros
# # --------------------------------------------------
# dark_factor_left_OP  = np.tanh(q0 * (x + x0_pair))
# dark_factor_right_OP = np.tanh(q0 * (x - x0_pair))

# psi_target_OP = psi_bg_OP * dark_factor_left_OP * dark_factor_right_OP

# norm_target_OP = np.trapezoid(np.abs(psi_target_OP)**2, x)
# pm.target_norm = float(norm_target_OP)

# print('\n── OP target diagnostics ──')
# print(f'norm target OP       = {norm_target_OP:.8f}')
# print(f'|psi_bg(0)|          = {np.abs(psi_bg_OP[center_idx]):.8f}')
# print(f'rho target OP max    = {np.max(np.abs(psi_target_OP)**2):.8f}')
# print(f'rho target OP min    = {np.min(np.abs(psi_target_OP)**2):.8e}')

# # --------------------------------------------------
# # plot target OP
# # --------------------------------------------------
# plt.figure(figsize=(8, 5))
# plt.plot(x, np.abs(psi_bg_OP)**2, label='scaled GS density')
# plt.plot(x, np.abs(psi_target_OP)**2, '--', label='two-dark OP target')
# plt.axvline(-x_eq, linestyle=':', label=r'$-x_\mathrm{eq}$')
# plt.axvline(+x_eq, linestyle=':', label=r'$+x_\mathrm{eq}$')
# plt.axvline(-x0_pair, linestyle='--', label='initial notches')
# plt.axvline(+x0_pair, linestyle='--')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi|^2$')
# plt.title('NQS target: two dark solitons OP mode')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(x, np.unwrap(np.angle(psi_target_OP)))
# plt.xlabel('x')
# plt.ylabel(r'$\arg[\psi_\mathrm{target}]$')
# plt.title('Two-dark OP target phase')
# plt.grid(alpha=0.2)
# plt.show()

# # --------------------------------------------------
# # crear nueva NQS para ajustar el target OP
# # --------------------------------------------------
# # IMPORTANT:
# # We do NOT load the GS state_dict here, because the architecture is different.
# # The GS model has hidden_layers=[5,5], while the OP model has hidden_layers_OP.
# model_OP_fit = models.NQS(input_size, output_size, hidden_layers_OP).to(device)
# model_OP_fit.train()

# psi_target_OP_torch = torch.tensor(
#     psi_target_OP,
#     dtype=torch.complex128,
#     device=device
# )

# target_norm_OP_torch = torch.tensor(
#     norm_target_OP,
#     dtype=torch.float64,
#     device=device
# )

# def normalize_torch_to_target(psi, x_tensor, target_norm):
#     x_real = x_tensor.squeeze().real
#     norm = torch.trapz(torch.abs(psi)**2, x_real).real

#     if norm <= 0 or not torch.isfinite(norm):
#         raise ValueError(f"Invalid torch norm during normalization: {norm}")

#     return psi / torch.sqrt(norm) * torch.sqrt(target_norm)

# # --------------------------------------------------
# # ajuste supervisado
# # --------------------------------------------------
# optimizer = torch.optim.Adam(model_OP_fit.parameters(), lr=1e-3)

# n_epochs = 250000
# print_every = 20000

# loss_history_OP = []

# for epoch in range(n_epochs):
#     optimizer.zero_grad()

#     psi_pred_raw = torch.exp(model_OP_fit(x_tensor)).squeeze()

#     # normalizar a la misma norma física que el target
#     psi_pred = normalize_torch_to_target(
#         psi_pred_raw,
#         x_tensor,
#         target_norm_OP_torch
#     )

#     # pérdida compleja
#     loss_complex = torch.mean(torch.abs(psi_pred - psi_target_OP_torch)**2)

#     # pérdida en densidad para reforzar los notches
#     loss_density = torch.mean(
#         (
#             torch.abs(psi_pred)**2
#             - torch.abs(psi_target_OP_torch)**2
#         )**2
#     )

#     loss = loss_complex + 0.1 * loss_density

#     loss.backward()
#     optimizer.step()

#     loss_history_OP.append(loss.item())

#     if epoch % print_every == 0:
#         print(
#             f'Epoch {epoch:6d} | '
#             f'loss = {loss.item():.6e} | '
#             f'L2 = {loss_complex.item():.6e} | '
#             f'density = {loss_density.item():.6e}'
#         )

# print(f'Final OP fit loss = {loss_history_OP[-1]:.6e}')

# model_OP_fit.eval()

# # --------------------------------------------------
# # guardar modelo OP ajustado
# # --------------------------------------------------
# OP_arch_tag = "HL_" + "_".join(str(h) for h in hidden_layers_OP)

# OP_fit_model_path = utils.file_ID(
#     pm.data_dir,
#     file_name(
#         pm.architecture,
#         model_OP_fit.architecture,
#         'two_dark_OP_fit',
#         OP_arch_tag
#     ) + f'_g_{pm.g}_w_{pm.w}',
#     'pt'
# )

# torch.save(model_OP_fit.state_dict(), OP_fit_model_path)
# print(f'Two-dark OP fitted model saved to: {OP_fit_model_path}')

# # --------------------------------------------------
# # validar ajuste
# # --------------------------------------------------
# with torch.no_grad():
#     psi_OP_fit_raw = torch.exp(model_OP_fit(x_tensor)).squeeze()
#     psi_OP_fit_torch = normalize_torch_to_target(
#         psi_OP_fit_raw,
#         x_tensor,
#         target_norm_OP_torch
#     )

# psi_OP_fit = psi_OP_fit_torch.detach().cpu().numpy()

# den_target_OP = np.abs(psi_target_OP)**2
# den_fit_OP = np.abs(psi_OP_fit)**2

# norm_fit_OP = np.trapezoid(den_fit_OP, x)

# rho_l2_OP = np.sqrt(
#     np.trapezoid((den_fit_OP - den_target_OP)**2, x)
# )

# psi_l2_OP = np.sqrt(
#     np.trapezoid(np.abs(psi_OP_fit - psi_target_OP)**2, x)
# )

# print('\n── OP fit diagnostics ──')
# print(f'norm target OP = {norm_target_OP:.8f}')
# print(f'norm fit OP    = {norm_fit_OP:.8f}')
# print(f'L2 density error OP = {rho_l2_OP:.6e}')
# print(f'L2 psi error OP     = {psi_l2_OP:.6e}')
# print(f'target rho min OP   = {den_target_OP.min():.6e}')
# print(f'fit rho min OP      = {den_fit_OP.min():.6e}')

# plt.figure(figsize=(8, 5))
# plt.plot(x, den_target_OP, label='target density')
# plt.plot(x, den_fit_OP, '--', label='fitted NQS density')
# plt.xlim(-10, 10)
# plt.xlabel('x')
# plt.ylabel(r'$|\psi|^2$')
# plt.title('Two-dark OP target vs fitted NQS density')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(x, np.unwrap(np.angle(psi_target_OP)), label='target phase')
# plt.plot(x, np.unwrap(np.angle(psi_OP_fit)), '--', label='fit phase')
# plt.xlim(-10, 10)
# plt.xlabel('x')
# plt.ylabel(r'$\arg[\psi]$')
# plt.title('Two-dark OP target vs fitted NQS phase')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(loss_history_OP)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Two-dark OP target fit loss')
# plt.grid(alpha=0.2)
# plt.show()

# # %%
# print('Real-time evolution from fitted two-dark OP state')

# # --------------------------------------------------
# # cargar modelo OP ajustado con la misma arquitectura grande
# # --------------------------------------------------
# model_OP_dyn = models.NQS(input_size, output_size, hidden_layers_OP).to(device)
# model_OP_dyn.load_state_dict(torch.load(OP_fit_model_path, map_location=device))
# model_OP_dyn.to(device)
# model_OP_dyn.eval()

# # --------------------------------------------------
# # parámetros físicos
# # --------------------------------------------------
# pm.g = 1.0
# pm.x0 = 0.0
# pm.w = Omega
# pm.mu = 1.0

# pm.target_norm = float(norm_target_OP)

# pm.wall = 0.0

# pm.gauss_amplitude = 0.0
# pm.gauss_width = 1.0
# pm.gauss_x0 = 0.0

# pm.k = 0.0

# pm.phase = 0.0
# pm.phase_center = 0.0
# pm.phase_width = 1.0
# pm.use_phase_step = False

# pm.enforce_even_parity = False
# pm.enforce_odd_parity = False

# # --------------------------------------------------
# # tiempo real
# # --------------------------------------------------
# pm.dt = 0.01
# pm.t_max = 450.0
# pm.t_size = 1000

# pm.evolution = 'real'
# pm.integrator = 'RK4'

# # Stronger regularization for the OP NQS dynamics
# pm.lambda_reg = 1e-1

# # Keep using pseudo-inverse, but with robust settings if implemented in EOM
# pm.pinv_rtol = 1e-8

# # Optional parameter-step clipping if implemented in EOM
# pm.max_param_step = 0.05

# pm.stopper = False

# # --------------------------------------------------
# # preflight check
# # --------------------------------------------------
# with torch.no_grad():
#     lnpsi_check = model_OP_dyn(x_tensor)
#     psi_check = torch.exp(lnpsi_check)

# print("\n── Preflight OP NQS check ──")
# print(f"finite lnpsi = {torch.isfinite(lnpsi_check).all().item()}")
# print(f"finite psi   = {torch.isfinite(psi_check).all().item()}")
# print(f"lnpsi real min/max = {lnpsi_check.real.min().item():.6e}, {lnpsi_check.real.max().item():.6e}")
# print(f"lnpsi imag min/max = {lnpsi_check.imag.min().item():.6e}, {lnpsi_check.imag.max().item():.6e}")
# print(f"|psi| min/max       = {torch.abs(psi_check).min().item():.6e}, {torch.abs(psi_check).max().item():.6e}")

# if not torch.isfinite(lnpsi_check).all() or not torch.isfinite(psi_check).all():
#     raise ValueError("Initial OP NQS has non-finite values before real-time evolution.")

# # --------------------------------------------------
# # ejecutar evolución
# # --------------------------------------------------
# file_path_OP_real = utils.file_ID(
#     pm.data_dir,
#     file_name(
#         pm.architecture,
#         model_OP_dyn.architecture,
#         'real_two_dark_OP',
#         OP_arch_tag
#     ) + f'_g_{pm.g}_w_{pm.w}_dt_{pm.dt}_reg_{pm.lambda_reg}',
#     pm.data_format
# )

# integrator(model_OP_dyn, grid, file_path=file_path_OP_real)

# # leer dinámica
# OP_evo = Dynamics(file_path=file_path_OP_real, x_grid=grid.x())

# psi_OP = OP_evo.psi
# den_OP = np.abs(psi_OP)**2
# t_OP = OP_evo.t_grid
# params_OP = OP_evo.get_params()

# # --------------------------------------------------
# # plot evolución
# # --------------------------------------------------
# fig_path_OP = utils.file_ID(
#     pm.figs_dir,
#     file_name(
#         pm.architecture,
#         model_OP_dyn.architecture,
#         'real_two_dark_OP',
#         OP_arch_tag
#     ) + f'_g_{pm.g}_w_{pm.w}_dt_{pm.dt}_reg_{pm.lambda_reg}',
#     pm.fig_format
# )

# plots.evo_fig_params_poster(
#     OP_evo.t_grid,
#     grid.mesh,
#     den_OP.T,
#     params_OP,
#     fig_path=fig_path_OP
# )

# # --------------------------------------------------
# # tracking local de los dos mínimos
# # --------------------------------------------------
# def parabolic_minimum(x, y, i):
#     if i <= 0 or i >= len(x) - 1:
#         return x[i], y[i]

#     x1, x2, x3 = x[i - 1], x[i], x[i + 1]
#     y1, y2, y3 = y[i - 1], y[i], y[i + 1]

#     denom = (x1 - x2) * (x1 - x3) * (x2 - x3)

#     if abs(denom) < 1e-14:
#         return x[i], y[i]

#     A = (
#         x3 * (y2 - y1)
#         + x2 * (y1 - y3)
#         + x1 * (y3 - y2)
#     ) / denom

#     B = (
#         x3**2 * (y1 - y2)
#         + x2**2 * (y3 - y1)
#         + x1**2 * (y2 - y1)
#     ) / denom

#     if abs(A) < 1e-14:
#         return x[i], y[i]

#     C = y1 - A * x1**2 - B * x1

#     x_min = -B / (2.0 * A)
#     y_min = A * x_min**2 + B * x_min + C

#     return x_min, y_min


# def track_two_minima_local(den, x, x_left0, x_right0, halfwidth=5.0):
#     x_left_list = []
#     x_right_list = []
#     rho_left_list = []
#     rho_right_list = []

#     x_left_prev = x_left0
#     x_right_prev = x_right0

#     for it in range(den.shape[0]):
#         rho_t = den[it, :]

#         mask_l = np.abs(x - x_left_prev) < halfwidth
#         idx_l = np.where(mask_l)[0]
#         if len(idx_l) == 0:
#             idx_l = np.arange(len(x))

#         i_l_local = np.argmin(rho_t[idx_l])
#         i_l = idx_l[i_l_local]
#         x_l, rho_l = parabolic_minimum(x, rho_t, i_l)

#         mask_r = np.abs(x - x_right_prev) < halfwidth
#         idx_r = np.where(mask_r)[0]
#         if len(idx_r) == 0:
#             idx_r = np.arange(len(x))

#         i_r_local = np.argmin(rho_t[idx_r])
#         i_r = idx_r[i_r_local]
#         x_r, rho_r = parabolic_minimum(x, rho_t, i_r)

#         x_left_list.append(x_l)
#         x_right_list.append(x_r)
#         rho_left_list.append(rho_l)
#         rho_right_list.append(rho_r)

#         x_left_prev = x_l
#         x_right_prev = x_r

#     return (
#         np.array(x_left_list),
#         np.array(x_right_list),
#         np.array(rho_left_list),
#         np.array(rho_right_list)
#     )


# x_left_OP, x_right_OP, rho_left_OP, rho_right_OP = track_two_minima_local(
#     den_OP,
#     x,
#     x_left0=-x0_pair,
#     x_right0=+x0_pair,
#     halfwidth=5.0
# )

# separation_OP = x_right_OP - x_left_OP

# # --------------------------------------------------
# # frecuencia OP
# # --------------------------------------------------
# from scipy.signal import find_peaks

# dt_saved = t_OP[1] - t_OP[0]
# min_distance = max(5, int(0.4 * T_OP_exact / dt_saved))

# peaks_sep_OP, _ = find_peaks(
#     separation_OP,
#     distance=min_distance,
#     prominence=0.05
# )

# troughs_sep_OP, _ = find_peaks(
#     -separation_OP,
#     distance=min_distance,
#     prominence=0.05
# )

# if len(peaks_sep_OP) >= 2:
#     periods_OP = np.diff(t_OP[peaks_sep_OP])
#     T_OP_num = np.mean(periods_OP)
#     omega_OP_num = 2.0 * np.pi / T_OP_num
#     err_OP = abs(omega_OP_num - omega_OP_exact) / omega_OP_exact

#     print('\n── NQS OP frequency from separation maxima ──')
#     print(f'Number of maxima detected = {len(peaks_sep_OP)}')
#     print(f'T_OP numerical            = {T_OP_num:.6f}')
#     print(f'T_OP exact                = {T_OP_exact:.6f}')
#     print(f'omega_OP numerical        = {omega_OP_num:.8f}')
#     print(f'omega_OP exact            = {omega_OP_exact:.8f}')
#     print(f'relative error            = {err_OP:.6e}')

# elif len(troughs_sep_OP) >= 2:
#     periods_OP = np.diff(t_OP[troughs_sep_OP])
#     T_OP_num = np.mean(periods_OP)
#     omega_OP_num = 2.0 * np.pi / T_OP_num
#     err_OP = abs(omega_OP_num - omega_OP_exact) / omega_OP_exact

#     print('\n── NQS OP frequency from separation minima ──')
#     print(f'Number of minima detected = {len(troughs_sep_OP)}')
#     print(f'T_OP numerical            = {T_OP_num:.6f}')
#     print(f'T_OP exact                = {T_OP_exact:.6f}')
#     print(f'omega_OP numerical        = {omega_OP_num:.8f}')
#     print(f'omega_OP exact            = {omega_OP_exact:.8f}')
#     print(f'relative error            = {err_OP:.6e}')

# else:
#     print('\nNot enough peaks detected for OP frequency.')
#     omega_OP_num = np.nan
#     err_OP = np.nan

# # --------------------------------------------------
# # conservación de norma
# # --------------------------------------------------
# norm_OP = np.array([
#     np.trapezoid(den_OP[it, :], x)
#     for it in range(len(t_OP))
# ])

# norm_rel_drift_OP = np.max(np.abs(norm_OP - norm_OP[0])) / abs(norm_OP[0])

# print('\n── NQS OP conservation diagnostics ──')
# print(f'norm initial = {norm_OP[0]:.8f}')
# print(f'norm final   = {norm_OP[-1]:.8f}')
# print(f'max relative norm drift = {norm_rel_drift_OP:.6e}')

# if hasattr(OP_evo, 'energy') and OP_evo.energy is not None:
#     energy_OP = np.real(OP_evo.energy)

#     energy_rel_drift_OP = (
#         np.max(np.abs(energy_OP - energy_OP[0]))
#         / abs(energy_OP[0])
#         if abs(energy_OP[0]) > 0
#         else np.nan
#     )

#     print(f'energy initial = {energy_OP[0]:.8f}')
#     print(f'energy final   = {energy_OP[-1]:.8f}')
#     print(f'max relative energy drift = {energy_rel_drift_OP:.6e}')

# # --------------------------------------------------
# # plots
# # --------------------------------------------------
# plt.figure(figsize=(10, 5))
# plt.plot(t_OP, separation_OP, label='NQS separation')
# plt.axhline(2.0 * x_eq, linestyle=':', label=r'$2x_\mathrm{eq}$')

# if len(peaks_sep_OP) > 0:
#     plt.plot(
#         t_OP[peaks_sep_OP],
#         separation_OP[peaks_sep_OP],
#         'x',
#         color='red',
#         label='detected maxima'
#     )

# if len(troughs_sep_OP) > 0:
#     plt.plot(
#         t_OP[troughs_sep_OP],
#         separation_OP[troughs_sep_OP],
#         'o',
#         fillstyle='none',
#         color='green',
#         label='detected minima'
#     )

# plt.xlabel('t')
# plt.ylabel(r'$x_R(t)-x_L(t)$')
# plt.title('NQS two-dark OP separation')
# plt.legend()
# plt.grid(alpha=0.2)
# plt.show()

# t_traj = np.linspace(0, t_OP[-1], 1000)
# x_right_analytical = +x_eq + delta_OP * np.cos(omega_OP_exact * t_traj)
# x_left_analytical  = -x_eq - delta_OP * np.cos(omega_OP_exact * t_traj)

# plt.figure(figsize=(10, 5))
# plt.plot(t_OP, x_right_OP, label='right dark soliton NQS')
# plt.plot(t_OP, x_left_OP, label='left dark soliton NQS')

# plt.plot(
#     t_traj,
#     x_right_analytical,
#     '--',
#     color='red',
#     label=rf'analytical $\omega_{{OP}}={omega_OP_exact:.4f}$'
# )

# plt.plot(
#     t_traj,
#     x_left_analytical,
#     '--',
#     color='red'
# )

# plt.axhline(+x_eq, linestyle=':', color='gray')
# plt.axhline(-x_eq, linestyle=':', color='gray')

# plt.xlabel('t')
# plt.ylabel('dark soliton position')
# plt.title('NQS OP oscillation vs analytical prediction')
# plt.legend()
# plt.grid(alpha=0.2)
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(t_OP, norm_OP)
# plt.xlabel('t')
# plt.ylabel('Norm')
# plt.title('NQS OP norm conservation')
# plt.grid(alpha=0.2)
# plt.show()

# if hasattr(OP_evo, 'energy') and OP_evo.energy is not None:
#     plt.figure(figsize=(8, 5))
#     plt.plot(t_OP, np.real(OP_evo.energy))
#     plt.xlabel('t')
#     plt.ylabel('Energy')
#     plt.title('NQS OP energy')
#     plt.grid(alpha=0.2)
#     plt.show()
# %%

# %%
# print('Phase-imprinted state: target construction + NQS fit')

# # --------------------------------------------------
# # cargar el ground state normal
# # --------------------------------------------------
# model_gs = models.NQS(input_size, output_size, hidden_layers).to(device)
# model_gs.load_state_dict(torch.load(gs_model_path, map_location=device))
# model_gs.to(device)
# model_gs.eval()

# # asegurarnos de que NO haya phase step activo en el forward
# pm.use_phase_step = False
# pm.phase = 0.0
# pm.enforce_even_parity = False
# pm.enforce_odd_parity = False
# pm.k = 0.0

# x_tensor = grid.x()
# x = grid.mesh.detach().cpu().numpy() if torch.is_tensor(grid.mesh) else np.array(grid.mesh)

# # --------------------------------------------------
# # evaluar el GS normal en la malla
# # --------------------------------------------------
# with torch.no_grad():
#     psi_gs_torch = torch.exp(model_gs(x_tensor)).squeeze()

# psi_gs = psi_gs_torch.detach().cpu().numpy()

# # --------------------------------------------------
# # construir el estado objetivo con diferencia de fase pi
# # phi(x) pasa suavemente de 0 a pi en el origen
# # --------------------------------------------------
# phase_width_target = 1.0

# phi = 0.5 * np.pi * (1.0 + np.tanh((x - 0.0) / phase_width_target))
# psi_target = psi_gs * np.exp(1j * phi)

# # --------------------------------------------------
# # crear una nueva NQS para ajustar ese estado objetivo
# # la inicializamos desde el GS, para que parta cerca
# # --------------------------------------------------
# model_fit = models.NQS(input_size, output_size, hidden_layers).to(device)
# model_fit.load_state_dict(torch.load(gs_model_path, map_location=device))
# model_fit.to(device)
# model_fit.train()

# # target en torch complejo
# psi_target_torch = torch.tensor(psi_target, dtype=torch.complex128, device=device)

# # --------------------------------------------------
# # ajuste supervisado simple del estado con phase imprinting
# # --------------------------------------------------
# optimizer = torch.optim.Adam(model_fit.parameters(), lr=1e-3)

# n_epochs = 40000
# print_every = 5000

# loss_history = []

# for epoch in range(n_epochs):
#     optimizer.zero_grad()

#     psi_pred = torch.exp(model_fit(x_tensor)).squeeze()

#     # pérdida compleja L2 sobre la función de onda
#     loss = torch.mean(torch.abs(psi_pred - psi_target_torch)**2)

#     loss.backward()
#     optimizer.step()

#     loss_history.append(loss.item())

#     if epoch % print_every == 0:
#         print(f'Epoch {epoch:6d} | loss = {loss.item():.6e}')

# print(f'Final fit loss = {loss_history[-1]:.6e}')

# model_fit.eval()

# # --------------------------------------------------
# # guardar el modelo ajustado al estado con phase imprinting
# # --------------------------------------------------
# phase_fit_model_path = utils.file_ID(
#     pm.data_dir,
#     file_name(pm.architecture, model_fit.architecture, 'phase_imprinted_fit')
#     + f'_g_{pm.g}_w_{pm.w}',
#     'pt'
# )
# torch.save(model_fit.state_dict(), phase_fit_model_path)
# print(f'Phase-imprinted fitted model saved to: {phase_fit_model_path}')
# # %%
# # comparar el target con la NQS ajustada
# with torch.no_grad():
#     psi_fit_torch = torch.exp(model_fit(x_tensor)).squeeze()

# psi_fit = psi_fit_torch.detach().cpu().numpy()

# den_gs = np.abs(psi_gs)**2
# den_target = np.abs(psi_target)**2
# den_fit = np.abs(psi_fit)**2

# phase_target = np.unwrap(np.angle(psi_target))
# phase_fit = np.unwrap(np.angle(psi_fit))

# # 1) densidad GS vs target vs fit
# plt.figure(figsize=(8,5))
# plt.plot(x, den_gs, label='GS density')
# plt.plot(x, den_target, '--', label='target density')
# plt.plot(x, den_fit, ':', label='fitted NQS density')
# plt.axvline(0.0, color='k', linestyle=':', label='x=0')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi(x)|^2$')
# plt.title('Ground state and phase-imprinted target')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # 2) fase target vs fit
# plt.figure(figsize=(8,5))
# plt.plot(x, phase_target, label='target phase')
# plt.plot(x, phase_fit, '--', label='fitted NQS phase')
# plt.axvline(0.0, color='k', linestyle=':', label='x=0')
# plt.xlabel('x')
# plt.ylabel(r'$\arg[\psi(x)]$')
# plt.title('Phase-imprinted target: phase profile')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # 3) parte real e imaginaria del target y del fit
# plt.figure(figsize=(8,5))
# plt.plot(x, psi_target.real, label='Re(target)')
# plt.plot(x, psi_target.imag, '--', label='Im(target)')
# plt.axvline(0.0, color='k', linestyle=':')
# plt.xlabel('x')
# plt.ylabel(r'$\psi_{\rm target}(x)$')
# plt.title('Target wavefunction with phase imprinting')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(x, psi_fit.real, label='Re(fit)')
# plt.plot(x, psi_fit.imag, '--', label='Im(fit)')
# plt.axvline(0.0, color='k', linestyle=':')
# plt.xlabel('x')
# plt.ylabel(r'$\psi_{\rm fit}(x)$')
# plt.title('Fitted NQS wavefunction')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # 4) pérdida del ajuste
# plt.figure(figsize=(8,5))
# plt.plot(loss_history)
# plt.xlabel('epoch')
# plt.ylabel('L2 loss')
# plt.title('Fit loss history')
# plt.grid(alpha=0.2)
# plt.show()
# # %%
# print('Real-time evolution from fitted phase-imprinted state')

# # cargar el modelo ya ajustado al estado con imprinting
# model_dyn = models.NQS(input_size, output_size, hidden_layers).to(device)
# model_dyn.load_state_dict(torch.load(phase_fit_model_path, map_location=device))
# model_dyn.to(device)
# model_dyn.eval()

# # --------------------------------------------------
# # parámetros físicos
# # --------------------------------------------------
# pm.g = 1.0
# pm.x0 = 0.0
# pm.w = 0.01
# pm.mu = 1.0

# pm.wall = 0.0

# # sin barrera al principio
# pm.gauss_amplitude = 0.0
# pm.gauss_width = 1.0
# pm.gauss_x0 = 0.0

# # sin kick
# pm.k = 0.0

# # MUY IMPORTANTE:
# # el phase imprinting ya está metido en el estado inicial ajustado,
# # así que ahora lo apagamos
# pm.phase = 0.0
# pm.phase_center = 0.0
# pm.phase_width = 1.0
# pm.use_phase_step = False

# pm.enforce_even_parity = False
# pm.enforce_odd_parity = False

# # --------------------------------------------------
# # tiempo real
# # --------------------------------------------------
# pm.dt = 0.005
# pm.t_max = 10.0

# pm.evolution = 'real'
# pm.lambda_reg = 1e-3

# file_path_real = utils.file_ID(
#     pm.data_dir,
#     file_name(pm.architecture, model_dyn.architecture, 'real_from_phase_imprinted_fit')
#     + f'_g_{pm.g}_w_{pm.w}',
#     pm.data_format
# )

# integrator(model_dyn, grid, file_path=file_path_real)

# # leer dinámica
# real_evo = Dynamics(file_path=file_path_real, x_grid=grid.x())
# psi = real_evo.psi
# den = np.abs(psi)**2
# params = real_evo.get_params()

# # figura evolución
# fig_path = utils.file_ID(
#     pm.figs_dir,
#     file_name(pm.architecture, model_dyn.architecture, 'real_from_phase_imprinted_fit')
#     + f'_g_{pm.g}_w_{pm.w}',
#     pm.fig_format
# )
# plots.evo_fig_params_poster(real_evo.t_grid, grid.mesh, den.T, params, fig_path=fig_path)
# # %%
# x = grid.mesh.detach().cpu().numpy() if torch.is_tensor(grid.mesh) else np.array(grid.mesh)
# t = real_evo.t_grid

# psi0 = psi[0, :]
# psif = psi[-1, :]
# den0 = den[0, :]
# denf = den[-1, :]

# phase0 = np.unwrap(np.angle(psi0))
# phasef = np.unwrap(np.angle(psif))

# # 1) densidad inicial y final
# plt.figure(figsize=(8,5))
# plt.plot(x, den0, label='t = 0')
# plt.plot(x, denf, label=rf't = {t[-1]:.2f}')
# plt.axvline(0.0, color='k', linestyle=':', label='x=0')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi(x)|^2$')
# plt.title('Density before and after real-time evolution')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # 2) fase inicial y final
# plt.figure(figsize=(8,5))
# plt.plot(x, phase0, label='phase at t=0')
# plt.plot(x, phasef, label=rf'phase at t={t[-1]:.2f}')
# plt.axvline(0.0, color='k', linestyle=':', label='x=0')
# plt.xlabel('x')
# plt.ylabel(r'$\arg[\psi(x)]$')
# plt.title('Phase profile')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # 3) función de onda final
# plt.figure(figsize=(8,5))
# plt.plot(x, psif.real, label='Re(psi)')
# plt.plot(x, psif.imag, '--', label='Im(psi)')
# plt.axvline(0.0, color='k', linestyle=':')
# plt.xlabel('x')
# plt.ylabel(r'$\psi(x)$')
# plt.title('Final wavefunction')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # 4) zoom central de la densidad
# plt.figure(figsize=(8,5))
# mask_center = np.abs(x) < 10.0
# plt.plot(x[mask_center], denf[mask_center], label='final density')
# plt.axvline(0.0, color='k', linestyle=':', label='x=0')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi(x)|^2$')
# plt.title('Central density notch')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()
# %%
# print('Dynamics')

# # trap
# pm.x0 = 0
# pm.w = 0.1

# pm.mu = 1*0

# pm.gauss_amplitude = 5

# pm.k = 0.5*0
# pm.phase = torch.pi*0

# # Time parameters
# pm.dt = 0.1
# pm.t_max = 50

# # Integrator parameters
# pm.evolution = 'real'
# pm.lambda_reg = (1*0 + 1j)*1e-2

# # Perform imag time evolution
# file_path = utils.file_ID(pm.data_dir,
#                         file_name(pm.architecture, model.architecture, pm.evolution) + f'g_{pm.g}',
#                         pm.data_format)
# integrator(model, grid, file_path=file_path)

# # Get the dynamics
# real_evo = Dynamics(file_path=file_path, x_grid=grid.x())
# # Compute density
# psi = real_evo.psi
# den = np.abs(real_evo.psi)**2
# # Get parameters
# params = real_evo.get_params()

# # Plot data
# fig_path = utils.file_ID(pm.figs_dir,
#                         file_name(pm.architecture, model.architecture, pm.evolution) + f'_g_{pm.g}',
#                         pm.fig_format)
# plots.evo_fig_params_poster(real_evo.t_grid, grid.mesh, den.T, params, fig_path=fig_path)

# # %%

# norm = np.trapz(den[-1,:], mesh)
# plt.plot(mesh, den[-1,:], label=f'g={str(g)}, w={str(w)}')
# x_max =grid.mesh[np.where(den[-1,:]==den[-1,:].max())]
# plt.vlines(x_max,0,den[-1,:].max(),linestyles=':',label=f'{round(x_max.item(),3)}')
# plt.legend()
# plt.show()

# # %%
# # BLOQUE 2: GS con pared + benchmark tanh/healing length
# # ======================================================================
# print('Ground State Search — wall benchmark')

# # Malla específica para el benchmark con pared
# grid_wall = utils.PointGrid(N=2**9, start=-24, end=24, device=device)

# model_wall = models.NQS(input_size, output_size, hidden_layers).to(device)

# # Parámetros físicos
# pm.g = 1.0
# pm.w = 0.0
# pm.x0 = 0.0
# pm.mu = 1.0
# pm.k = 0.0
# pm.gauss_amplitude = 0.0
# pm.gauss_width = 0.1
# pm.gauss_x0 = 0.0
# pm.wall = 80.0   # barrera alta en x > 0 con tu Hamiltoniano actual

# # Sin simetrías impuestas en el benchmark de pared
# pm.enforce_even_parity = False
# pm.enforce_odd_parity = False

# # Tiempo imaginario
# pm.evolution = 'imag'
# pm.dt = 0.01
# pm.t_max = 150
# pm.lambda_reg = 1e-2
# pm.e_error = 1e-8

# file_path_wall = utils.file_ID(
#     pm.data_dir,
#     file_name(pm.architecture, model_wall.architecture, 'imag_wall')
#     + f'_g_{pm.g}_mu_{pm.mu}_wall_{pm.wall}',
#     pm.data_format
# )

# integrator(model_wall, grid_wall, file_path=file_path_wall)

# wall_evo = Dynamics(file_path=file_path_wall, x_grid=grid_wall.x())
# psi_wall = wall_evo.psi
# den_wall = np.abs(psi_wall)**2
# params_wall = wall_evo.get_params()

# psi_final = psi_wall[-1]
# den_final = den_wall[-1]
# x = grid_wall.mesh.detach().cpu().numpy() if torch.is_tensor(grid_wall.mesh) else np.array(grid_wall.mesh)

# # Con tu implementación actual de wall, la pared actúa en x > 0,
# # así que la región permitida es x < 0.
# s_half, psi_half = benchmarks.extract_allowed_half(
#     x, psi_final, wall_position=0.0, allowed_side='left'
# )

# amp_half = np.abs(psi_half)
# rho_half = amp_half**2

# psi_exact = benchmarks.dark_wall_profile(s_half, mu=pm.mu, g=pm.g)
# rho_exact = benchmarks.dark_wall_density_profile(s_half, mu=pm.mu, g=pm.g)

# (A_fit, xi_fit), _ = benchmarks.fit_dark_wall_profile(
#     s_half,
#     amp_half,
#     A0=benchmarks.bulk_amplitude(pm.mu, pm.g),
#     xi0=benchmarks.healing_length(pm.mu)
# )

# # --------------------------------------------------
# # FIT LOCAL cerca de la pared
# # --------------------------------------------------
# mask_fit = s_half < 4.0
# s_fit = s_half[mask_fit]
# amp_fit_data = amp_half[mask_fit]

# (A_fit_local, xi_fit_local), _ = benchmarks.fit_dark_wall_profile(
#     s_fit,
#     amp_fit_data,
#     A0=benchmarks.bulk_amplitude(pm.mu, pm.g),
#     xi0=benchmarks.healing_length(pm.mu)
# )
# # --------------------------------------------------
# # FIT con A fija y solo xi libre
# # --------------------------------------------------
# A_exact = benchmarks.bulk_amplitude(pm.mu, pm.g)

# def tanh_xi_only(s, xi):
#     return A_exact * np.tanh(s / (np.sqrt(2.0) * xi))

# # fit global con A fija
# xi_fit_fixedA, _ = curve_fit(
#     tanh_xi_only,
#     s_half,
#     amp_half,
#     p0=[benchmarks.healing_length(pm.mu)]
# )
# xi_fit_fixedA = xi_fit_fixedA[0]

# # fit local con A fija
# xi_fit_fixedA_local, _ = curve_fit(
#     tanh_xi_only,
#     s_fit,
#     amp_fit_data,
#     p0=[benchmarks.healing_length(pm.mu)]
# )
# xi_fit_fixedA_local = xi_fit_fixedA_local[0]

# xi_exact = benchmarks.healing_length(pm.mu)

# err_amp = benchmarks.relative_l2_error_profile(amp_half, psi_exact, s_half)
# err_rho = benchmarks.relative_l2_error_profile(rho_half, rho_exact, s_half)

# print('[Dark wall benchmark]')
# # print(f'  A_exact              = {A_exact:.6e}')
# # print(f'  A_fit                = {A_fit:.6e}')
# # print(f'  A_fit_local          = {A_fit_local:.6e}')
# # print(f'  xi_exact             = {xi_exact:.6e}')
# # print(f'  xi_fit               = {xi_fit:.6e}')
# # print(f'  xi_fit_local         = {xi_fit_local:.6e}')
# # print(f'  xi_fit_fixedA        = {xi_fit_fixedA:.6e}')
# # print(f'  xi_fit_fixedA_local  = {xi_fit_fixedA_local:.6e}')
# # print(f'  |xi_fit - xi_exact|              = {abs(xi_fit - xi_exact):.6e}')
# # print(f'  |xi_fit_local - xi_exact|        = {abs(xi_fit_local - xi_exact):.6e}')
# # print(f'  |xi_fit_fixedA - xi_exact|       = {abs(xi_fit_fixedA - xi_exact):.6e}')
# # print(f'  |xi_fit_fixedA_local - xi_exact| = {abs(xi_fit_fixedA_local - xi_exact):.6e}')
# print(f'  Relative L2 error in |psi| = {err_amp:.6e}')
# print(f'  Relative L2 error in rho   = {err_rho:.6e}')

# fig_path_wall = utils.file_ID(
#     pm.figs_dir,
#     file_name(pm.architecture, model_wall.architecture, 'imag_wall')
#     + f'_g_{pm.g}_mu_{pm.mu}_wall_{pm.wall}',
#     pm.fig_format
# )
# plots.evo_fig_params_poster(
#     wall_evo.t_grid,
#     grid_wall.mesh,
#     den_wall.T,
#     params_wall,
#     fig_path=fig_path_wall
# )

# # Plot estado final completo
# plt.figure(figsize=(8,5))
# plt.plot(x, den_final, label='numérico')
# plt.axvline(0.0, color='k', linestyle=':', label='pared en x=0')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi(x)|^2$')
# plt.title('Estado final con pared')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # Plot amplitud benchmark
# plt.figure(figsize=(8,5))
# plt.plot(s_half, amp_half, label=r'numérico $|\psi|$')
# plt.plot(s_half, psi_exact, '--', label=r'$\psi_0 \tanh(s/\sqrt{2}\xi)$')
# plt.plot(
#     s_half,
#     benchmarks.tanh_profile_for_fit(s_half, A_fit, xi_fit),
#     ':',
#     label=rf'fit tanh ($\xi_{{fit}}={xi_fit:.4f}$)'
# )
# plt.xlabel(r'$s$')
# plt.ylabel(r'$|\psi|$')
# plt.title('Benchmark wall: amplitud')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # Plot densidad benchmark
# plt.figure(figsize=(8,5))
# plt.plot(s_half, rho_half, label=r'numérico $|\psi|^2$')
# plt.plot(s_half, rho_exact, '--', label=r'analítico')
# plt.xlabel(r'$s$')
# plt.ylabel(r'$|\psi|^2$')
# plt.title('Benchmark wall: densidad')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # #Plot final
# # plt.figure(figsize=(8,5))
# # plt.plot(s_half, amp_half, label=r'numérico $|\psi|$')
# # plt.plot(s_half, psi_exact, '--', label=r'$\psi_0 \tanh(s/\sqrt{2}\xi)$')

# # plt.plot(
# #     s_half,
# #     benchmarks.tanh_profile_for_fit(s_half, A_fit, xi_fit),
# #     ':',
# #     label=rf'fit global ($\xi_{{fit}}={xi_fit:.4f}$)'
# # )

# # plt.plot(
# #     s_half,
# #     benchmarks.tanh_profile_for_fit(s_half, A_fit_local, xi_fit_local),
# #     '-.',
# #     label=rf'fit local ($\xi_{{fit,loc}}={xi_fit_local:.4f}$)'
# # )

# # plt.plot(
# #     s_half,
# #     tanh_xi_only(s_half, xi_fit_fixedA),
# #     '--',
# #     linewidth=1.5,
# #     label=rf'fit $A$ fija ($\xi_{{fixA}}={xi_fit_fixedA:.4f}$)'
# # )

# # plt.plot(
# #     s_half,
# #     tanh_xi_only(s_half, xi_fit_fixedA_local),
# #     '-',
# #     linewidth=1.2,
# #     label=rf'fit local $A$ fija ($\xi_{{fixA,loc}}={xi_fit_fixedA_local:.4f}$)'
# # )

# # plt.axvline(4.0, color='gray', linestyle='--', alpha=0.7, label='corte fit local')
# # plt.xlabel(r'$s$')
# # plt.ylabel(r'$|\psi|$')
# # plt.title('Benchmark wall: amplitud')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()
# # Guardar state_dict del GS con pared si luego quieres reutilizarlo
# wall_model_path = utils.file_ID(
#     pm.data_dir,
#     file_name(pm.architecture, model_wall.architecture, 'ground_state_wall')
#     + f'_g_{pm.g}_mu_{pm.mu}_wall_{pm.wall}',
#     'pt'
# )
# torch.save(model_wall.state_dict(), wall_model_path)
# print(f'  GS con pared guardado en: {wall_model_path}')

# #%%
# # BLOQUE 3: Ground state del dark soliton centrado con continuación de caja
# # ======================================================================
# print('Ground State Search — centered dark soliton with box continuation')

# # Lista de semianchos de caja
# box_list = [170]

# # La primera semilla es el estado con pared
# prev_model_path = wall_model_path

# # Variables para quedarnos con el último caso exitoso
# last_successful_L = None
# dark_evo = None
# psi_dark = None
# den_dark = None
# params_dark = None
# dark_model_path = None
# grid_dark = None

# for L in box_list:
#     print(f'\n=== Running dark soliton on box [-{L}, {L}] ===')

#     # Malla específica para esta etapa
#     grid_dark = utils.PointGrid(N=2**9, start=-L, end=L, device=device)

#     # Nuevo modelo para esta etapa
#     model_dark = models.NQS(input_size, output_size, hidden_layers).to(device)

#     # Cargar la semilla desde la etapa anterior
#     model_dark.load_state_dict(torch.load(prev_model_path, map_location=device))
#     model_dark.to(device)
#     model_dark.eval()

#     # --------------------------
#     # parámetros físicos
#     # --------------------------
#     pm.g = 1.0
#     pm.w = 0.01
#     pm.x0 = 0.0
#     pm.mu = 1.0

#     # sin pared
#     pm.wall = 0.0

#     # sin gaussiana
#     pm.gauss_amplitude = 0.0
#     pm.gauss_width = 1.0
#     pm.gauss_x0 = 0.0

#     # sin kick
#     pm.k = 0.0

#     # imponer imparidad para forzar el nodo en x=0
#     pm.enforce_even_parity = False
#     pm.enforce_odd_parity = True

#     # --------------------------
#     # tiempo imaginario
#     # --------------------------
#     pm.evolution = 'imag'
#     pm.dt = 0.02
#     pm.t_max = 100
#     pm.lambda_reg = 1e-3
#     pm.e_error = 1e-8

#     # --------------------------------------------------
#     # DIAGNÓSTICO PREVIO: comprobar que la semilla es finita
#     # --------------------------------------------------
#     x_test = grid_dark.x()
#     with torch.no_grad():
#         lnpsi_test = model_dark(x_test)
#         psi_test = torch.exp(lnpsi_test)

#     finite_lnpsi = torch.isfinite(lnpsi_test).all().item()
#     finite_psi = torch.isfinite(psi_test).all().item()

#     print("finite lnpsi :", finite_lnpsi)
#     print("finite psi   :", finite_psi)

#     if finite_psi:
#         print("min |psi|    :", torch.min(torch.abs(psi_test)).item())
#         print("max |psi|    :", torch.max(torch.abs(psi_test)).item())

#     # Si la semilla ya no es finita en esta caja, paramos aquí
#     if (not finite_lnpsi) or (not finite_psi):
#         print(f'La semilla deja de ser válida en la caja [-{L}, {L}].')
#         print('Se detiene la continuación aquí.')
#         break

#     # --------------------------------------------------
#     # INTEGRACIÓN
#     # --------------------------------------------------
#     file_path_dark = utils.file_ID(
#         pm.data_dir,
#         file_name(pm.architecture, model_dark.architecture, 'imag_dark_centered')
#         + f'_g_{pm.g}_w_{pm.w}_L_{L}',
#         pm.data_format
#     )

#     integrator(model_dark, grid_dark, file_path=file_path_dark)

#     # leer dinámica
#     dark_evo = Dynamics(file_path=file_path_dark, x_grid=grid_dark.x())
#     psi_dark = dark_evo.psi
#     den_dark = np.abs(psi_dark)**2
#     params_dark = dark_evo.get_params()

#     # guardar modelo final de esta etapa
#     dark_model_path = utils.file_ID(
#         pm.data_dir,
#         file_name(pm.architecture, model_dark.architecture, 'ground_state_dark_centered')
#         + f'_g_{pm.g}_w_{pm.w}_L_{L}',
#         'pt'
#     )
#     torch.save(model_dark.state_dict(), dark_model_path)
#     print(f'Dark-soliton GS guardado en: {dark_model_path}')

#     # esta será la semilla de la siguiente caja
#     prev_model_path = dark_model_path
#     last_successful_L = L

#     # --------------------------------------------------
#     # PLOTS DE ESTA ETAPA
#     # --------------------------------------------------
#     import matplotlib.pyplot as plt

#     x = grid_dark.mesh.detach().cpu().numpy() if torch.is_tensor(grid_dark.mesh) else np.array(grid_dark.mesh)
#     psi_f = psi_dark[-1, :]
#     den_f = den_dark[-1, :]

#     # figura evolución
#     fig_path_dark = utils.file_ID(
#         pm.figs_dir,
#         file_name(pm.architecture, model_dark.architecture, 'imag_dark_centered')
#         + f'_g_{pm.g}_w_{pm.w}_L_{L}',
#         pm.fig_format
#     )
#     plots.evo_fig_params_poster(
#         dark_evo.t_grid,
#         grid_dark.mesh,
#         den_dark.T,
#         params_dark,
#         fig_path=fig_path_dark
#     )

#     # densidad final
#     plt.figure(figsize=(8,5))
#     plt.plot(x, den_f, label=f'densidad final, L={L}')
#     plt.axvline(0.0, color='k', linestyle=':', label='x=0')
#     plt.xlabel('x')
#     plt.ylabel(r'$|\psi(x)|^2$')
#     plt.title(f'Ground state del dark soliton, caja [-{L},{L}]')
#     plt.grid(alpha=0.2)
#     plt.legend()
#     plt.show()

#     # parte real e imaginaria final
#     plt.figure(figsize=(8,5))
#     plt.plot(x, psi_f.real, label='Re(psi)')
#     plt.plot(x, psi_f.imag, '--', label='Im(psi)')
#     plt.axvline(0.0, color='k', linestyle=':')
#     plt.xlabel('x')
#     plt.ylabel(r'$\psi(x)$')
#     plt.title(f'Función de onda final, caja [-{L},{L}]')
#     plt.grid(alpha=0.2)
#     plt.legend()
#     plt.show()

#     # chequeo de imparidad
#     plt.figure(figsize=(8,5))
#     plt.plot(x, psi_f.real, label=r'Re$\psi(x)$')
#     plt.plot(x, -psi_f[::-1].real, '--', label=r'$-\mathrm{Re}\,\psi(-x)$')
#     plt.axvline(0.0, color='k', linestyle=':')
#     plt.xlabel('x')
#     plt.ylabel(r'Re$\psi$')
#     plt.title(f'Chequeo de imparidad, caja [-{L},{L}]')
#     plt.grid(alpha=0.2)
#     plt.legend()
#     plt.show()

#     # mínimo central
#     plt.figure(figsize=(8,5))
#     plt.plot(x, den_f, label='densidad final')
#     plt.axvline(0.0, color='k', linestyle=':', label='x=0')
#     mask_center = np.abs(x) < 20.0
#     x_center = x[mask_center]
#     den_center = den_f[mask_center]
#     x_min = x_center[np.argmin(den_center)]
#     plt.axvline(x_min, color='r', linestyle='--', label=rf'mínimo central en x={x_min:.3f}')
#     plt.xlabel('x')
#     plt.ylabel(r'$|\psi(x)|^2$')
#     plt.title(f'Mínimo central del dark soliton, caja [-{L},{L}]')
#     plt.grid(alpha=0.2)
#     plt.legend()
#     plt.show()

# print('\n==========================================')
# print('Continuación de caja terminada')
# print(f'Última caja convergida: {last_successful_L}')
# print(f'Último modelo guardado: {dark_model_path}')
# print('==========================================')
# #%%
# # MÉTRICAS DEL DARK SOLITON
# # ======================================================================
# import numpy as np
# import matplotlib.pyplot as plt

# x = grid_dark.mesh.detach().cpu().numpy() if torch.is_tensor(grid_dark.mesh) else np.array(grid_dark.mesh)
# psi_f = psi_dark[-1, :]
# den_f = den_dark[-1, :]

# # --------------------------
# # 1) Mínimo central
# # --------------------------
# mask_center = np.abs(x) < 20.0
# x_center = x[mask_center]
# den_center = den_f[mask_center]

# idx_min = np.argmin(den_center)
# x_min = x_center[idx_min]
# rho_min = den_center[idx_min]

# # --------------------------
# # 2) Fondo local
# # Tomamos una región intermedia, lejos del notch y lejos de los bordes
# # --------------------------
# mask_bg = (np.abs(x) > 20.0) & (np.abs(x) < 60.0)
# rho_bg = np.mean(den_f[mask_bg])

# # Contraste del notch
# contrast = 1.0 - rho_min / rho_bg if rho_bg != 0 else np.nan

# # --------------------------
# # 3) Anchura del notch a media profundidad
# # Nivel intermedio entre rho_min y rho_bg
# # --------------------------
# rho_half = 0.5 * (rho_min + rho_bg)

# # buscamos cruces en la región central
# mask_half = np.abs(x) < 20.0
# x_half = x[mask_half]
# den_half = den_f[mask_half]

# below = den_half <= rho_half
# indices = np.where(below)[0]

# if len(indices) >= 2:
#     x_left = x_half[indices[0]]
#     x_right = x_half[indices[-1]]
#     notch_width = x_right - x_left
# else:
#     x_left = np.nan
#     x_right = np.nan
#     notch_width = np.nan

# # --------------------------
# # 4) Imparidad de Re(psi)
# # --------------------------
# repsi = psi_f.real
# odd_error = np.linalg.norm(repsi + repsi[::-1]) / np.linalg.norm(repsi)

# # --------------------------
# # 5) Paridad de la densidad
# # --------------------------
# even_error = np.linalg.norm(den_f - den_f[::-1]) / np.linalg.norm(den_f)

# # --------------------------
# # Mostrar métricas
# # --------------------------
# print("=== Métricas del dark soliton ===")
# print(f"x_min          = {x_min:.6f}")
# print(f"rho_min        = {rho_min:.6e}")
# print(f"rho_bg         = {rho_bg:.6e}")
# print(f"contrast       = {contrast:.6e}")
# print(f"rho_half       = {rho_half:.6e}")
# print(f"notch_width    = {notch_width:.6e}")
# print(f"odd_error      = {odd_error:.6e}")
# print(f"even_error     = {even_error:.6e}")

# # --------------------------
# # Plot zoom central con niveles
# # --------------------------
# plt.figure(figsize=(8,5))
# plt.plot(x_center, den_center, label='densidad central')
# plt.axvline(0.0, color='k', linestyle=':', label='x=0')
# plt.axvline(x_min, color='r', linestyle='--', label=rf'$x_{{min}}={x_min:.3f}$')
# plt.axhline(rho_bg, color='g', linestyle='--', label=rf'$\rho_{{bg}}={rho_bg:.3f}$')
# plt.axhline(rho_half, color='m', linestyle=':', label=rf'$\rho_{{half}}={rho_half:.3f}$')
# if np.isfinite(x_left):
#     plt.axvline(x_left, color='gray', linestyle='--')
#     plt.axvline(x_right, color='gray', linestyle='--', label=rf'width={notch_width:.3f}')
# plt.xlabel('x')
# plt.ylabel(r'$|\psi(x)|^2$')
# plt.title('Zoom central y métricas del notch')
# plt.grid(alpha=0.2)
# plt.legend()
# plt.show()

# # %%
# # print('Wall benchmark sweep in mu')

# # # ---------------------------
# # # Lista de valores de mu
# # # ---------------------------
# # mu_list = [0.5, 0.75, 1.0, 1.5]



# # # ---------------------------
# # # Parámetros físicos fijos
# # # ---------------------------
# # pm.g = 1.0
# # pm.w = 0.0
# # pm.x0 = 0.0
# # pm.k = 0.0
# # pm.phase = 0.0
# # pm.gauss_amplitude = 0.0
# # pm.gauss_width = 0.1
# # pm.wall = 80.0

# # # ---------------------------
# # # Tiempo imaginario
# # # ---------------------------
# # pm.evolution = 'imag'
# # pm.dt = 0.01
# # pm.t_max = 150
# # pm.lambda_reg = 1e-3
# # pm.e_error = 1e-8

# # # ---------------------------
# # # Contenedores de resultados
# # # ---------------------------
# # results = []

# # # Para guardar algunos perfiles si luego quieres compararlos
# # profiles = []

# # for mu_val in mu_list:
# #     print(f'\n--- Running wall benchmark for mu = {mu_val:.3f} ---')

# #     # Crear un modelo nuevo para cada mu
# #     model_wall = models.NQS(input_size, output_size, hidden_layers).to(device)

# #     # Parámetro barrido
# #     pm.mu = float(mu_val)

# #     # Nombre de archivo
# #     file_path_wall = utils.file_ID(
# #         pm.data_dir,
# #         file_name(pm.architecture, model_wall.architecture, 'imag_wall')
# #         + f'_g_{pm.g}_mu_{pm.mu}_wall_{pm.wall}',
# #         pm.data_format
# #     )

# #     # Integración
# #     integrator(model_wall, grid, file_path=file_path_wall)

# #     # Dinámica
# #     wall_evo = Dynamics(file_path=file_path_wall, x_grid=grid.x())
# #     psi_wall = wall_evo.psi
# #     den_wall = np.abs(psi_wall)**2

# #     psi_final = psi_wall[-1]
# #     den_final = den_wall[-1]
# #     x = grid.mesh

# #     # Región permitida: x < 0  -> s = distancia a la pared
# #     s_half, psi_half = benchmarks.extract_allowed_half(
# #         x, psi_final, wall_position=0.0, allowed_side='left'
# #     )

# #     amp_half = np.abs(psi_half)
# #     rho_half = amp_half**2

# #     # Solución analítica
# #     psi_exact = benchmarks.dark_wall_profile(s_half, mu=pm.mu, g=pm.g)
# #     rho_exact = benchmarks.dark_wall_density_profile(s_half, mu=pm.mu, g=pm.g)

# #     # Fit
# #     (A_fit, xi_fit), _ = benchmarks.fit_dark_wall_profile(
# #         s_half,
# #         amp_half,
# #         A0=benchmarks.bulk_amplitude(pm.mu, pm.g),
# #         xi0=benchmarks.healing_length(pm.mu)
# #     )

# #     # Valores teóricos
# #     A_exact = benchmarks.bulk_amplitude(pm.mu, pm.g)
# #     xi_exact = benchmarks.healing_length(pm.mu)

# #     # Errores
# #     err_amp = benchmarks.relative_l2_error_profile(amp_half, psi_exact, s_half)
# #     err_rho = benchmarks.relative_l2_error_profile(rho_half, rho_exact, s_half)

# #     # ----------------------------------------------------------------------
# #     # PLOTS INDIVIDUALES PARA CADA mu
# #     # ----------------------------------------------------------------------

# #     # Plot estado final completo
# #     plt.figure(figsize=(8,5))
# #     plt.plot(x, den_final, label='numérico')
# #     plt.axvline(0.0, color='k', linestyle=':', label='pared en x=0')
# #     plt.xlabel('x')
# #     plt.ylabel(r'$|\psi(x)|^2$')
# #     plt.title(rf'Estado final con pared ($\mu={pm.mu}$)')
# #     plt.grid(alpha=0.2)
# #     plt.legend()
# #     plt.show()

# #     # Plot amplitud benchmark
# #     plt.figure(figsize=(8,5))
# #     plt.plot(s_half, amp_half, label=r'numérico $|\psi|$')
# #     plt.plot(s_half, psi_exact, '--', label=r'$\psi_0 \tanh(s/\sqrt{2}\xi)$')
# #     plt.plot(
# #         s_half,
# #         benchmarks.tanh_profile_for_fit(s_half, A_fit, xi_fit),
# #         ':',
# #         label=rf'fit tanh ($\xi_{{fit}}={xi_fit:.4f}$)'
# #     )
# #     plt.xlabel(r'$s$')
# #     plt.ylabel(r'$|\psi|$')
# #     plt.title(rf'Benchmark wall: amplitud ($\mu={pm.mu}$)')
# #     plt.grid(alpha=0.2)
# #     plt.legend()
# #     plt.show()

# #     # Plot densidad benchmark
# #     plt.figure(figsize=(8,5))
# #     plt.plot(s_half, rho_half, label=r'numérico $|\psi|^2$')
# #     plt.plot(s_half, rho_exact, '--', label=r'analítico')
# #     plt.xlabel(r'$s$')
# #     plt.ylabel(r'$|\psi|^2$')
# #     plt.title(rf'Benchmark wall: densidad ($\mu={pm.mu}$)')
# #     plt.grid(alpha=0.2)
# #     plt.legend()
# #     plt.show()

# #     xi_abs_err = abs(xi_fit - xi_exact)
# #     xi_rel_err = xi_abs_err / abs(xi_exact)

# #     A_abs_err = abs(A_fit - A_exact)
# #     A_rel_err = A_abs_err / abs(A_exact)

# #     # Guardar resultados numéricos
# #     results.append({
# #         'mu': pm.mu,
# #         'g': pm.g,
# #         'A_exact': A_exact,
# #         'A_fit': A_fit,
# #         'A_abs_err': A_abs_err,
# #         'A_rel_err': A_rel_err,
# #         'xi_exact': xi_exact,
# #         'xi_fit': xi_fit,
# #         'xi_abs_err': xi_abs_err,
# #         'xi_rel_err': xi_rel_err,
# #         'err_amp_L2': err_amp,
# #         'err_rho_L2': err_rho
# #     })

# #     # Guardar perfiles por si quieres compararlos después
# #     profiles.append({
# #         'mu': pm.mu,
# #         's_half': s_half.copy(),
# #         'amp_half': amp_half.copy(),
# #         'rho_half': rho_half.copy(),
# #         'psi_exact': psi_exact.copy(),
# #         'rho_exact': rho_exact.copy(),
# #         'den_final': den_final.copy(),
# #         'x': x.clone()
# #     })

# #     # Guardar state_dict por si luego quieres reutilizarlo
# #     wall_model_path = utils.file_ID(
# #         pm.data_dir,
# #         file_name(pm.architecture, model_wall.architecture, 'ground_state_wall')
# #         + f'_g_{pm.g}_mu_{pm.mu}_wall_{pm.wall}',
# #         'pt'
# #     )
# #     torch.save(model_wall.state_dict(), wall_model_path)

# #     print(f'  A_exact = {A_exact:.6e}')
# #     print(f'  A_fit   = {A_fit:.6e}')
# #     print(f'  xi_exact = {xi_exact:.6e}')
# #     print(f'  xi_fit   = {xi_fit:.6e}')
# #     print(f'  xi_abs_err = {xi_abs_err:.6e}')
# #     print(f'  xi_rel_err = {xi_rel_err:.6e}')
# #     print(f'  err_amp_L2 = {err_amp:.6e}')
# #     print(f'  err_rho_L2 = {err_rho:.6e}')

# # # ----------------------------------------------------------------------
# # # TABLA RESUMEN
# # # ----------------------------------------------------------------------
# # df_results = pd.DataFrame(results)

# # # Ordenar por mu por claridad
# # df_results = df_results.sort_values(by='mu').reset_index(drop=True)

# # print('\n=== Summary table ===')
# # print(df_results.to_string(index=False))

# # # Si quieres redondeada para leerla mejor por pantalla:
# # print('\n=== Summary table (rounded) ===')
# # print(df_results.round({
# #     'mu': 3,
# #     'g': 3,
# #     'A_exact': 6,
# #     'A_fit': 6,
# #     'A_abs_err': 3,
# #     'A_rel_err': 3,
# #     'xi_exact': 6,
# #     'xi_fit': 6,
# #     'xi_abs_err': 3,
# #     'xi_rel_err': 3,
# #     'err_amp_L2': 3,
# #     'err_rho_L2': 3
# # }).to_string(index=False))

# # # Guardar tabla a csv
# # csv_path = utils.file_ID(
# #     pm.data_dir,
# #     file_name(pm.architecture, 'wall_benchmark_mu_sweep'),
# #     'csv'
# # )
# # df_results.to_csv(csv_path, index=False)
# # print(f'\nSummary table saved to: {csv_path}')

# # # ----------------------------------------------------------------------
# # # GRÁFICA 1: xi_fit vs xi_exact
# # # ----------------------------------------------------------------------
# # plt.figure(figsize=(7,5))
# # plt.plot(df_results['mu'], df_results['xi_exact'], 'o-', label=r'$\xi_{\rm th}=1/\sqrt{2\mu}$')
# # plt.plot(df_results['mu'], df_results['xi_fit'], 's--', label=r'$\xi_{\rm fit}$')
# # plt.xlabel(r'$\mu$')
# # plt.ylabel(r'$\xi$')
# # plt.title('Healing length: theory vs fit')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()

# # # ----------------------------------------------------------------------
# # # GRÁFICA 2: xi_fit vs xi_exact con línea y=x
# # # ----------------------------------------------------------------------
# # xi_min = min(df_results['xi_exact'].min(), df_results['xi_fit'].min())
# # xi_max = max(df_results['xi_exact'].max(), df_results['xi_fit'].max())

# # plt.figure(figsize=(6,6))
# # plt.plot(df_results['xi_exact'], df_results['xi_fit'], 'o', label='data')
# # plt.plot([xi_min, xi_max], [xi_min, xi_max], '--', label=r'$y=x$')
# # plt.xlabel(r'$\xi_{\rm th}$')
# # plt.ylabel(r'$\xi_{\rm fit}$')
# # plt.title(r'$\xi_{\rm fit}$ vs $\xi_{\rm th}$')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()

# # # ----------------------------------------------------------------------
# # # GRÁFICA 3: error relativo de xi
# # # ----------------------------------------------------------------------
# # plt.figure(figsize=(7,5))
# # plt.plot(df_results['mu'], df_results['xi_rel_err'], 'o-')
# # plt.xlabel(r'$\mu$')
# # plt.ylabel(r'Relative error in $\xi$')
# # plt.title(r'Relative error of healing length')
# # plt.grid(alpha=0.2)
# # plt.show()
# # %%
# # print('Dark-soliton dynamics')

# # # cargar ground state
# # model = models.NQS(input_size, output_size, hidden_layers).to(device)
# # model.load_state_dict(torch.load(gs_model_path, map_location=device))
# # model.to(device)
# # model.eval()

# # # --------------------------
# # # parámetros físicos
# # # --------------------------
# # pm.g = 1.0
# # pm.x0 = 0.0
# # pm.w = 0.01

# # # muy importante: mantener el mismo mu que en el GS
# # pm.mu = 1.0

# # pm.wall = 0.0

# # # barrera gaussiana opcional para pinchar el notch en el centro
# # # empieza con algo pequeño; si quieres quitarla luego, pon 0.0
# # pm.gauss_amplitude = 0.0
# # pm.gauss_width = 2.0
# # pm.gauss_x0 = 0.0

# # # imprinting de fase
# # pm.k = 0.0
# # pm.phase = np.pi
# # pm.phase_center = 0.0
# # pm.phase_width = 1.0
# # pm.use_phase_step = True

# # # --------------------------
# # # tiempo real
# # # --------------------------
# # pm.dt = 0.01
# # pm.t_max = 50

# # pm.evolution = 'real'
# # pm.lambda_reg = 1e-3 + 1j*1e-3

# # file_path_real = utils.file_ID(
# #     pm.data_dir,
# #     file_name(pm.architecture, model.architecture, 'real_dark')
# #     + f'_g_{pm.g}_w_{pm.w}_phase_{round(float(pm.phase),3)}',
# #     pm.data_format
# # )

# # integrator(model, grid, file_path=file_path_real)

# # # leer dinámica
# # real_evo = Dynamics(file_path=file_path_real, x_grid=grid.x())
# # psi = real_evo.psi
# # den = np.abs(psi)**2
# # params = real_evo.get_params()

# # # figura evolución
# # fig_path = utils.file_ID(
# #     pm.figs_dir,
# #     file_name(pm.architecture, model.architecture, 'real_dark')
# #     + f'_g_{pm.g}_w_{pm.w}_phase_{round(float(pm.phase),3)}',
# #     pm.fig_format
# # )
# # plots.evo_fig_params_poster(real_evo.t_grid, grid.mesh, den.T, params, fig_path=fig_path)

# # # --------------------------------------------------
# # # PLOTS ADICIONALES
# # # --------------------------------------------------
# # import matplotlib.pyplot as plt

# # x = grid.mesh.detach().cpu().numpy() if torch.is_tensor(grid.mesh) else np.array(grid.mesh)
# # t = real_evo.t_grid

# # psi0 = psi[0, :]
# # psif = psi[-1, :]
# # den0 = den[0, :]
# # denf = den[-1, :]

# # phase0 = np.unwrap(np.angle(psi0))
# # phasef = np.unwrap(np.angle(psif))

# # # 1) Densidad inicial y final
# # plt.figure(figsize=(8,5))
# # plt.plot(x, den0, label='t = 0')
# # plt.plot(x, denf, label=rf't = {t[-1]:.2f}')
# # plt.axvline(0.0, color='k', linestyle=':', label='centro')
# # plt.xlabel('x')
# # plt.ylabel(r'$|\psi(x)|^2$')
# # plt.title('Densidad: estado inicial y final')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()

# # # 2) Parte real e imaginaria inicial
# # plt.figure(figsize=(8,5))
# # plt.plot(x, psi0.real, label=r'Re$[\psi(x,0)]$')
# # plt.plot(x, psi0.imag, '--', label=r'Im$[\psi(x,0)]$')
# # plt.axvline(0.0, color='k', linestyle=':', label='centro')
# # plt.xlabel('x')
# # plt.ylabel(r'$\psi(x,0)$')
# # plt.title('Función de onda inicial')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()

# # # 3) Parte real e imaginaria final
# # plt.figure(figsize=(8,5))
# # plt.plot(x, psif.real, label=rf'Re$[\psi(x,t_f)]$')
# # plt.plot(x, psif.imag, '--', label=rf'Im$[\psi(x,t_f)]$')
# # plt.axvline(0.0, color='k', linestyle=':', label='centro')
# # plt.xlabel('x')
# # plt.ylabel(r'$\psi(x,t_f)$')
# # plt.title('Función de onda final')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()

# # # 4) Fase inicial y final
# # plt.figure(figsize=(8,5))
# # plt.plot(x, phase0, label='fase t = 0')
# # plt.plot(x, phasef, label=rf'fase t = {t[-1]:.2f}')
# # plt.axvline(0.0, color='k', linestyle=':', label='centro')
# # plt.xlabel('x')
# # plt.ylabel(r'$\arg[\psi(x)]$')
# # plt.title('Fase: inicial y final')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()

# # # 5) Densidad final sola
# # plt.figure(figsize=(8,5))
# # plt.plot(x, denf, label='densidad final')
# # plt.axvline(0.0, color='k', linestyle=':', label='centro')
# # x_min = x[np.argmin(denf)]
# # plt.axvline(x_min, color='r', linestyle='--', label=rf'mínimo en x={x_min:.3f}')
# # plt.xlabel('x')
# # plt.ylabel(r'$|\psi(x,t_f)|^2$')
# # plt.title('Densidad final')
# # plt.grid(alpha=0.2)
# # plt.legend()
# # plt.show()

# # # 6) Mapa de densidad en el tiempo
# # plt.figure(figsize=(8,5))
# # plt.pcolormesh(t, x, den.T, shading='auto')
# # plt.colorbar(label=r'$|\psi(x,t)|^2$')
# # plt.xlabel('t')
# # plt.ylabel('x')
# # plt.title('Evolución temporal de la densidad')
# # plt.show()


# # %%
# # imag_evo.load_model_state(time_step=len(imag_evo.t_grid)-1)
# # den = np.abs(imag_evo.psi)**2
# # norm = np.trapz(den[-1,:], mesh)
# # plt.plot(mesh, den[-1,:] / norm, label=f'g={str(g)}, w={str(w)}')
# # x_max =grid.mesh[np.where(den[-1,:]==den[-1,:].max())]
# # plt.vlines(x_max,0,den[-1,:].max(),linestyles=':',label=f'{round(x_max.item(),3)}')
# # plt.legend()
# # plt.show()

# # %%
# # fig_path = utils.file_ID(pm.figs_dir,
# #                         file_name(pm.architecture, model.architecture, pm.evolution) + f'_g_{pm.g}',
# #                         pm.fig_format)
# # plots.evo_fig_complex(grid.mesh, real_evo.t_grid[0:500], den.real.T[:,0:500], real_evo.get_params()[:,0:500],
# #                        (real_evo.energy-real_evo.energy[0]), fig_path=fig_path)
# # %%
