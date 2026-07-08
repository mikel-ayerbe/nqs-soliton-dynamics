# Architecture
architecture = 'NQS'

# Trapping potential
x0 = 0.
w = 1.

# Time evolution parameters
t_size = 1000           # time vector size without t = 0
dt = 0.1                # time discretization
t_max = 1               # last time instance
evolution = 'real'
integrator = 'RK4'
progress_bar = True

# Convergence parameters
stopper = True
e_error = 1e-5
steps = 10

# RK45
tol = 1e-6
dt_min = 0.1
dt_max = 2.0

# Parameteres to save data
data_dir = "./data/"
data_format = "h5"
figs_dir = "./figs/"
fig_format = "png"
file_path = "model_states.h5"
overwrite = True
version = 1

# Regularization parameters
lambda_reg = 1e-3
reg = 'diagonal_shift'

# Gaussian barrier
gauss_width = 1
gauss_amplitude = 0
gauss_x0 = 0

# Wall
wall = 0

# Kick
k = 0

# Mean field
g = 1e-5*0
mu = 0
# Phase imprinting
phase = 0.0
phase_width = 2.0
phase_center = 0.0
use_phase_step = False
# Parity constraints
enforce_even_parity = False
enforce_odd_parity = False