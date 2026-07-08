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
k = 0.0

# Mean field
g = 1e-5*0
mu = 0

enforce_even_parity = False
# Soft pinning: centra el solitón durante la búsqueda del ground state
# Ponlo a 0.0 para evolución real (el solitón se moverá libremente)
pin_strength = 0
pin_x0 = 0.0

# Norma objetivo del estado (1 para un solitón, 2 para dos solitones)
target_norm = 1.0

enforce_even_parity = False
enforce_odd_parity  = False

adaptive_reg = False  # False por defecto — no afecta a nada existente