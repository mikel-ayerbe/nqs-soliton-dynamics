# nqs-soliton-dynamics

Time-dependent Neural Quantum States for soliton dynamics in the one-dimensional Gross–Pitaevskii equation.

This repository contains the code used to simulate bright and dark soliton dynamics with Neural Quantum States (NQS).

## Structure

The repository is organized into two main folders:

```text
bright/
dark/
```

The `bright/` folder contains the code for attractive interactions and bright-soliton dynamics.

The `dark/` folder contains the code for repulsive interactions and dark-soliton dynamics.

Both folders follow the same general structure:

```text
main.py                         Main script used to run the simulations.
parameters.py                   Global simulation parameters.
models.py                       Neural Quantum State architecture.
utilities.py                    Grids, derivatives, time grids, and file utilities.
integrators.py                  Imaginary-time and real-time evolution routines.
stochastic_reconfiguration.py   TDVP / stochastic reconfiguration equations.
analysis.py                     Tools to load and analyze saved simulations.
benchmarks.py                   Benchmark and error-analysis functions.
plots.py                        Plotting routines.
exact_nls.py                    Analytical nonlinear Schrödinger equation solutions used for comparison.
```


## Author

Mikel Ayerbe
