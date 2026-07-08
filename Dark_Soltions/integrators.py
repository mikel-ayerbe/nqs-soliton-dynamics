# PyTorch imports
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from tqdm import tqdm
import os

# Custom imports
import utilities as utils
import analysis
import parameters as pm
import stochastic_reconfiguration as SR

# -----------------------------------------------------------------

def EOM(parameters, model, grid):
    """
    Equations of motion for TDVP / stochastic reconfiguration.

    Real time:
        theta_dot = pinv(S) @ (-i F)

    Imaginary time:
        theta_dot = pinv(S) @ (-F)

    Notes
    -----
    The physical Hamiltonian is not defined here.
    It is defined inside SR.compute_variational_forces / SR.hamiltonian.
    """

    # --------------------------------------------------
    # Make sure the model contains the parameters passed to EOM
    # --------------------------------------------------
    vector_to_parameters(parameters, model.parameters())

    # --------------------------------------------------
    # Compute log-wavefunction
    # --------------------------------------------------
    lnpsi = model(grid)

    if not torch.isfinite(lnpsi).all():
        raise ValueError("Non-finite values found in lnpsi inside EOM.")

    # --------------------------------------------------
    # Compute Jacobian, force vector and QGT
    # --------------------------------------------------
    jacobian = SR.compute_wirtinger_jacobian(model, lnpsi)

    F = SR.compute_variational_forces(
        lnpsi,
        jacobian,
        grid
    )

    S = SR.compute_qgt(
        lnpsi,
        jacobian
    )

    if not torch.isfinite(F).all():
        raise ValueError("Non-finite values found in force vector F inside EOM.")

    if not torch.isfinite(S).all():
        raise ValueError("Non-finite values found in QGT matrix S inside EOM.")

    # --------------------------------------------------
    # QGT regularization
    # --------------------------------------------------
    lambda_reg = getattr(pm, "lambda_reg", 0.0)
    pinv_rtol  = getattr(pm, "pinv_rtol", 1e-8)

    S = S + utils.eye_like(S) * lambda_reg

    if not torch.isfinite(S).all():
        raise ValueError("Non-finite values found in regularized QGT S inside EOM.")

    # --------------------------------------------------
    # Right-hand side
    # --------------------------------------------------
    match pm.evolution:
        case 'real':
            rhs = -1j * F

        case 'imag':
            rhs = -F

        case _:
            raise ValueError(f"Unknown evolution type: {pm.evolution}")

    # --------------------------------------------------
    # Solve TDVP linear system using pseudo-inverse
    # --------------------------------------------------
    theta_dot = torch.linalg.pinv(S, rtol=pinv_rtol) @ rhs

    if not torch.isfinite(theta_dot).all():
        raise ValueError("Non-finite parameter velocity theta_dot inside EOM.")

    # --------------------------------------------------
    # Optional clipping of parameter update
    # --------------------------------------------------
    max_param_step = getattr(pm, "max_param_step", None)

    if max_param_step is not None:
        step = pm.dt * theta_dot
        step_norm = torch.linalg.vector_norm(step)

        if torch.isfinite(step_norm) and step_norm > max_param_step:
            theta_dot = theta_dot * (max_param_step / (step_norm + 1e-14))

    return theta_dot

# -----------------------------------------------------------------
def integrator(model, grid, t_grid=None, file_path=pm.file_path):
    """
    Performs numerical integration.

    Args:
        model (torch.nn.Module): The model whose parameters evolve in time.
        grid: spatial grid object.
        t_grid (torch.Tensor or np.ndarray): The time steps to save the model.
        file_path (str): Path to the HDF5 file where the model is saved.
    """
    # Ensure data directory exists
    if not os.path.exists(pm.data_dir):
        os.makedirs(pm.data_dir)

    # Save model architecture and time steps
    analysis.save_model_architecture(model, file_path)

    if t_grid is None:
        t_grid = utils.time_grid()

    # Build spatial grid once.
    # Important: this must exist even when pm.stopper = False.
    x = grid.x()

    # Flatten parameters
    u = parameters_to_vector(model.parameters())

    # Time steps to compute before saving the data
    t_step = round(pm.t_max / len(t_grid) / pm.dt)

    if t_step < 1:
        t_step = 1

    # Save the model state at initial time step in HDF5
    analysis.save_model_states(model, time_step=0, file_path=file_path)

    if pm.stopper:
        e0 = SR.compute_energy(model(x), x)
        check_point = 0

    # For each data time
    with tqdm(total=len(t_grid) - 1, disable=not pm.progress_bar) as pbar:
        for it in range(1, len(t_grid)):

            # For each internal integration step
            for _ in range(t_step):

                # Choose integration method
                match pm.integrator:
                    case 'RK4':
                        u = RK4(u, model, x)

                    case 'RK45':
                        accepted = False
                        while not accepted:
                            u, accepted = RK45(u, model, x)

                    case 'Euler':
                        u = Euler(u, model, x)

                    case _:
                        raise ValueError(f"Unknown integrator: {pm.integrator}")

                # Update model
                vector_to_parameters(u, model.parameters())

            try:
                # Check for NaN / inf in lnpsi
                lnpsi_check = model(x)

                if not torch.isfinite(lnpsi_check).all():
                    n_bad = (~torch.isfinite(lnpsi_check)).sum().item()
                    raise ValueError(
                        f"Non-finite lnpsi encountered at time step {it}: "
                        f"{n_bad} bad values. Check dt, lambda_reg, and fit quality."
                    )

                # Check for NaN / inf in psi
                psi_check = torch.exp(lnpsi_check)

                if not torch.isfinite(psi_check).all():
                    n_bad = (~torch.isfinite(psi_check)).sum().item()
                    raise ValueError(
                        f"Non-finite psi encountered at time step {it}: "
                        f"{n_bad} bad values. Check dt, lambda_reg, and fit quality."
                    )

                # Save the model state at this time step in HDF5
                analysis.save_model_states(model, time_step=it, file_path=file_path)

                # Check for convergence
                if pm.stopper:
                    energy = SR.compute_energy(model(x), x)
                    e_diff = abs(energy - e0)

                    norm = torch.trapz(
                        torch.abs(torch.exp(model(x)))**2,
                        x,
                        dim=0
                    )

                    pbar.set_postfix_str(
                        f"E={energy.real:.5e}; "
                        f"Err={e_diff:.5e}; "
                        f"Norm={norm.real[0]:.1e};"
                    )

                    if pm.evolution == 'imag' and e_diff <= pm.e_error:
                        check_point += 1

                        if check_point == pm.steps // t_step + 1:
                            t_grid = t_grid[:it + 1]
                            print('Convergence reached')
                            break

                    else:
                        if pm.evolution == 'imag':
                            e0 = energy

                        check_point = 0

                # Update progress bar
                pbar.update(1)

            except ValueError as e:
                print(e)
                t_grid = t_grid[:it]
                break

        analysis.save_variable(t_grid, "t_grid", file_path)
        analysis.save_variable(pbar.format_dict['elapsed'], 'cmp_time', file_path)

# -----------------------------------------------------------------
def RK4(u, model, grid):

    ku1 = EOM(u, model, grid)
    vector_to_parameters(u + 0.5 * ku1 * pm.dt, model.parameters())
    ku2 = EOM(u + 0.5 * ku1 * pm.dt, model, grid)
    vector_to_parameters(u + 0.5 * ku2 * pm.dt, model.parameters())
    ku3 = EOM(u + 0.5 * ku2 * pm.dt, model, grid)
    vector_to_parameters(u + ku3 * pm.dt, model.parameters())
    ku4 = EOM(u + ku3 * pm.dt, model, grid)

    return u + pm.dt * ( ku1 + 2 * ku2 + 2 * ku3 + ku4 ) / 6

# -----------------------------------------------------------------
def Euler(u, model, grid):

    ku1 = EOM(u, model, grid)
    return u + pm.dt * ku1

# -----------------------------------------------------------------
def RK45(u, model, grid):
    """Performs ONE adaptive RK45 step and returns (u_next, accepted)."""

    # Coefficients: Runge-Kutta-Fehlberg (4th/5th order)
    # k1
    k1 = EOM(u, model, grid)
    update = u + pm.dt * k1 * 1/4

    # k2
    vector_to_parameters(update, model.parameters())
    k2 = EOM(update, model, grid)
    update = u + pm.dt * (3/32*k1 + 9/32*k2)

    # k3
    vector_to_parameters(update, model.parameters())
    k3 = EOM(update, model, grid)
    update = u + pm.dt * (1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)

    # k4
    vector_to_parameters(update, model.parameters())
    k4 = EOM(update, model, grid)
    update = u + pm.dt * (439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4)

    # k5
    vector_to_parameters(update, model.parameters())
    k5 = EOM(update, model, grid)
    update = u + pm.dt * (-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5)

    # k6
    vector_to_parameters(update, model.parameters())
    k6 = EOM(update, model, grid)

    # 4th-order estimate
    u4 = u + pm.dt * (25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)

    # 5th-order estimate
    u5 = u + pm.dt * (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)

    # Local truncation error estimate
    err = (u5 - u4).norm().item()

    # Accept or reject step
    if err == 0:
        # Step accepted
        # Update pm.dt using safety factor
        pm.dt = pm.dt * pm.dt_max
        return u5, True
    elif err < pm.tol:
        # Step accepted
        # Update pm.dt using safety factor
        pm.dt = pm.dt * min(max(pm.dt_min, 0.84 * (pm.tol / err)**0.25), pm.dt_max)
        return u5, True
    else:
        # Step rejected — shrink pm.dt
        pm.dt = pm.dt * max(pm.dt_min, 0.84 * (pm.tol / err)**0.25)
        return u, False
