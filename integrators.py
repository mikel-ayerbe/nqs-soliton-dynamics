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
    lnpsi = model(grid)

    if not torch.isfinite(lnpsi).all():
        raise ValueError(f"[EOM] lnpsi no finito: {(~torch.isfinite(lnpsi)).sum()} valores")

    jacobian = SR.compute_wirtinger_jacobian(model, lnpsi)

    if not torch.isfinite(jacobian).all():
        raise ValueError(f"[EOM] jacobian no finito: {(~torch.isfinite(jacobian)).sum()} valores")

    F = SR.compute_variational_forces(lnpsi, jacobian, grid)

    if not torch.isfinite(F).all():
        raise ValueError(f"[EOM] F no finito: {(~torch.isfinite(F)).sum()} valores")

    S = SR.compute_qgt(lnpsi, jacobian, grid)

    if not torch.isfinite(S).all():
        raise ValueError(f"[EOM] S no finito: {(~torch.isfinite(S)).sum()} valores")

    S += utils.eye_like(S) * pm.lambda_reg

    match pm.evolution:
        case 'real':
            parameters = torch.linalg.pinv(S, rtol=1e-8) @ (-1j * F)
        case 'imag':
            parameters = torch.linalg.pinv(S, rtol=1e-8) @ (-F)

    return parameters
# -----------------------------------------------------------------
def integrator(model, grid, t_grid=None, file_path=pm.file_path):
    """
    Performs numerical integration

    Args:
        model (torch.nn.Module): The model whose parameters evolve in time.
        x_grid (torch.Tensor): The spatial grid (inputs) for the model.
        t_grid (torch.Tensor): The time steps to save the model.   
        fiel_path (str): Path to the HDF5 file where the model is saved.
    """
    # Ensure data directory exists
    if not os.path.exists(pm.data_dir):
        os.makedirs(pm.data_dir)

    # Save model architecture and time steps
    analysis.save_model_architecture(model, file_path)
    if not t_grid:
        t_grid = utils.time_grid()

    # Flatten parameters
    # TODO: compare methods to export and import parameters
    # u = torch.cat([p.view(-1) for p in model.parameters()])
    u = parameters_to_vector(model.parameters())

    # Time steps to compute before saving the data
    t_step = round(pm.t_max / len(t_grid) / pm.dt)

    # Save the model state at initial time step in HDF5
    analysis.save_model_states(model, time_step=0, file_path=file_path)
    x = grid.x()
    if pm.stopper:
        
        e0 = SR.compute_energy(model(x), x)
        check_point = 0

    # For each data time
    with tqdm(total=len(t_grid)-1, disable=not pm.progress_bar) as pbar:
        for it in range(1, len(t_grid)):
            # For each time step
            for _ in range(t_step):
                # Choose integration method
                match pm.integrator:
                    case 'RK4':    
                        u = RK4(u, model, grid.x())  
                    case 'RK45':
                        accepted = False
                        while not accepted:
                            u, accepted = RK45(u, model, grid.x())
                    case 'Euler':
                        u = Euler(u, model, grid.x())
                # Update model
                vector_to_parameters(u, model.parameters())    

            try:
                # Check for NaN
                if torch.isnan(model(x)).any():
                    raise ValueError(f"NaN encountered in wavefunction at time step {it}. " +\
                                      "Check parameters: dt, lambda_reg, reg.")   
                else:
                    # Save the model state at this time step in HDF5
                    analysis.save_model_states(model, time_step=it, file_path=file_path)
                    # Check for convergence
                    if pm.stopper:
                        energy = SR.compute_energy(model(x), x)
                        e_diff  = abs(energy - e0)
                        pbar.set_postfix_str(f"E={energy.real:.5e}; Err={e_diff:.5e}")
                        if pm.evolution == 'imag' and e_diff <= pm.e_error:
                            check_point += 1
                            if check_point == pm.steps // t_step + 1:
                                t_grid = t_grid[:it+1]
                                print('Convergence reached')                                
                                break
                        else:
                            if pm.evolution=='imag': e0 = energy
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
