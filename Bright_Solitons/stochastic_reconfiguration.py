''' [1] A. Sinibaldi et al., Quantum 7, 1131 (2023).'''

# PyTorch imports
import numpy as np
import torch
from torch.autograd import grad

# Custom imports
import utilities as utils
import parameters as pm
def trapz_weights(x):
    """
    Pesos de la regla del trapecio para una malla 1D.
    x puede tener forma (N,) o (N,1).
    """
    x = x.squeeze(-1) if x.ndim > 1 else x
    w = torch.zeros_like(x, dtype=torch.float64)

    dx = x[1:] - x[:-1]
    w[0] = dx[0] / 2
    w[-1] = dx[-1] / 2
    if len(x) > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2

    return w.to(torch.complex128)


def normalize_psi(psi, grid):
    """
    Normaliza psi con la medida continua:
    integral |psi|^2 dx = pm.target_norm
    """
    norm = torch.trapz(torch.abs(psi)**2, grid, dim=0)
    target = torch.tensor(
        pm.target_norm,
        dtype=psi.real.dtype,
        device=psi.device
    )
    return psi / torch.sqrt(norm) * torch.sqrt(target)

def compute_energy(lnpsi, x_grid):
    psi = torch.exp(lnpsi)
    psi = normalize_psi(psi, x_grid)

    H_psi = hamiltonian(psi, x_grid)
    psi_H_psi = torch.trapz(torch.conj(psi) * H_psi, x_grid, dim=0)
    return psi_H_psi[0]

# -----------------------------------------------------------------
# Function to compute the Hamiltonian
def hamiltonian(psi, grid):
    kinetic    = -(1/2) * utils.second_derivative(psi, grid)
    potential  = (1/2) * pm.w**2 * (grid - pm.x0).pow(2) * psi
    gaussian   = pm.gauss_amplitude * torch.exp(-(grid - pm.gauss_x0)**2 / 2 / pm.gauss_width**2) * psi
    wall       = pm.wall * torch.heaviside(grid.real, torch.tensor([0.5])) * psi
    mean_field = pm.g * torch.abs(psi).pow(2) * psi
    mu         = -1 * psi
    pinning    = pm.pin_strength * (grid - pm.pin_x0).pow(2) * psi
    return kinetic + potential + gaussian + wall + mean_field - mu + pinning
# -----------------------------------------------------------------
# Function to compute the Jacobian of Ψ(x) wrt parameters
def compute_wirtinger_jacobian(model, outputs):
    """
    Computes the Jacobian of the outputs of the model wrt the parameters of the model.
    
    Arguments:
        model (nn.Module): PyTorch model containing parameters.
        outputs (Tensor): Complex-valued outputs from the model.

    Returns:
        Tensor: Wirtinger Jacobian (complex-valued).
    """
    # Split wavefunction into real and imaginary
    u =       0.5 * (outputs + outputs.conj())
    v = -1j * 0.5 * (outputs - outputs.conj())

    # grid length
    n = outputs.shape[0]

    # Compute partial derivatives - Jacobians
    def partial_derivatives(outputs, inputs):
        # Identity Tensor
        eye = torch.eye(n).type(torch.complex128).unsqueeze(-1)
        # Gradients
        gradients = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=eye,
            # retain_graph=True,
            create_graph=True,
            is_grads_batched=True
            )
        # flattened_gradients = [gradient.reshape(n, -1) for gradient in gradients]
        flattened_gradients = list(map(lambda gradient: gradient.reshape(n, -1), gradients))

        return  torch.view_as_real(
                    torch.cat(flattened_gradients, dim=-1)
                    ).unbind(-1)

    # Partial derivatives
    (du_dx, du_dy) = partial_derivatives(u, model.parameters())
    (dv_dx, dv_dy) = partial_derivatives(v, model.parameters())
    
    # Wirtinger derivatives
    df_dx = du_dx + 1j * dv_dx
    df_dy = du_dy + 1j * dv_dy
    df_dz = 0.5 * (df_dx - 1j * df_dy)
    df_dc = 0.5 * (df_dx + 1j * df_dy)

    # Return a complex Tensor of size [num_parameters, num_inputs]
    return df_dz.clone().detach() #, df_dc.clone().detach()

# -----------------------------------------------------------------
# Function to compute the variational forces
def compute_variational_forces(lnpsi, jacobian, grid):
    psi = torch.exp(lnpsi)
    psi = normalize_psi(psi, grid)

    # Proteger división
    eps = 1e-12
    psi_safe = psi.clone()
    psi_safe[torch.abs(psi_safe) < eps] = eps + 0j

    # Local energy
    E_loc = hamiltonian(psi, grid) / psi_safe - compute_energy(lnpsi, grid)

    p = psi.squeeze(1)
    E_l = E_loc.squeeze(1)
    dz_lnpsi = jacobian

    # VUELTA A LA DEFINICIÓN ORIGINAL
    norm = torch.vdot(p, p)
    P = p * p.conj() / norm
    PM = torch.diag(P)

    dcPHP    = torch.einsum('ji,jk,k->i', torch.conj(dz_lnpsi), PM, E_l)
    dcPPPHP  = torch.einsum('ji,j->i', torch.conj(dz_lnpsi), P) * torch.einsum('j,j->', P, E_l)

    Qc = pm.g * P[:, None] * torch.conj(dz_lnpsi)
    PdcHP = torch.einsum('ji,j->i', Qc, P)

    variational_forces = dcPHP - dcPPPHP + PdcHP
    return variational_forces.clone().detach()

# -----------------------------------------------------------------
# Function to compute the Quantum Geometric Tensor (QGT)
def compute_qgt(lnpsi, jacobian, grid):
    psi = torch.exp(lnpsi)
    psi = normalize_psi(psi, grid)

    p = psi.squeeze(1)
    P = p * p.conj() / torch.vdot(p, p)
    PM = torch.diag(P)

    dz_lnpsi = jacobian

    S =   torch.einsum('ji,jk,kl->il', torch.conj(dz_lnpsi), PM, dz_lnpsi) \
        - torch.einsum('ji,j->i', torch.conj(dz_lnpsi), P) \
        * torch.einsum('ji,j->i', dz_lnpsi, P)

    return S.clone().detach()

# -----------------------------------------------------------------

# -----------------------------------------------------------------

# -----------------------------------------------------------------
