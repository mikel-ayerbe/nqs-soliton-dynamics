''' [1] A. Sinibaldi et al., Quantum 7, 1131 (2023).'''

# PyTorch imports
import numpy as np
import torch
from torch.autograd import grad

# Custom imports
import utilities as utils
import parameters as pm

def compute_energy(lnpsi, x_grid):
    """
    Compute the energy of a state described by psi.
    psi is the output of self.model wiht input x_grid.

    Args:
        psi (torch.Tensor): Wavefunction.
        x_grid (torch.Tensor): Spatial grid points.

    Returns:
        energy (np.ndarray): Energy per particle.
    """
    # Compute psi
    psi = torch.exp(lnpsi)
    
    H_psi = hamiltonian(psi, x_grid)  # Apply Hamiltonian to psi
    norm = torch.trapz(torch.abs(psi)**2, x_grid, dim=0)
    psi_H_psi = torch.trapz(torch.conj(psi) * H_psi, x_grid, dim=0)
    energy_new = psi_H_psi #/ norm
    # print(energy)
    # energy = torch.vdot(psi[:,0], H_psi[:,0]) / torch.vdot(psi[:,0], psi[:,0])
    # print(energy, energy_new[0])
    return energy_new[0]

# -----------------------------------------------------------------
# Function to compute the Hamiltonian
# -----------------------------------------------------------------
# Function to compute the Hamiltonian
def hamiltonian(psi, grid): 

    # Kinetic term
    kinetic_prefactor = getattr(pm, "kinetic_prefactor", -0.5)
    kinetic = kinetic_prefactor * utils.second_derivative(psi, grid)

    # Potential term
    potential = 0.5 * pm.w**2 * (grid - pm.x0).pow(2) * psi  

    # Gaussian barrier
    gaussian = (
        pm.gauss_amplitude
        * torch.exp(-(grid - pm.gauss_x0)**2 / (2.0 * pm.gauss_width**2))
        * psi
    )

    # Wall
    wall = pm.wall * torch.heaviside(
        grid.real,
        torch.tensor([0.5], dtype=grid.real.dtype, device=grid.device)
    ) * psi

    # Mean field
    mean_field = pm.g * torch.abs(psi).pow(2) * psi

    # Chemical potential / background rotation
    mu = pm.mu * psi

    return kinetic + potential + gaussian + wall + mean_field - mu
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
    '''
    Computes the variational forces

    Parameters:
        psi         (complex tensor): wavefunction
        psi_grads   (complex tensor): gradients of psi wrt parameters
        grid        (tensor): grid points
    
    Returns:
        variational_forces (complex tensor): variational forces
    '''
    # Compute psi
    psi = torch.exp(lnpsi)

    # Local energy
    E_loc = hamiltonian(psi, grid) / psi #- compute_energy(lnpsi, grid)

    # Remove last dimension
    p = psi.squeeze(1)
    q = torch.conj(p)
    E_l = E_loc.squeeze(1)    

    # Conjugate jacobian
    dz_lnpsi = jacobian

    # Probability density
    norm = torch.vdot(p, p)
    P = p * p.conj() / norm

    # Probability Matrix
    PM = torch.diag(P)
    
    # Terms
    dcPHP = torch.einsum('ji,jk,k->i', torch.conj(dz_lnpsi), PM, E_l)
    # PHdzP = torch.einsum('ji,jk,k->i', dz_lnpsi, PM, torch.conj(E_l))
    dcPPPHP = torch.einsum('ji,j->i', torch.conj(dz_lnpsi), P) * torch.einsum('j,j->', P, E_l)
    # PdzPPHP = torch.einsum('ji,j->i', dz_lnpsi, P) * torch.einsum('j,j->', P, torch.conj(E_l))
    
    # Interaction contribution 
    Qc = pm.g * P[:, None] * torch.conj(dz_lnpsi)
    PdcHP= torch.einsum('ji,j->i', Qc, P)
    if pm.g != pm.g:
        print(PdcHP)

    # Variational forces
    variational_forces = norm * ( dcPHP + PdcHP )#+ PHdzP- PdzPPHP
 
    return variational_forces.clone().detach()

# -----------------------------------------------------------------
# Function to compute the Quantum Geometric Tensor (QGT)
def compute_qgt(lnpsi, jacobian):
    '''
    Computes the Quantum Geometric Tensor

    Parameters:
    psi         (complex tensor): wavefunction
    jacobian    (complex tensor): gradients of psi wrt parameters
    
    Returns:
    S         (complex tensor): Quantum Geometric Tensor
    '''
    # Compute psi
    psi = torch.exp(lnpsi)

    # Remove last dimension
    p = psi.squeeze(1) 
    q = torch.conj(p) 

    # Probability density
    norm = torch.vdot(p, p)
    P = p * p.conj() / norm

    # Probability Matrix
    PM = torch.diag(P)

    # Conjugate jacobian
    dz_lnpsi = jacobian

    S = norm * torch.einsum('ji,jk,kl->il', torch.conj(dz_lnpsi),  PM, dz_lnpsi)
   
    return S.clone().detach()

# -----------------------------------------------------------------
