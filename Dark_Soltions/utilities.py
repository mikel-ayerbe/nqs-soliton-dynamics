# PyTorch imports
import torch
from torch.autograd import grad

# Other imports
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import benchmarks as bm
# Custom imports
import parameters as pm

def file_ID(directory, file_name, format):
    """
    Check if a file exists.
    If it exists and do not want to overwrite it, change name.

    Args:
        directory (str): path to directory.                    
        file_name (str): file name.
        format (str): file format.

    Returns:
        file_path (str): Path to file.
    """
    # TODO: fails when original file has been deleted
    file_path = directory + file_name + '.' + format
    v_id = pm.version
    msg = False
    # Check if original file exists
    if os.path.exists(file_path) and not pm.overwrite:
        file_path = directory + file_name + f"_v{v_id}." +  format
        msg = True
    # Check if version files exist
    while os.path.exists(file_path) and not pm.overwrite:
        v_id += 1
        file_path = directory + file_name + f"_v{v_id}." +  format
        msg = True

    pm.version = v_id
    if msg: print(f"This file version is v{v_id}")
    return file_path

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def derivative(f, x):
    """
    Compute the derivative of function f(x) at x.

    Args:
        f (torch.Tensor): Function that depends on x.                    
        x (torch.Tensor): Parameters of f.

    Returns:
        dfdx (torch.Tensor): Derivative of f at x.
    """
    try:
        dfdx, = grad(
                    outputs=f,
                    inputs=x,
                    grad_outputs=torch.ones_like(f).type_as(f),
                    create_graph=True
                    )
    except Exception as error:
        print(f"x_grid dtype: {x.dtype}, requires_grad: {x.requires_grad}")
        print(f"psi dtype: {f.dtype}, requires_grad: {f.requires_grad}")
        print('Error: ', error)

    return dfdx

def second_derivative(f, x):
    # Split function
    u =       0.5 * (f + f.conj())
    v = -1j * 0.5 * (f - f.conj())  

    (du_dx, _) = torch.view_as_real(derivative(u, x)).type_as(u).unbind(-1)
    (dv_dx, _) = torch.view_as_real(derivative(v, x)).type_as(v).unbind(-1)
    (d2u_d2x, _) = torch.view_as_real(derivative(du_dx, x)).type_as(u).unbind(-1)
    (d2v_d2x, _) = torch.view_as_real(derivative(dv_dx, x)).type_as(v).unbind(-1)

    return d2u_d2x + 1j * d2v_d2x


def time_grid():
    # Time parameters
    Nt = pm.t_size          # time vector size without t = 0
    dt = pm.dt              # time discretization
    t_max = pm.t_max        # last time instance

    # check if we can provide enough time data points
    # otherwise create time grid accordingly
    if t_max / dt < Nt:
        return np.arange(0, t_max + dt, dt)   # time vector
    else:
        return np.linspace(0, t_max, Nt+1)    # time vector

def sample_pdf(pdf, xmin:float, xmax:float, n:int, m:float=1.0) -> torch.Tensor:
    """
    Sample n random values from a probability density function (pdf)
    using rejection sampling.
    """
    samples = []
    while len(samples) < n-1:
        x = (xmax - xmin) * torch.rand(1).item() + xmin
        y = m * torch.rand(1).item()  # assume pdf <= m
        if y < pdf(x):
            samples.append(x)
    samples.append(1e-20)
    return torch.as_tensor(samples).sort()[0]

class PointGrid:
    def __init__(self, N: int, start: float, end: float, device: torch.device):
        """
        Initialize the PointGrid object.

        Args:
        - N (int): Number of points in the grid.
        - start (float): The starting value of the grid.
        - end (float): The ending value of the grid.
        """
        # basic grid info
        self.N = N
        self.start = start
        self.end = end

        # device to store teh grid
        self.device = device

        # probability density function to sample the grid
        self.pdf = None

        # default mesh 
        self.mesh = torch.linspace(self.start, self.end, self.N)
        self.dx = self.spacing()

    def sampler(self, pdf):
        """Define the PDF"""
        self.pdf = pdf

    def x(self):
        # torch.unsqueeze(1) creates a [Nx, 1] tensor
        # torch.requires_grad_() tells autograd to record operations on this tensor

        if self.pdf is not None:
            self.mesh = sample_pdf(pdf=self.pdf, 
                                    xmin=self.start,
                                    xmax=self.end,
                                    n=self.N)
            self.dx = self.spacing()
        
        return self.mesh.clone().unsqueeze(1).requires_grad_().type(torch.complex128).to(self.device) 
    
    def spacing(self):
        dx_left = self.mesh[1] - self.mesh[0]
        dx_in = 0.5 * (self.mesh[2:] - self.mesh[:-2])
        dx_right = self.mesh[-1] - self.mesh[-2]
        return torch.cat([dx_left.unsqueeze(0), dx_in, dx_right.unsqueeze(0)])

    def get_points(self):
        """
        Get the grid points.

        Returns:
        - torch.Tensor: The grid points.
        """
        return self.points

    def get_limits(self):
        """
        Get the limits of the grid.

        Returns:
        - tuple: The start and end values of the grid.
        """
        return self.start, self.end
        
    def get_weights(self):
        """
        Get the interation weights for each point.

        Returns:
        - torch.Tensor: The weights.
        """
        return torch.empty(self.N, 1).fill_(self.get_spacing())

    def get_properties(self):
        """
        Get all properties of the grid.

        Returns:
        - dict: A dictionary containing all grid properties.
        """
        return {
            'N': self.N,
            'start': self.start,
            'end': self.end,
            'points': self.points
        }


def fit_model_to_target(model, x, psi_target, lr=1e-3, epochs=3000, verbose=True):
    """
    Ajusta una NQS libre para reproducir psi_target(x).
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if x.ndim == 1:
        x_eval = x[:, None]
        x_int = x
    else:
        x_eval = x
        x_int = x.squeeze()

    psi_target = psi_target.detach()
    loss_history = []

    epoch_iter = tqdm(range(epochs), desc="Fitting psi_target") if verbose else range(epochs)

    for epoch in epoch_iter:
        optimizer.zero_grad()

        lnpsi = model(x_eval)
        psi = torch.exp(lnpsi).squeeze()

        # normalización continua
        norm = torch.trapz(torch.abs(psi)**2, x_int)
        psi = psi / torch.sqrt(norm)

        # pérdida L2 compleja
        loss = torch.trapz(torch.abs(psi - psi_target)**2, x_int).real

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f"{loss.item():.3e}")

        if verbose and epoch % 5000 == 0:
            tqdm.write(f"epoch={epoch:5d}, loss={loss.item():.6e}")

    model.eval()
    return model, loss_history