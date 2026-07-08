import torch
import torch.nn as nn

import parameters as pm

class NQS(nn.Module):
    '''
    Create a Neural Quantum State

    :input_size:        size of the input layer
    :output_size:       size of the output layer
    :hidden_layer:      list containing the size of each hidden layer
    :weights:           (optional) list of numpy arrays or tensors
    :biases:            (optional) list of numpy arrays or tensors
    :activavtion_fns:   (optional) list of activation functions - Sigmoid by default
    '''
    def __init__(self, input_size, output_size, hidden_layers, weights=None, biases=None, activation_fns=None):
        super(NQS, self).__init__()

        # Ensure activation_fns is provided and has the correct length
        if activation_fns is None:
            activation_fns = [nn.Sigmoid] * len(hidden_layers)
        elif len(activation_fns) != len(hidden_layers):
            raise ValueError("Length of activation_fns must match the length of hidden_layers")
        
        # List to hold all layers
        layers = []

        # Network architecture
        self.architecture = "-".join(map(str, [input_size, *hidden_layers, output_size]))
        
        # Add the input layer and hidden layers
        prev_size = input_size
        for ii, (hidden_size, activation_fn) in enumerate(zip(hidden_layers, activation_fns)):
            # Create a linear layer from prev_size to hidden_size
            linear_layer = nn.Linear(prev_size, hidden_size, bias=(biases is None or biases[ii] is not None))
            layers.append(linear_layer)
            # Add activation function
            layers.append(activation_fn())
            # Update prev_size for the next layer
            prev_size = hidden_size
        # Add the output layer
        output_layer = nn.Linear(prev_size, output_size, bias=(biases is None or biases[len(hidden_layers)] is not None))
        layers.append(output_layer)
        # Combine layers into a sequential container
        self.network = nn.Sequential(*layers)
        
        # Optionally set custom weights and biases
        if weights and biases:
            self._initialize_weights_and_biases(weights, biases)
        else:
            weights, biases = self.initialize_weights_and_biases_xavier(input_size, hidden_layers, output_size)
            # Complex weights and biases
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                weights[i] = torch.view_as_complex(torch.stack((weight, weight), -1))*0.1
                biases[i] = torch.view_as_complex(torch.stack((bias, bias), -1))*0.1

            self._initialize_weights_and_biases(weights, biases)

    def initialize_weights_and_biases_xavier(self, input_size, hidden_layers, output_size):
        '''
        Initialize the weights and biases.

        The weights are initialized with the Xavier initialization, 
        which is is well-suited for networks with sigmoid activation functions,
        as it helps maintain stable gradients and activations throughout the layers. 
        '''      
        # Initialize weights
        weights = []
        biases = []

        # Input to first hidden layer
        w_input = torch.empty(hidden_layers[0], input_size, requires_grad=True)
        nn.init.xavier_uniform_(w_input)
        weights.append(w_input)
        
        b_input = torch.zeros(hidden_layers[0], requires_grad=True)
        biases.append(b_input)

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            w_hidden = torch.empty(hidden_layers[i+1], hidden_layers[i], requires_grad=True)
            nn.init.xavier_uniform_(w_hidden)
            weights.append(w_hidden)
            
            b_hidden = torch.zeros(hidden_layers[i+1], requires_grad=True)
            biases.append(b_hidden)
        
        # Last hidden layer to output
        w_output = torch.empty(output_size, hidden_layers[-1], requires_grad=True)
        nn.init.xavier_uniform_(w_output)
        weights.append(w_output)
        
        b_output = torch.zeros(output_size, requires_grad=True)
        biases.append(b_output)
        
        return weights, biases

    def _initialize_weights_and_biases(self, weights, biases):
        # Iterate through layers and set weights and biases
        all_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        for ii, layer in enumerate(all_layers):
            layer.weight.data = weights[ii].clone().detach()
            if biases[ii] is not None:
                layer.bias.data = biases[ii].clone().detach()
    def forward(self, x):
        outputs = self.network(x)

        psi_x = outputs * torch.exp(1j * pm.k * x) if pm.k != 0 else outputs

        if pm.enforce_odd_parity:
            psi_mx = self.network(-x) * (torch.exp(-1j * pm.k * x) if pm.k != 0 else 1)
            psi = 0.5 * (psi_x - psi_mx)

        elif pm.enforce_even_parity:
            psi_mx = self.network(-x) * (torch.exp(-1j * pm.k * x) if pm.k != 0 else 1)
            psi = 0.5 * (psi_x + psi_mx)

        else:
            psi = psi_x

        eps = torch.tensor(1e-12, dtype=psi.real.dtype, device=psi.device)
        psi_safe = psi + eps.to(psi.dtype)

        return torch.log(psi_safe)

    # def forward(self, x):
    #     outputs = self.network(x)
    #     psi_x = outputs * torch.exp(1j * pm.k * x) if pm.k != 0 else outputs

    #     if pm.enforce_odd_parity:
    #         psi_mx = self.network(-x) * (torch.exp(-1j * pm.k * x) if pm.k != 0 else 1)
    #         psi = 0.5 * (psi_x - psi_mx)
    #     elif pm.enforce_even_parity:
    #         psi_mx = self.network(-x) * (torch.exp(-1j * pm.k * x) if pm.k != 0 else 1)
    #         psi = 0.5 * (psi_x + psi_mx)
    #     else:
    #         psi = psi_x

    #     # Protección robusta: separar magnitud y fase
    #     mag = torch.abs(psi)
    #     phase = psi / (mag + 1e-30)          # fase unitaria, siempre finita
    #     log_mag = torch.log(mag.clamp(min=1e-30))  # log real, nunca -inf/-nan
    #     # log(psi) = log(mag) + i*arg(psi) = log_mag + i*angle(phase)
    #     return log_mag + 1j * torch.angle(phase)
        #if pm.k != 0:
         #   outputs2 = self.network(-x)
          #  psi = outputs * torch.exp(1j * pm.k*  x) \
           #     + outputs2
        # psi = (outputs[:,0] * torch.exp(outputs[:,1])).unsqueeze(1)
        # psi = outputs.real * torch.exp(-1j * outputs.imag)
        # norm = torch.trapz(torch.abs(psi)**2, x, dim=0)        
        # psi[:len(x)//2] *= phase
        # psi[len(x)//2-1] = 1e-16*(1+1j)
        #return torch.log(psi)
        # return torch.log(outputs[:,0]).unsqueeze(1)
        # return self.network(x)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
class TwoSolitonNQS(nn.Module):
    """
    Estado restringido de dos solitones construido a partir de una red base.

    psi_total(x) = psi_right(x) + psi_left(x)

    con
        psi_right(x) = network(x - x0) * exp(-i k x)
        psi_left(x)  = exp(i phi) * network(x + x0) * exp(+i k x)

    Interpretación:
        - solitón derecho centrado en +x0, moviéndose hacia la izquierda (-k)
        - solitón izquierdo centrado en -x0, moviéndose hacia la derecha (+k)

    OJO:
        Este ansatz es útil como prototipo rápido, pero restringe la dinámica
        porque obliga al estado a vivir siempre en esta familia de dos copias.
    """
    def __init__(self, base_model, x0, k, phi=0.0):
        super().__init__()
        self.base_model = base_model
        self.x0 = x0
        self.k = k
        self.phi = phi

        # nombre útil para ficheros/plots
        self.architecture = f"TwoSol({base_model.architecture})"

    def forward(self, x):
        phi_t = torch.as_tensor(self.phi, dtype=torch.float64, device=x.device)
        phase = torch.exp(1j * phi_t)

        lnpsi_right = self.base_model(x - self.x0)
        lnpsi_left  = self.base_model(x + self.x0)

        psi_right = torch.exp(lnpsi_right) * torch.exp(-1j * self.k * x)
        psi_left  = torch.exp(lnpsi_left)  * torch.exp(+1j * self.k * x) * phase

        psi_total = psi_right + psi_left

        mag = torch.abs(psi_total)
        eps = torch.as_tensor(1e-12, dtype=mag.dtype, device=mag.device)
        psi_total_safe = torch.where(
            mag < eps,
            eps.to(psi_total.dtype) * torch.ones_like(psi_total),
            psi_total
        )

        return torch.log(psi_total_safe)
    def num_parameters(self):
        return self.base_model.num_parameters()