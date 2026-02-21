
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, num_frequencies=32, sigma=1.0):
        super().__init__()
        B_tensor = torch.randn(input_dim, num_frequencies) * sigma
        self.register_buffer('B', B_tensor)
        
        self.out_dim = num_frequencies * 2

    def forward(self, x):
        x_proj = (2.0 * torch.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class mFCNet(nn.Module):
    """
    Modified Fully-Connected Network (mFCN) based on:
    
    - "Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks"
    (Wang et al., NeurIPS 2021)
    - Appendix A of "Self-adaptive weighting and sampling for physics-informed neural networks"
    (Chen et al., arXiv:2511.05452v2)
    """

    def __init__(self, input_dim: int = 2,
                 output_dim: int = 2, 
                 num_layers: int = 6,
                 num_neurons: int = 64, 
                 activation: nn.Module = nn.Tanh,
                 bias: bool = True,
                 isFourier: bool = False):
        
        super(mFCNet, self).__init__()

        self.activation = activation()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.isFourier = isFourier

        if isFourier: 
            # Fourier Feature Mapping
            self.fourier = FourierFeatureMapping(input_dim=input_dim, num_frequencies=32, sigma=2.0)
            input_dim = self.fourier.out_dim

        # U and V branches: linear tranformations of the input
        self.WU = nn.Linear(input_dim, num_neurons, bias=bias)
        self.WV = nn.Linear(input_dim, num_neurons, bias=bias)

        # First hidden layer
        self.WH = nn.Linear(input_dim, num_neurons, bias=bias)

        # Hidden layers from 2 until n_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(num_neurons, num_neurons, bias=bias)
            for _ in range(num_layers - 1) 
        ])
    
        # Output layer
        self.out_layer = nn.Linear(num_neurons, output_dim, bias=False)

    def forward(self, x):
        """
        x: tensor de forma [N, input_dim]
        """
        if self.isFourier:
            x = self.fourier(x)  # [N, fourier_out_dim]

        # Parallelal branches U and V (constant through hidden layers)
        U = self.activation(self.WU(x))  # [N, num_neurons]
        V = self.activation(self.WV(x))  # [N, num_neurons]

        # First hidden layer
        H = self.activation(self.WH(x))  # [N, num_neurons]

        # Hidden layers from 2 until n_layers
        for layer in self.hidden_layers:
            z = self.activation(layer(H))                 # [N, num_neurons]
            H = (1 - z) * U + z * V                # [N, num_neurons]

        # Output
        out = self.out_layer(H)
        return out

class FCResBlock(nn.Module):
    """Fully Connected Residual Block"""
    def __init__(self, features: int, 
                 num_layers: int = 4, 
                 activation: nn.Module = nn.Tanh, 
                 pre_act: bool = False):
        
        super().__init__()

        self.pre_act = pre_act              # True: pre-activation (as ResNet v2), False: post-activation (traditional)
        self.activation = activation()

        # Layers: features → features (output = input for next residual block)
        self.layers = nn.ModuleList([nn.Linear(features, features) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(features) for _ in range(num_layers)])

    def forward(self, x):
        identity = x
        out = x

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            if self.pre_act and i > 0:
                out = self.activation(out)
            out = layer(out)
            out = norm(out)
            if not self.pre_act or i == 0:
                out = self.activation(out) if i < len(self.layers) - 1 else out

        if self.pre_act:
            out = self.activation(out + identity)
        else:
            out = out + identity
            out = self.activation(out)
        return out

class FCResNet(nn.Module):
    """Fully Connected Residual Network (ResNet MLP-style)"""
    def __init__(self, num_blocks: int = 4, 
                 num_layers: int = 12,          # per block...
                 num_neurons: int = 64, 
                 activation: nn.Module = nn.Tanh):
        
        super().__init__()
        
        self.input_layer = nn.Linear(2, num_neurons)
        
        self.res_blocks = nn.Sequential(*[
            FCResBlock(features = num_neurons,
                       num_layers = num_layers,
                       activation = activation)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(num_neurons, 2)
        self.act = activation()

    def forward(self, x):
        x = self.act(self.input_layer(x))
        x = self.res_blocks(x)
        out = self.output_layer(x)
        return out