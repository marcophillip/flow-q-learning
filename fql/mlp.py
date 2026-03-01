"""
Multi-layer perceptron network with optional layer normalization.
"""

import torch
import torch.nn as nn
from typing import Sequence


class MLP(nn.Module):
    """Multi-layer perceptron with layer normalization and GELU activation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activate_final: bool = True,
        use_layer_norm: bool = False,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dims: Sequence of hidden layer dimensions (including output)
            activate_final: Whether to apply activation to final layer
            use_layer_norm: Whether to use layer normalization after each layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            # layers.append(nn.BatchNorm1d(prev_dim))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Layer norm (if enabled)
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation (GELU) - skip on last layer if activate_final=False
            is_last_layer = (i == len(hidden_dims) - 1)
            if not is_last_layer or activate_final:
                layers.append(nn.GELU())
            
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            output: Output tensor of shape [batch_size, hidden_dims[-1]]
        """
        return self.network(x)


class EnsembleMLP(nn.Module):
    """Ensemble of MLP networks (for double Q-learning)."""
    
    def __init__(
        self,
        num_ensemble: int,
        input_dim: int,
        hidden_dims: Sequence[int],
        activate_final: bool = True,
        use_layer_norm: bool = False,
    ):
        """
        Args:
            num_ensemble: Number of networks in ensemble
            input_dim: Dimension of input features
            hidden_dims: Sequence of hidden layer dimensions
            activate_final: Whether to apply activation to final layer
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.num_ensemble = num_ensemble
        
        # Create ensemble of networks
        self.networks = nn.ModuleList([
            MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                activate_final=activate_final,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(num_ensemble)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all networks.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            outputs: Stacked outputs of shape [num_ensemble, batch_size, output_dim]
        """
        outputs = [net(x) for net in self.networks]
        return torch.stack(outputs, dim=0)


if __name__ == "__main__":
    # Test MLP
    batch_size = 32
    input_dim = 10
    hidden_dims = [256, 256, 256, 256, 5]
    
    mlp = MLP(input_dim, hidden_dims, activate_final=False, use_layer_norm=True)
    x = torch.randn(batch_size, input_dim)
    output = mlp(x)
    
    print(f"MLP test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {hidden_dims[-1]})")
    
    # Test EnsembleMLP
    ensemble = EnsembleMLP(
        num_ensemble=2,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activate_final=False,
        use_layer_norm=True,
    )
    ensemble_output = ensemble(x)
    
    print(f"\nEnsemble MLP test:")
    print(f"Ensemble output shape: {ensemble_output.shape}")
    print(f"Expected: (2, {batch_size}, {hidden_dims[-1]})")