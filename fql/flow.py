"""
Flow matching networks for behavioral cloning policy.
"""

import torch
import torch.nn as nn
from typing import Sequence
from fql.mlp import MLP


class FlowVelocityField(nn.Module):
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512, 512),
    ):
        """
        Args:
            state_dim: Dimension of state observation
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Input: [t, state, noisy_action]
        input_dim = 1 + state_dim + action_dim  
        
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=list(hidden_dims) + [action_dim],
            activate_final=False,
            use_layer_norm=False,
        )
    
    def forward(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity at time t.
        
        Args:
            t: Time step [batch_size, 1] in [0, 1]
            state: State observation [batch_size, state_dim]
            x: Noisy action [batch_size, action_dim]
            
        """
        inputs = torch.cat([t, state, x], dim=-1)
        
        velocity = self.network(inputs)
        
        return velocity


class FlowPolicy(nn.Module):
    """Flow-matching policy using Euler ODE solver."""
    
    def __init__(
        self,
        velocity_field: FlowVelocityField,
        num_steps: int = 10,
    ):
        """
        Args:
            velocity_field: The velocity field network
            num_steps: Number of  steps 
        """
        super().__init__()
        
        self.velocity_field = velocity_field
        self.num_steps = num_steps
        self.action_dim = velocity_field.action_dim
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate action by solving ODE from noise z.
        
        Args:
            state: State observation [batch_size, state_dim]
            z: Initial noise [batch_size, action_dim] ~ N(0, I)
               If None, will sample from standard normal
               
        Returns:
            action: Generated action [batch_size, action_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        z = torch.randn(batch_size, self.action_dim, device=device)  # Sample initial noise if not provided
        
        # Start from noise
        x = z
        dt = 1.0 / self.num_steps
        
        # Euler integration: x_{t+dt} = x_t + v(t, s, x_t) * dt
        for i in range(self.num_steps):
            t = torch.ones(batch_size, 1, device=device) * (i * dt)
            velocity = self.velocity_field(t, state, x)
            x = x + velocity * dt
        
        x = torch.tanh(x)
        return x
    
    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample multiple actions for a given state.
        
        Args:
            state: State observation [batch_size, state_dim]
            
        Returns:
            actions: Sampled actions [batch_sizees, action_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample noise
        z = torch.randn(
            batch_size,
            self.action_dim,
            device=device
        )
        
        # Generate actions
        actions = self.forward(state, z)
    
        
        return actions


def compute_flow_matching_loss(
    velocity_field: FlowVelocityField,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Compute flow matching loss for behavioral cloning.
    
    L_Flow = E[||v_θ(t, s, x_t) - (x_1 - x_0)||²]
    
    Args:
        velocity_field: The velocity field network
        states: State observations [batch_size, state_dim]
        actions: Actions from dataset (x_1) [batch_size, action_dim]
        
    Returns:
        loss: Scalar loss value
        info: Dictionary with loss components
    """
    batch_size = states.shape[0]
    action_dim = actions.shape[-1]
    device = states.device
    
    # Sample x_0 ~ N(0, I)
    x_0 = torch.randn(batch_size, action_dim, device=device)
    
    # x_1 from dataset
    x_1 = actions
    
    # Sample t ~ Uniform[0, 1]
    t = torch.rand(batch_size, 1, device=device)
    
    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    x_t = (1 - t) * x_0 + t * x_1
    
    # Target velocity: x_1 - x_0
    target_velocity = x_1 - x_0
    
    # Predicted velocity: v_θ(t, s, x_t)
    predicted_velocity = velocity_field(t, states, x_t)
    
    # MSE loss
    loss = torch.mean((predicted_velocity - target_velocity) ** 2)
    
    info = {
        'flow_loss': loss.item(),
        'velocity_norm': torch.mean(torch.norm(predicted_velocity, dim=-1)).item(),
    }
    
    return loss, info


# if __name__ == "__main__":
#     # Test FlowVelocityField
#     batch_size = 32
#     state_dim = 17
#     action_dim = 6
    
#     print("Testing FlowVelocityField...")
#     velocity_field = FlowVelocityField(state_dim, action_dim)
    
#     t = torch.rand(batch_size, 1)
#     states = torch.randn(batch_size, state_dim)
#     x = torch.randn(batch_size, action_dim)
    
#     velocity = velocity_field(t, states, x)
#     print(f"Velocity shape: {velocity.shape}")
#     print(f"Expected: ({batch_size}, {action_dim})")
    
#     # Test FlowPolicy
#     print("\nTesting FlowPolicy...")
#     flow_policy = FlowPolicy(velocity_field, num_steps=10)
    
#     actions = flow_policy(states)
#     print(f"Generated actions shape: {actions.shape}")
#     print(f"Expected: ({batch_size}, {action_dim})")
    
    
#     # Test flow matching loss
#     print("\nTesting flow matching loss...")
#     dataset_actions = torch.randn(batch_size, action_dim)
#     loss, info = compute_flow_matching_loss(velocity_field, states, dataset_actions)
#     print(f"Flow loss: {loss.item():.4f}")
#     print(f"Info: {info}")