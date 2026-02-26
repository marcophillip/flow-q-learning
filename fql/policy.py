"""
One-step policy network that distills from flow policy.
"""

import torch
import torch.nn as nn
from typing import Sequence
from mlp import MLP


class OneStepPolicy(nn.Module):
    
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
        
        input_dim = state_dim + action_dim   # input is state + noise
        
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=list(hidden_dims) + [action_dim],
            activate_final=False,
            use_layer_norm=False,
        )
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate action from state and noise.
        
        Args:
            state: State observation [batch_size, state_dim]
            z: Noise sample [batch_size, action_dim] ~ N(0, I)
               If None, will sample from standard normal
               
        Returns:
            action: Generated action [batch_size, action_dim]
        """
        batch_size = state.shape[0]
        device = state.device

        z = torch.randn(batch_size, self.action_dim, device=device)
        
        # Concatenate state and noise
        inputs = torch.cat([state, z], dim=-1)
        
        # Predict action directly (one-step)
        action = self.network(inputs)
        
        return action
    
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
            actions: Sampled actions [batch_size, action_dim]
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


def compute_distillation_loss(
    one_step_policy: OneStepPolicy,
    flow_policy: nn.Module,
    states: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Compute distillation loss between one-step policy and flow policy.
    
    L_Distill = E[||μ_ω(s, z) - μ_θ(s, z)||²]
    
    Args:
        one_step_policy: The one-step policy network
        flow_policy: The flow policy to distill from (should be in eval mode)
        states: State observations [batch_size, state_dim]
        noise: Noise samples [batch_size, action_dim] (if None, will sample)
        
    Returns:
        loss: Scalar distillation loss
        info: Dictionary with loss components
    """
    batch_size = states.shape[0]
    device = states.device
    
    noise = torch.randn(batch_size, one_step_policy.action_dim, device=device)
    
    one_step_actions = one_step_policy(states)
    
    with torch.no_grad():
        flow_actions = flow_policy(states)
    
    # MSE loss between outputs
    loss = torch.mean((one_step_actions - flow_actions) ** 2)
    
    info = {
        'distill_loss': loss.item(),
        'action_diff': torch.mean(torch.norm(one_step_actions - flow_actions, dim=-1)).item(),
        'one_step_action_norm': torch.mean(torch.norm(one_step_actions, dim=-1)).item(),
        'flow_action_norm': torch.mean(torch.norm(flow_actions, dim=-1)).item(),
    }
    
    return loss, info


def compute_one_step_policy_loss(
    one_step_policy: OneStepPolicy,
    flow_policy: nn.Module,
    critic: nn.Module,
    states: torch.Tensor,
    noise: torch.Tensor = None,
    bc_coef: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute combined loss for one-step policy (Q maximization + distillation).
    
    L_π(ω) = E[-Q(s, a_π)] + α * L_Distill(ω)
    
    Args:
        one_step_policy: The one-step policy network
        flow_policy: The flow policy to distill from
        critic: The Q-network
        states: State observations [batch_size, state_dim]
        noise: Noise samples [batch_size, action_dim] (if None, will sample)
        bc_coef: Coefficient for distillation loss (α in paper)
        
    Returns:
        loss: Combined scalar loss
        info: Dictionary with loss components
    """
    batch_size = states.shape[0]
    device = states.device
    
    # Sample noise if not provided
    if noise is None:
        noise = torch.randn(batch_size, one_step_policy.action_dim, device=device)
    
    # Generate actions from one-step policy
    actions = one_step_policy(states, noise)
    
    # Q loss: -Q(s, a_π)
    q_values = critic(states, actions)  # [num_ensemble, batch_size, 1]
    q_mean = q_values.mean(dim=0)  # Mean over ensemble [batch_size, 1]
    q_loss = -q_mean.mean()
    
    # Distillation loss
    with torch.no_grad():
        flow_actions = flow_policy(states, noise)
    distill_loss = torch.mean((actions - flow_actions) ** 2)
    
    # Combined loss
    total_loss = q_loss + bc_coef * distill_loss
    
    info = {
        'total_policy_loss': total_loss.item(),
        'q_loss': q_loss.item(),
        'distill_loss': distill_loss.item(),
        'q_mean': q_mean.mean().item(),
        'q_std': q_values.std().item(),
        'action_norm': torch.mean(torch.norm(actions, dim=-1)).item(),
    }
    
    return total_loss, info


if __name__ == "__main__":
    # Test OneStepPolicy
    batch_size = 32
    state_dim = 17
    action_dim = 6
    
    print("Testing OneStepPolicy...")
    one_step_policy = OneStepPolicy(state_dim, action_dim)
    
    states = torch.randn(batch_size, state_dim)
    z = torch.randn(batch_size, action_dim)
    
    actions = one_step_policy(states)
    print(f"Generated actions shape: {actions.shape}")
    print(f"Expected: ({batch_size}, {action_dim})")
    
    
    # Test with automatic noise sampling
    actions_auto = one_step_policy(states)
    print(f"Auto-sampled actions shape: {actions_auto.shape}")
    
    # Test distillation loss (mock flow policy)
    print("\nTesting distillation loss...")
    from flow import FlowVelocityField, FlowPolicy
    
    velocity_field = FlowVelocityField(state_dim, action_dim)
    flow_policy = FlowPolicy(velocity_field, num_steps=10)
    
    loss, info = compute_distillation_loss(one_step_policy, flow_policy, states)
    print(f"Distillation loss: {loss.item():.4f}")
    print(f"Info: {info}")