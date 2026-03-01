"""
Critic (Q-network) for value estimation.
"""

import torch
import torch.nn as nn
from typing import Sequence, Optional
from fql.mlp import MLP, EnsembleMLP


class Critic(nn.Module):
    """Double Q-network with layer normalization."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512, 512),
        num_q_networks: int = 2,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            state_dim: Dimension of state observation
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            num_q_networks: Number of Q-networks (default: 2 for double Q-learning)
            use_layer_norm: Whether to use layer normalization (recommended)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.num_q_networks = num_q_networks
        
        # Input: [state, action]
        input_dim = state_dim + action_dim
        
        # Ensemble of Q-networks
        self.q_networks = EnsembleMLP(
            num_ensemble=num_q_networks,
            input_dim=input_dim,
            hidden_dims=list(hidden_dims) + [1],  # Output single Q-value
            activate_final=False,
            use_layer_norm=use_layer_norm,
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-values for state-action pairs.
        
        Args:
            state: State observations [batch_size, state_dim]
            action: Actions [batch_size, action_dim]
            
        Returns:
            q_values: Q-values [num_q_networks, batch_size, 1]
        """
        # Concatenate state and action
        inputs = torch.cat([state, action], dim=-1)
        
        # Compute Q-values from all networks
        q_values = self.q_networks(inputs)
        
        return q_values
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only the first Q-network output."""
        return self.forward(state, action)[0]
    
    def q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only the second Q-network output."""
        return self.forward(state, action)[1]
    
    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return minimum Q-value across networks (for conservative estimates)."""
        q_values = self.forward(state, action)
        return q_values.min(dim=0)[0]
    
    def q_mean(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return mean Q-value across networks."""
        q_values = self.forward(state, action)
        return q_values.mean(dim=0)


def compute_critic_loss(
    critic: Critic,
    target_critic: Critic,
    policy: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    use_clipped_target: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Bellman error loss for critic.
    
    L_Q(φ) = E[(Q_φ(s,a) - (r + γ*Q_φ̄(s',a')))²]
    
    Args:
        critic: Current Q-network
        target_critic: Target Q-network (EMA of critic)
        policy: Policy for sampling next actions
        states: Current states [batch_size, state_dim]
        actions: Current actions [batch_size, action_dim]
        rewards: Rewards [batch_size, 1]
        next_states: Next states [batch_size, state_dim]
        dones: Done flags [batch_size, 1]
        gamma: Discount factor
        use_clipped_target: Whether to use min(Q1, Q2) for target (clipped double Q)
        
    Returns:
        loss: Scalar critic loss
        info: Dictionary with loss components
    """
    batch_size = states.shape[0]
    
    # Current Q-values
    current_q = critic(states, actions)  # [num_q, batch_size, 1]
    
    # Next actions from policy (no gradient)
    with torch.no_grad():
        next_actions = policy(next_states)
        
        # Target Q-values
        target_q = target_critic(next_states, next_actions)  # [num_q, batch_size, 1]
        
        # Use mean or min for target value
        if use_clipped_target:
            # Clipped double Q-learning: min(Q1, Q2)
            target_q_value = target_q.min(dim=0)[0]
        else:
            # Mean of Q-networks
            target_q_value = target_q.mean(dim=0)
        
        # Bellman target: r + γ * (1 - done) * Q(s', a')
        target = rewards + gamma * (1 - dones) * target_q_value
    
    # MSE loss for each Q-network
    td_errors = current_q - target.unsqueeze(0)
    loss = (td_errors ** 2).mean()
    
    info = {
        'critic_loss': loss.item(),
        'current_q_mean': current_q.mean().item(),
        'current_q_std': current_q.std().item(),
        'target_q_mean': target.mean().item(),
        'td_error_mean': td_errors.abs().mean().item(),
        'reward_mean': rewards.mean().item(),
    }
    
    return loss, info


@torch.no_grad()
def update_target_network(
    source: nn.Module,
    target: nn.Module,
    tau: float = 0.005,
):
    """
    Soft update of target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        source: Source network (current)
        target: Target network (EMA)
        tau: Soft update coefficient (default: 0.005)
    """
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1 - tau) * target_param.data
        )


if __name__ == "__main__":
    # Test Critic
    batch_size = 32
    state_dim = 17
    action_dim = 6
    
    print("Testing Critic...")
    critic = Critic(state_dim, action_dim, num_q_networks=2)
    
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    q_values = critic(states, actions)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Expected: (2, {batch_size}, 1)")
    
    # Test individual Q-networks
    q1 = critic.q1(states, actions)
    q2 = critic.q2(states, actions)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
    
    # Test min and mean
    q_min = critic.q_min(states, actions)
    q_mean = critic.q_mean(states, actions)
    print(f"Q_min shape: {q_min.shape}, Q_mean shape: {q_mean.shape}")
    
    # Test critic loss
    print("\nTesting critic loss...")
    from policy import OneStepPolicy
    
    target_critic = Critic(state_dim, action_dim, num_q_networks=2)
    policy = OneStepPolicy(state_dim, action_dim)
    
    rewards = torch.randn(batch_size, 1)
    next_states = torch.randn(batch_size, state_dim)
    dones = torch.zeros(batch_size, 1)
    
    loss, info = compute_critic_loss(
        critic=critic,
        target_critic=target_critic,
        policy=policy,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        gamma=0.99,
        use_clipped_target=False,
    )
    
    print(f"Critic loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    # Test target network update
    print("\nTesting target network update...")
    print(f"Before update - Target Q1 bias: {target_critic.q_networks.networks[0].network[0].bias[0].item():.6f}")
    update_target_network(critic, target_critic, tau=0.005)
    print(f"After update - Target Q1 bias: {target_critic.q_networks.networks[0].network[0].bias[0].item():.6f}")