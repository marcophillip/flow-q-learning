"""
Flow Q-Learning (FQL) Agent - Main implementation.

This agent combines:
1. BC Flow Policy (trained only with flow matching loss)
2. One-Step Policy (trained with Q-loss + distillation)
3. Critic (trained with Bellman error)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import copy

from fql.flow import FlowVelocityField, FlowPolicy, compute_flow_matching_loss
from fql.policy import OneStepPolicy, compute_one_step_policy_loss
from fql.critic import Critic, compute_critic_loss, update_target_network


class FQLAgent:
    """Flow Q-Learning Agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512, 512),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        bc_coef: float = 1.0,
        num_flow_steps: int = 10,
        use_clipped_target: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize FQL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions for all networks
            lr: Learning rate (default: 3e-4)
            gamma: Discount factor (default: 0.99)
            tau: Target network update rate (default: 0.005)
            bc_coef: Behavioral cloning coefficient α (default: 1.0)
            num_flow_steps: Number of Euler integration steps (default: 10)
            use_clipped_target: Whether to use min(Q1,Q2) for targets
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.bc_coef = bc_coef
        self.num_flow_steps = num_flow_steps
        self.use_clipped_target = use_clipped_target
        self.device = device

        # BC Flow Policy 
        self.velocity_field = FlowVelocityField(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(device)
        
        self.flow_policy = FlowPolicy(
            velocity_field=self.velocity_field,
            num_steps=num_flow_steps,
        ).to(device)
        
        # One-Step Policy
        self.one_step_policy = OneStepPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(device)
        
        #Critic
        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_q_networks=2,
            use_layer_norm=True,
        ).to(device)
        
        # Target critic
        self.target_critic = copy.deepcopy(self.critic).to(device)
        
        
        for param in self.target_critic.parameters():
            param.requires_grad = False
        
        
        self.velocity_field_optimizer = optim.Adam(
            self.velocity_field.parameters(),
            lr=lr,
        )
        
        self.one_step_policy_optimizer = optim.Adam(
            self.one_step_policy.parameters(),
            lr=lr,
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=lr,
        )

        self.total_steps = 0
    
    def select_action(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        select action for given state using one-step policy.
        
        Args:
            state: State observation [batch_size, state_dim] or [state_dim]            
        Returns:
            action: Selected action [batch_size, action_dim] or [action_dim]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0) 
       
        with torch.no_grad():
            action = self.one_step_policy(state)

        return action
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one update step using a batch of transitions.
        
        Args:
            batch: Dictionary containing:
                - 'states': [batch_size, state_dim]
                - 'actions': [batch_size, action_dim]
                - 'rewards': [batch_size, 1]
                - 'next_observations': [batch_size, state_dim]
                - 'dones': [batch_size, 1]
                
        """
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)
        
        info = {}

        # update critic  
        critic_loss, critic_info = compute_critic_loss(
            critic=self.critic,
            target_critic=self.target_critic,
            policy=self.one_step_policy,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=self.gamma,
            use_clipped_target=self.use_clipped_target,
        )
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        info.update(critic_info)
        
        #  Update BC Flow Policy
        flow_loss, flow_info = compute_flow_matching_loss(
            velocity_field=self.velocity_field,
            states=states,
            actions=actions,
        )
        
        self.velocity_field_optimizer.zero_grad()
        flow_loss.backward()
        self.velocity_field_optimizer.step()
        
        info.update(flow_info)
        
        # Update One-Step Policy 
        # Train with Q-loss + distillation from flow policy

        policy_loss, policy_info = compute_one_step_policy_loss(
            one_step_policy=self.one_step_policy,
            flow_policy=self.flow_policy,
            critic=self.critic,
            states=states,
            bc_coef=self.bc_coef,
        )
        
        self.one_step_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.one_step_policy_optimizer.step()
        
        info.update(policy_info)
        
        # Update Target Networks
        update_target_network(
            source=self.critic,
            target=self.target_critic,
            tau=self.tau,
        )
        
        self.total_steps += 1
        info['total_steps'] = self.total_steps
        
        return info
    
    def save(self, filepath: str="fql_agent.pt"):
        """
        Save agent state.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'velocity_field': self.velocity_field.state_dict(),
            'one_step_policy': self.one_step_policy.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'velocity_field_optimizer': self.velocity_field_optimizer.state_dict(),
            'one_step_policy_optimizer': self.one_step_policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dims': self.hidden_dims,
                'lr': self.lr,
                'gamma': self.gamma,
                'tau': self.tau,
                'bc_coef': self.bc_coef,
                'num_flow_steps': self.num_flow_steps,
                'use_clipped_target': self.use_clipped_target,
            }
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """
        Load agent state.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.velocity_field.load_state_dict(checkpoint['velocity_field'])
        self.one_step_policy.load_state_dict(checkpoint['one_step_policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        
        self.velocity_field_optimizer.load_state_dict(checkpoint['velocity_field_optimizer'])
        self.one_step_policy_optimizer.load_state_dict(checkpoint['one_step_policy_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        self.total_steps = checkpoint['total_steps']
    
    def train(self):
        """Set agent to training mode."""
        self.velocity_field.train()
        self.one_step_policy.train()
        self.critic.train()
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.velocity_field.eval()
        self.one_step_policy.eval()
        self.critic.eval()
    
