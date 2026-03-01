import numpy as np
from typing import Dict, List, Optional
import gymnasium as gym
import gymnasium_robotics
import torch

gym.register_envs(gymnasium_robotics)


class ReplayBuffer:
    
    def __init__(self, env, capacity: int = 1000000):

        self.env = env

        self.capacity = capacity
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.total_transitions = 0
        
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._dones = []
        
    def add_episode(self, 
                   states: np.ndarray,      # Shape: (T, state_dim)
                   actions: np.ndarray,     # Shape: (T, action_dim) 
                   rewards: np.ndarray,     # Shape: (T,)
                   next_states: np.ndarray, # Shape: (T, state_dim)
                   dones: np.ndarray,       # Shape: (T,)
                   info: Optional[Dict] = None):
        """
        Add a complete episode to the buffer.
        
        Args:
            states: State trajectory, shape (episode_length, state_dim)
            actions: Action trajectory, shape (episode_length, action_dim)
            rewards: Reward trajectory, shape (episode_length,)
            next_states: Next state trajectory, shape (episode_length, state_dim)
            dones: Done flags, shape (episode_length,)
            info: Optional metadata (success, goal, etc.)
        """
        episode = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
            'length': len(states),
            'info': info or {}
        }
        
        self.episodes.append(episode)
        self.total_transitions += len(states)
        
        #drop oldest episodes
        while self.total_transitions > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_transitions -= removed['length']
    
    def collect_episode(self, policy=None, max_steps=1000, render=False):
        """
        Collect a single episode from the environment and append it to the buffer.
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []

        state, info = self.env.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < max_steps:
            if policy is None:
                action = self.env.action_space.sample()
            else:
                state_ = torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(0)

                # state_ = state_.to("cuda" if torch.cuda.is_available() else "cpu")

                with torch.no_grad():
                    action_tensor = policy(state_)

                # ensure numpy on cpu
                action = action_tensor.cpu().numpy().squeeze(0)

            next_state, reward, done, truncated, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)

            if render:
                self.env.render()

            state = next_state
            step += 1

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        self.add_episode(states, actions, rewards, next_states, dones, info)
       
    
    def collect_episodes(self, num_episodes: int, **kwargs):
        """
        Collect multiple episodes.
        """
        for _ in range(num_episodes):
            self.collect_episode(**kwargs)
         
    def _build_flat_buffer(self):
            
        if not self.episodes:
            raise ValueError("Buffer is empty!")
        
        # Concatenate all episodes
        self._states = np.concatenate([ep['states'] for ep in self.episodes])
        self._actions = np.concatenate([ep['actions'] for ep in self.episodes])
        self._rewards = np.concatenate([ep['rewards'] for ep in self.episodes])
        self._next_states = np.concatenate([ep['next_states'] for ep in self.episodes])
        self._dones = np.concatenate([ep['dones'] for ep in self.episodes])

        self._states = torch.tensor(self._states, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._actions = torch.tensor(self._actions, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._rewards = torch.tensor(self._rewards, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._next_states = torch.tensor(self._next_states, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._dones = torch.tensor(self._dones, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample random transitions for training.
        """
        self._build_flat_buffer()
        
        indices = np.random.randint(0, len(self._states), size=batch_size)
        indices = torch.tensor(indices, dtype=torch.long, device=self._states.device)
        
        return {
            'states': self._states[indices],
            'actions': self._actions[indices],
            'rewards': self._rewards[indices],
            'next_states': self._next_states[indices],
            'dones': self._dones[indices]
        }
    
    def sample_episode(self) -> Dict[str, np.ndarray]:
        """Sample a random complete episode (for episode-level training)."""
        idx = np.random.randint(len(self.episodes))
        return self.episodes[idx]
    
    
    def __len__(self):
        return self.total_transitions
    

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())

    from fql.policy import OneStepPolicy
    
    # example_map = [[1, 1, 1, 1, 1],
    #         [1, 0, 0, 0, 1],  
    #         [1, 0, 0, 0, 1],       
    #         [1, 1, 1, 1, 1]]

    # env = gym.make('AntMaze_UMazeDense-v5', 
    #                maze_map=example_map, 
    #                continuing_task=True, 
    #             #    max_episode_steps=10000
    #                )

    env = gym.make('HalfCheetah-v5')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    one_step_policy = OneStepPolicy(state_dim=state_dim, action_dim=action_dim)
    
    buffer =ReplayBuffer(env = env, capacity=1000000)
    buffer.collect_episodes(num_episodes=10, max_steps=10000, policy=one_step_policy)
    print(f"Collected {len(buffer)} transitions across {len(buffer.episodes)} episodes.")
    