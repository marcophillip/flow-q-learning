import torch
import numpy as np
from torch.utils.data import Dataset
from data.replay_buffer import AntMazeReplayBuffer



class AntMazeDataset(Dataset):
    
    def __init__(self, replay_buffer: AntMazeReplayBuffer):
        self.replay_buffer = replay_buffer
        self.transitions = []
        
        # Flatten episodes into transitions
        for episode in replay_buffer.episodes:
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            next_states = episode['next_states']
            dones = episode['dones']
            
            for i in range(len(states)):
                self.transitions.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.transitions[idx]
        return {
            'state': torch.tensor(state, dtype=torch.float32),
            'action': torch.tensor(action, dtype=torch.float32),
            'reward': torch.tensor(reward, dtype=torch.float32),
            'next_state': torch.tensor(next_state, dtype=torch.float32),
            'done': torch.tensor(done, dtype=torch.float32)
        }