import torch
import numpy as np
from typing import Dict
import time
import os
import sys
from pathlib import Path

print(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

print(sys.path)


from fql.fql_agent import FQLAgent
from data.dataset import ReplayBuffer

import gymnasium as gym
import gymnasium_robotics
import torch

gym.register_envs(gymnasium_robotics)


class OfflineTrainer:
    
    def __init__(
        self,
        agent: FQLAgent,
        dataset: ReplayBuffer,
        eval_env,
        log_dir: str = "./logs",
        save_dir: str = "./checkpoints",
        eval_episodes: int = 2,
        eval_frequency: int = 10000,
        save_frequency: int = 100000,
        log_frequency: int = 5,
        batch_size: int = 2048,
    ):
        """        
        Args:
            agent: FQL agent to train
            eval_env: Environment for evaluation
            log_dir: Directory for logs
            save_dir: Directory for checkpoints
            eval_episodes: Number of episodes for evaluation
            eval_frequency: Evaluate every N gradient steps
            save_frequency: Save checkpoint every N gradient steps
            log_frequency: Log training metrics every N gradient steps
            batch_size: Batch size for training
            verbose: Whether to print training progress
        """
        self.agent = agent
        self.dataset = dataset
        self.eval_env = eval_env

        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.eval_episodes = eval_episodes
        self.eval_frequency = eval_frequency
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.batch_size = batch_size


        # Training state
        self.total_steps = 0
        self.start_time = time.time()
        
        # Metrics storage
        self.train_metrics = []
        self.eval_metrics = []

        self.dataset.collect_episodes(num_episodes=10, max_steps=10000) # collect initial dataset
    
    def train(
        self,
        num_steps: int,
        initial_eval: bool = False,
    ) -> Dict[str, list]:
        """
        Train the agent for a specified number of gradient steps.
        
        Args:
            num_steps: Number of gradient steps
            initial_eval: Whether to evaluate before training

        """
        if initial_eval:
            self._evaluate(step=0)
        
        self.agent.train()
        
        for step in range(1, num_steps + 1):
            # Sample batch from offline dataset and update
            print(f"Sampling batch for training step {step}...")
            batch = self.dataset.sample(self.batch_size)
            
            update_info = self.agent.update(batch)
            
            self.total_steps += 1
            
            # Log training metrics
            if step % self.log_frequency == 0:
                self._log_train_metrics(update_info, step)
         
            # Evaluate
            if step % self.eval_frequency == 0:
                self._evaluate(step)

            #collect new trajectories based on the policy
            if (num_steps // 20) == 0 and step > 0:
                
                print(f"Collecting {20} episodes with behaviour policy...")
                self.dataset.collect_episodes(
                    num_episodes=100,
                    max_steps=1000,
                    policy=self.agent.select_action,
                )
            
            # # Save checkpoint
            # if step % self.save_frequency == 0:
            #     self._save_checkpoint(step)
        
        # Final evaluation and save
        self._evaluate(num_steps)
        self._save_checkpoint(num_steps, final=True)
        
        return {
            'train_metrics': self.train_metrics,
            'eval_metrics': self.eval_metrics,
        }
    
    def _evaluate(self, step: int) -> Dict[str, float]:
        """Evaluate agent."""
        self.agent.eval()
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for _ in range(self.eval_episodes):
            observation, info = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            truncated = False
            
            while not done and not truncated:
                # Select action
                state_tensor = torch.from_numpy(observation).float().to(self.agent.device)
                with torch.no_grad():
                    action = self.agent.select_action(state_tensor).squeeze(0) 
                action = action.cpu().numpy()
                
                # Step environment
                observation, reward, done, truncated, _ = self.eval_env.step(action)
               
                episode_reward += reward
                episode_length += 1
                
                # Check for success (if available in info)
                if done and 'success' in info:
                    success_count += info['success']
            
            print(f"Episode reward: {episode_reward:.2f} | Length: {episode_length}")
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Compute statistics
        eval_stats = {
            'step': step,
            'mean_return': np.mean(episode_rewards),
            'std_return': np.std(episode_rewards),
            'min_return': np.min(episode_rewards),
            'max_return': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
        }
        
        self.eval_metrics.append(eval_stats)
        
      
        elapsed_time = time.time() - self.start_time
        print(f"\n[Eval @ step {step:,} | {elapsed_time:.1f}s]")
        print(f"  Return: {eval_stats['mean_return']:.2f} ± {eval_stats['std_return']:.2f} "
                f"[{eval_stats['min_return']:.2f}, {eval_stats['max_return']:.2f}]")

        
        self.agent.train()
        
        return eval_stats
    
    def _log_train_metrics(self, update_info: Dict[str, float], step: int):
        """Log training metrics."""
        metrics = {
            'step': step,
            'total_steps': self.total_steps,
            **update_info,
        }
        
        self.train_metrics.append(metrics)
        
        elapsed_time = time.time() - self.start_time
        steps_per_sec = self.total_steps / elapsed_time if elapsed_time > 0 else 0
        
        print(f"[Train @ step {step:,} | {elapsed_time:.1f}s | {steps_per_sec:.1f} steps/s]")
        print(f"  Critic Loss: {metrics['critic_loss']:.4f} | "
                f"Q Mean: {metrics['current_q_mean']:.2f}")
        print(f"  Flow Loss: {metrics['flow_loss']:.4f}")
        print(f"  Policy Loss: {metrics['total_policy_loss']:.4f} | "
                f"Q Loss: {metrics['q_loss']:.4f} | "
                f"Distill Loss: {metrics['distill_loss']:.4f}")
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """Save checkpoint."""
        if final:
            checkpoint_path = self.save_dir / "final_checkpoint.pt"
        else:
            checkpoint_path = self.save_dir / f"checkpoint_{step}.pt"
        
        self.agent.save(str(checkpoint_path))
        
        # Also save training metrics
        metrics_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metrics.pt"
        torch.save({
            'train_metrics': self.train_metrics,
            'eval_metrics': self.eval_metrics,
            'total_steps': self.total_steps,
        }, metrics_path)
        
        if self.verbose:
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        self.agent.load(checkpoint_path)
        
        # Load metrics if available
        metrics_path = Path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}_metrics.pt"
        if metrics_path.exists():
            metrics = torch.load(metrics_path)
            self.train_metrics = metrics['train_metrics']
            self.eval_metrics = metrics['eval_metrics']
            self.total_steps = metrics['total_steps']
            
            if self.verbose:
                print(f"Loaded checkpoint from step {self.total_steps}")




if __name__ == "__main__":

    env = gym.make('HalfCheetah-v5')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    buffer =ReplayBuffer(env = env, capacity=1000000)

    fql_agent = FQLAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )

    trainer = OfflineTrainer(
        agent=fql_agent,
        dataset=buffer,
        eval_env=env
    )

    trainer.train(num_steps=10)
