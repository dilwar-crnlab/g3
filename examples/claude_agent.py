import os
import pickle
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import load_results, ts2xy
import gym
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
from IPython.display import clear_output

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for DeepRMSA environment"""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Now we expect a Box space instead of Dict
        super().__init__(observation_space, features_dim)
        
        # Get input dimension from the observation space
        n_input_features = np.prod(observation_space.shape)
        
        # Feature extraction layers
        self.shared_net = nn.Sequential(
            nn.Linear(n_input_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.shared_net(observations.float())

class MetricsCallback(BaseCallback):
    """Callback for monitoring detailed metrics"""
    def __init__(self, check_freq: int, log_dir: str):
        super().__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, 'detailed_metrics.csv')
        self.metrics = []
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            env = self.training_env.envs[0]
            info = env.get_info()
            
            metrics = {
                'timestep': self.n_calls,
                'service_blocking_rate': info['service_blocking_rate'],
                'bit_rate_blocking_rate': info['bit_rate_blocking_rate'],
                'network_compactness': info['network_compactness'],
                'avg_link_utilization': info['avg_link_utilization']
            }
            self.metrics.append(metrics)
            
            # Save metrics to CSV
            pd.DataFrame(self.metrics).to_csv(self.metrics_path)
        return True

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """Callback for saving the best model during training"""
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(f"{self.save_path}/best_model_{mean_reward:.2f}")
                    
                if self.verbose > 0:
                    print(f"Step: {self.n_calls}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f}")
                    print(f"Current mean reward: {mean_reward:.2f}")
        return True

def create_env(topology, env_args, log_dir, is_eval=False):
    """Create and wrap the environment"""
    env = gym.make('DeepRMSA-v0', **env_args)
    
    # Calculate observation size
    num_nodes = topology.number_of_nodes()
    num_edges = topology.number_of_edges()
    k_paths = env_args['k_paths']
    num_spectrum_resources = env_args['num_spectrum_resources']
    
    # Calculate size components
    adj_matrix_size = num_edges * num_edges
    node_features_size = num_edges * num_spectrum_resources
    service_tau_size = 2 * num_nodes
    bit_rate_size = 1
    path_features_size = k_paths * num_spectrum_resources
    spectrum_metrics_size = k_paths * 5  # 5 metrics per path
    
    total_size = (adj_matrix_size + node_features_size + service_tau_size + 
                 bit_rate_size + path_features_size + spectrum_metrics_size)
    
    print(f"\nObservation space components:")
    print(f"Adjacency matrix: {adj_matrix_size}")
    print(f"Node features: {node_features_size}")
    print(f"Service tau: {service_tau_size}")
    print(f"Bit rate: {bit_rate_size}")
    print(f"Path features: {path_features_size}")
    print(f"Spectrum metrics: {spectrum_metrics_size}")
    print(f"Total size: {total_size}\n")
    
    # Create monitor wrapper
    monitor_path = os.path.join(log_dir, 'training' if not is_eval else 'eval')
    env = Monitor(
        env, 
        monitor_path,
        info_keywords=(
            'episode_service_blocking_rate',
            'episode_bit_rate_blocking_rate',
            'network_compactness',
            'avg_link_utilization'
        )
    )
    return env

def train_model(env, log_dir: str, total_timesteps: int = 1_000_000):
    """Train the DQN model"""
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 128],
        activation_fn=th.nn.ReLU
    )
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.15,
        exploration_final_eps=0.02,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        tensorboard_log=os.path.join(log_dir, "tb_logs"),
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Setup callbacks
    callbacks = [
        MetricsCallback(check_freq=1000, log_dir=log_dir),
        SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    ]
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    
    return model

def evaluate_model(model, eval_env, n_eval_episodes: int = 10):
    """Evaluate the trained model"""
    episode_rewards = []
    episode_blocking_rates = []
    episode_bit_rate_blocking_rates = []
    
    for episode in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        episode_blocking_rates.append(info['episode_service_blocking_rate'])
        episode_bit_rate_blocking_rates.append(info['episode_bit_rate_blocking_rate'])
        
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_blocking_rate': np.mean(episode_blocking_rates),
        'mean_bit_rate_blocking': np.mean(episode_bit_rate_blocking_rates)
    }
    
    return results

def main():
    # Setup
    topology_name = 'dutch'
    k_paths = 3
    log_dir = f"./results/deeprmsa_dqn_{topology_name}_{k_paths}paths/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load topology
    with open(f'../topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
        topology = pickle.load(f)
    
    # Environment arguments
    env_args = dict(
        topology=topology,
        seed=10,
        allow_rejection=True,
        j=1,
        mean_service_holding_time=10.0,
        mean_service_inter_arrival_time=0.1,
        episode_length=100,
        k_paths=k_paths,
        num_spectrum_resources=100,
        num_gcn_features=32,
        num_rnn_hidden=64
    )
    
    # Create environments
    train_env = create_env(topology, env_args, log_dir, is_eval=False)
    eval_env = create_env(topology, env_args, log_dir, is_eval=True)
    
    # Train model
    model = train_model(train_env, log_dir)
    
    # Evaluate model
    results = evaluate_model(model, eval_env)
    
    # Save results
    results_path = os.path.join(log_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Blocking Rate: {results['mean_blocking_rate']:.4f}")
    print(f"Mean Bit Rate Blocking: {results['mean_bit_rate_blocking']:.4f}")
    
    # Save final model
    model.save(os.path.join(log_dir, "final_model"))

if __name__ == "__main__":
    main()