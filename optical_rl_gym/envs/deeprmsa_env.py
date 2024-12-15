from typing import Tuple

import gym
import numpy as np

from .rmsa_env import RMSAEnv
from optical_rl_gym.topo_transform import TopologyTransform, SpectrumAvailability

class DeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        topology=None,
        j=1,
        episode_length=1000,
        mean_service_holding_time=10.0,
        mean_service_inter_arrival_time=None,
        num_spectrum_resources=100,
        node_request_probabilities=None,
        seed=None,
        allow_rejection=False,
        k_paths=5,
        num_gcn_features=32,  # New parameter for GCN feature size
        num_rnn_hidden=64,    # New parameter for RNN hidden size
        
    ):
        # Extract GCN/RNN specific parameters first
        self.num_gcn_features = num_gcn_features
        self.num_rnn_hidden = num_rnn_hidden

        super().__init__(
            topology=topology,
            episode_length=episode_length,
            load = mean_service_holding_time / mean_service_inter_arrival_time, # correct definitaion, verified xd950
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            reset=False,
        )

        # Initialize parameters
        self.j = j
        self.k_paths = k_paths
        self.num_gcn_features = num_gcn_features
        self.num_rnn_hidden = num_rnn_hidden

        # Add spectrum metrics constants here
        self.NUM_SPECTRUM_METRICS = 6  # Number of spectrum metrics per path
        self.spectrum_metric_names = [
            'total_available',
            'num_blocks',
            'largest_block',
            'first_block_start',
            'first_block_size',
            'required_slots'
        ]

        # Define new observation space for GCN-RNN
        self.observation_space = gym.spaces.Dict({
            # GCN features
            "topology": gym.spaces.Dict({
                "adjacency_matrix": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.topology_transform.transformed_topology.number_of_nodes(),
                          self.topology_transform.transformed_topology.number_of_nodes()),
                    dtype=np.int8
                ),
                "node_features": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.topology_transform.transformed_topology.number_of_nodes(),
                          self.num_spectrum_resources),
                    dtype=np.int8
                )
            }),
            # Path features for RNN
            "paths": gym.spaces.Dict({
                "features": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.k_paths, self.num_rnn_hidden),
                    dtype=np.float32
                ),
                "lengths": gym.spaces.Box(
                    low=0,
                    high=self.topology.number_of_nodes(),
                    shape=(self.k_paths,),
                    dtype=np.int32
                )
            }),
            # Spectrum metrics
            "spectrum": gym.spaces.Dict({
                "metrics": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.k_paths, 6),  # 6 metrics per path
                    dtype=np.float32
                )
            }),
            # Service features
            "service": gym.spaces.Dict({
                "source_destination_tau": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(2, self.topology.number_of_nodes()),
                    dtype=np.int8
                ),
                "bit_rate": gym.spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32
                )
            })
        })

        self.action_space = gym.spaces.Discrete(
            self.k_paths * self.j + self.reject_action
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        
        self.reset(only_episode_counters=False)

        #print("Load in DeepRMSA", self.load)

    def step(self, action: int):
        """Process single action into path and block selection"""
        if action < self.k_paths * self.j:
            route, block = self._get_route_block_id(action)
            initial_indices, lengths = self.get_available_blocks(route)
            
            if block < len(initial_indices):
                return super().step([route, initial_indices[block]])
                
        return super().step([self.k_paths, self.num_spectrum_resources])

    
    def observation(self):
        """Get observation using parent class implementation"""
        obs = super().observation()
        
        # Process paths through RNN if needed
        path_features = obs['paths']['features']
        processed_features = self._process_path_features(path_features)
        
        # Update path features with RNN processed version
        obs['paths']['features'] = processed_features
        
        return obs
    
    def _process_path_features(self, path_features: np.ndarray) -> np.ndarray:
        """Process path features through RNN-like aggregation"""
        # This is a placeholder for actual RNN processing
        # In practice, you would use a proper RNN here
        processed_features = np.zeros((self.k_paths, self.num_rnn_hidden))
        
        for i, features in enumerate(path_features):
            if np.any(features):  # If path exists
                # Simple feature aggregation (replace with actual RNN)
                processed_features[i] = np.mean(features) * np.ones(self.num_rnn_hidden)
                
        return processed_features


    def reward(self):
        return 1 if self.current_service.accepted else -1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def _get_route_block_id(self, action: int) -> Tuple[int, int]:
        route = action // self.j
        block = action % self.j
        return route, block




def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, _ = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, _ in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        initial_indices, _ = env.get_available_blocks(idp)
        if len(initial_indices) > 0:  # if there are available slots
            return idp * env.j  # this path uses the first one
    return env.k_paths * env.j
