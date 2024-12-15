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
        
        # Validate parameters
        assert num_gcn_features > 0, "GCN features must be positive"
        assert num_rnn_hidden > 0, "RNN hidden size must be positive"
        assert k_paths > 0, "Number of paths must be positive"
        assert j > 0, "J parameter must be positive"
        
        # Set parameters before parent initialization
        self.num_gcn_features = num_gcn_features
        self.num_rnn_hidden = num_rnn_hidden
        self.k_paths = k_paths
        self.j = j


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

        # Calculate observation space size
        total_size = (self.topology.number_of_nodes() * self.topology.number_of_nodes() + 
                     self.topology.number_of_nodes() * self.num_spectrum_resources +
                     2 * self.topology.number_of_nodes() + 1 +
                     self.k_paths * self.num_spectrum_resources +
                     self.k_paths * self.NUM_SPECTRUM_METRICS)

        # Flattened observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )

        # Single discrete action space
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
        """Get observation with essential GCN/RNN features and return flattened array"""
        # Get topology size
        num_nodes = self.topology.number_of_nodes()
        num_edges = self.topology.number_of_edges()
        
        # 1. Topology Features
        # Adjacency matrix (num_edges x num_edges)
        adj_matrix = self.topology_transform.get_adjacency_matrix().flatten()
        # Node features (num_edges x num_spectrum_resources)
        node_features = self.topology_transform.get_node_features(
            self.topology.graph['available_slots']
        ).flatten()
        
        # 2. Service Features
        # Source-destination encoding (2 x num_nodes)
        source_destination_tau = np.zeros((2, num_nodes))
        source_destination_tau[0, self.current_service.source_id] = 1
        source_destination_tau[1, self.current_service.destination_id] = 1
        source_destination_tau = source_destination_tau.flatten()
        # Bit rate (1)
        bit_rate = np.array([self.current_service.bit_rate/100.0])
        
        # 3. Path Features
        paths = self.k_shortest_paths[
            self.current_service.source, 
            self.current_service.destination
        ][:self.k_paths]
        
        path_features = np.zeros((self.k_paths, self.num_spectrum_resources))
        for idx, path in enumerate(paths):
            if idx < len(paths):
                slots = self.get_available_slots(path)
                path_features[idx] = slots
        path_features = path_features.flatten()
        
        # 4. Spectrum Metrics (5 metrics per path)
        spectrum_metrics = np.zeros((self.k_paths, 5))
        for idx, path in enumerate(paths):
            if idx < len(paths):
                available_slots = self.get_available_slots(path)
                num_slots = self.get_number_slots(path)
                initial_indices, lengths = self.get_available_blocks(idx)
                
                spectrum_metrics[idx] = [
                    np.sum(available_slots)/self.num_spectrum_resources,  # normalized total available
                    len(initial_indices),  # num blocks
                    max(lengths, default=0)/self.num_spectrum_resources,  # normalized largest block
                    initial_indices[0]/self.num_spectrum_resources if len(initial_indices) > 0 else -1,  # normalized first block start
                    lengths[0]/self.num_spectrum_resources if len(lengths) > 0 else -1  # normalized first block size
                ]
        spectrum_metrics = spectrum_metrics.flatten()
        
        # Concatenate all features
        observation = np.concatenate([
            adj_matrix,           # num_edges * num_edges
            node_features,        # num_edges * num_spectrum_resources
            source_destination_tau,  # 2 * num_nodes
            bit_rate,               # 1
            path_features,          # k_paths * num_spectrum_resources
            spectrum_metrics        # k_paths * 5
        ]).astype(np.float32)
        
        # Print shape info for debugging
        print(f"Shape info:")
        print(f"Adjacency matrix: {adj_matrix.shape}")
        print(f"Node features: {node_features.shape}")
        print(f"Source-dest tau: {source_destination_tau.shape}")
        print(f"Bit rate: {bit_rate.shape}")
        print(f"Path features: {path_features.shape}")
        print(f"Spectrum metrics: {spectrum_metrics.shape}")
        print(f"Total observation: {observation.shape}")
        
        return observation
        
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
