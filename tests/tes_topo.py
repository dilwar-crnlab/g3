import gym
import networkx as nx
import numpy as np
from optical_rl_gym.utils import Path, Modulation  # Import both Path and Modulation
from typing import List, Tuple

def create_test_topology():
    """Create a simple test topology"""
    G = nx.Graph()
    # Add nodes
    for i in range(4):
        G.add_node(i)
    
    # Add edges with indices and lengths
    edges = [(0,1), (1,2), (2,3), (0,2)]
    for idx, (i,j) in enumerate(edges):
        G.add_edge(i, j, index=idx, length=100)  # Adding length attribute
    
    # Add required attributes
    G.graph['name'] = 'test_topology'
    G.graph['node_indices'] = list(range(4))
    G.graph['k_paths'] = 2  # Number of paths per node pair
    
    # Create modulation for the paths
    test_modulation = Modulation(
        name="QPSK",
        maximum_length=2500,
        spectral_efficiency=2
    )
    
    # Create k-shortest paths dictionary
    k_paths_dict = {}
    path_id = 0
    for i in range(4):
        for j in range(4):
            if i != j:
                # Create Path objects with required attributes
                path1 = Path(
                    path_id=path_id,
                    node_list=(str(i), str(j)),  # Convert to strings
                    hops=1,
                    length=100,
                    best_modulation=test_modulation
                )
                path_id += 1
                
                path2 = Path(
                    path_id=path_id,
                    node_list=(str(i), str((i+1)%4), str(j)),  # Convert to strings
                    hops=2,
                    length=200,
                    best_modulation=test_modulation
                )
                path_id += 1
                
                k_paths_dict[(str(i),str(j))] = [path1, path2]  # Use string keys
    
    # Add ksp dictionary to graph (for path lookup)
    G.graph['ksp'] = k_paths_dict
    
    # Add modulations to topology
    G.graph['modulations'] = [test_modulation]
    
    return G

def test_environments():
    """Test both RMSA and DeepRMSA environments"""
    # Create test topology
    topology = create_test_topology()
    
    # Create environments with correct environment IDs
    rmsa_env = gym.make('RMSA-v0',
                        topology=topology,
                        episode_length=10,
                        mean_service_holding_time=10.0,
                        mean_service_inter_arrival_time=1.0)
    
    deep_rmsa_env = gym.make('DeepRMSA-v0',
                            topology=topology,
                            episode_length=10,
                            mean_service_holding_time=10.0,
                            mean_service_inter_arrival_time=1.0,
                            num_gcn_features=32,
                            num_rnn_hidden=64)
    
    # Test reset
    rmsa_obs = rmsa_env.reset()
    deep_rmsa_obs = deep_rmsa_env.reset()
    
    # Verify observation spaces match
    print("Testing observation spaces...")
    assert rmsa_obs.keys() == deep_rmsa_obs.keys(), "Observation keys don't match"
    
    # Test step
    action = [0, 0]  # Try first path, first slot
    rmsa_next_obs, rmsa_reward, rmsa_done, rmsa_info = rmsa_env.step(action)
    
    deep_action = 0  # First action in combined space
    deep_next_obs, deep_reward, deep_done, deep_info = deep_rmsa_env.step(deep_action)
    
    # Print some metrics
    print("\nRMSA Metrics:")
    print(f"Service blocking rate: {rmsa_info['service_blocking_rate']}")
    print(f"Network compactness: {rmsa_info['network_compactness']}")
    
    print("\nDeepRMSA Metrics:")
    print(f"Service blocking rate: {deep_info['service_blocking_rate']}")
    print(f"Network compactness: {deep_info['network_compactness']}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_environments()
