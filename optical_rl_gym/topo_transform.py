# import networkx as nx
# import numpy as np
# from typing import Dict, Tuple, List

# class TopologyTransform:
#     def __init__(self, topology: nx.Graph):
#         self.original_topology = topology
#         self.edge_to_node_map = {}  # Maps original edges to transformed nodes
#         self.node_to_edge_map = {}  # Maps transformed nodes back to original edges
#         self.transformed_topology = self._transform_topology()

#     def _transform_topology(self) -> nx.DiGraph:
#         """Transform topology where links become nodes"""
#         transformed = nx.DiGraph()
        
#         # Create nodes from original edges
#         for idx, (i, j) in enumerate(self.original_topology.edges()):
#             edge_id = self.original_topology[i][j]['index']
#             self.edge_to_node_map[(i, j)] = edge_id
#             self.node_to_edge_map[edge_id] = (i, j)
            
#             # Add node with original edge attributes
#             transformed.add_node(edge_id, 
#                                original_edge=(i, j),
#                                spectrum_slots=self.original_topology[i][j].get('slots', []))
        
#         # Add edges between transformed nodes if they're connected in paths
#         self._add_path_connections(transformed)
        
#         return transformed
    
#     def _add_path_connections(self, transformed: nx.DiGraph):
#         """Add edges between nodes if they represent connected links in original paths"""
#         for src, dst in self.original_topology.graph['k_paths'].keys():
#             paths = self.original_topology.graph['k_paths'][(src, dst)]
#             for path in paths:
#                 # Get consecutive pairs of edges in path
#                 path_edges = list(zip(path.node_list[:-1], path.node_list[1:]))
#                 for idx in range(len(path_edges)-1):
#                     edge1 = path_edges[idx]
#                     edge2 = path_edges[idx+1]
#                     # Add directed edge in transformed graph
#                     if edge1 in self.edge_to_node_map and edge2 in self.edge_to_node_map:
#                         node1 = self.edge_to_node_map[edge1]
#                         node2 = self.edge_to_node_map[edge2]
#                         transformed.add_edge(node1, node2)
    
#     def get_path_nodes(self, path: List[int]) -> List[int]:
#         """Convert path from original topology to sequence of nodes in transformed topology"""
#         path_edges = list(zip(path[:-1], path[1:]))
#         return [self.edge_to_node_map[edge] for edge in path_edges]

#     def get_adjacency_matrix(self) -> np.ndarray:
#         """Get adjacency matrix of transformed topology"""
#         return nx.adjacency_matrix(self.transformed_topology).todense()

#     def get_node_features(self, spectrum_slots: np.ndarray) -> np.ndarray:
#         """Get node features matrix (spectrum usage for each transformed node)"""
#         num_nodes = self.transformed_topology.number_of_nodes()
#         features = np.zeros((num_nodes, spectrum_slots.shape[1]))
        
#         for node in self.transformed_topology.nodes():
#             i, j = self.node_to_edge_map[node]
#             edge_idx = self.original_topology[i][j]['index']
#             features[node] = spectrum_slots[edge_idx]
            
#         return features
    

# class SpectrumAvailability:
#     def __init__(self, num_spectrum_resources: int):
#         self.num_spectrum_resources = num_spectrum_resources

#     def get_metrics(self, path: Path, topology: nx.Graph, required_slots: int) -> dict:
#         """
#         Calculate spectrum availability metrics for a path
#         Args:
#             path: Path object containing node sequence
#             topology: Network topology
#             required_slots: Number of contiguous slots needed
#         Returns:
#             Dictionary containing spectrum metrics
#         """
#         # Get common available slots across path
#         path_slots = self._get_path_slots(path, topology)
        
#         # Find contiguous blocks
#         blocks = self._get_contiguous_blocks(path_slots)
        
#         # Calculate metrics
#         nfs = np.sum(path_slots)  # Total number of free slots
#         nfsb = len(blocks)  # Number of free slot blocks
        
#         # Count valid blocks (satisfying required_slots)
#         valid_blocks = [block for block in blocks if block[1] >= required_slots]
#         nfsb_valid = len(valid_blocks)
        
#         # Get first-fit block properties
#         if valid_blocks:
#             istart = valid_blocks[0][0]  # Starting index of first valid block
#             sfirst = valid_blocks[0][1]  # Size of first valid block
#         else:
#             istart = -1
#             sfirst = -1
            
#         # Calculate average block size
#         sfsb = np.mean([block[1] for block in blocks]) if blocks else 0
        
#         return {
#             'nfs': nfs,
#             'nfsb': nfsb,
#             'nfsb_valid': nfsb_valid,
#             'istart': istart,
#             'sfirst': sfirst,
#             'sfsb': sfsb
#         }
    
#     def _get_path_slots(self, path: Path, topology: nx.Graph) -> np.ndarray:
#         """Get common available slots across all links in path"""
#         # Initialize with all slots available
#         common_slots = np.ones(self.num_spectrum_resources, dtype=bool)
        
#         # Get consecutive node pairs in path
#         path_edges = list(zip(path.node_list[:-1], path.node_list[1:]))
        
#         # Update common slots based on each link's availability
#         for i, j in path_edges:
#             edge_idx = topology[i][j]['index']
#             edge_slots = topology.graph['available_slots'][edge_idx]
#             common_slots &= edge_slots
            
#         return common_slots
    
#     def _get_contiguous_blocks(self, slots: np.ndarray) -> List[Tuple[int, int]]:
#         """
#         Find contiguous blocks of available spectrum
#         Returns list of tuples (start_index, block_size)
#         """
#         blocks = []
#         block_start = None
        
#         for i in range(len(slots)):
#             if slots[i]:  # Slot is available
#                 if block_start is None:
#                     block_start = i
#             elif block_start is not None:  # End of block
#                 block_size = i - block_start
#                 blocks.append((block_start, block_size))
#                 block_start = None
                
#         # Handle last block if it extends to the end
#         if block_start is not None:
#             block_size = len(slots) - block_start
#             blocks.append((block_start, block_size))
            
#         return blocks




import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from optical_rl_gym.utils import Path

class TopologyTransform:
    def __init__(self, topology: nx.Graph):
        self.original_topology = topology
        self.edge_to_node_map = {}  # Maps original edges to transformed nodes
        self.node_to_edge_map = {}  # Maps transformed nodes back to original edges
        self.transformed_topology = self._transform_topology()

    def _transform_topology(self) -> nx.DiGraph:
        """Transform topology where links become nodes"""
        transformed = nx.DiGraph()
        
        # Create nodes from original edges
        for idx, (i, j) in enumerate(self.original_topology.edges()):
            edge_id = self.original_topology[i][j]['index']
            self.edge_to_node_map[(i, j)] = edge_id
            self.node_to_edge_map[edge_id] = (i, j)
            
            # Add node with original edge attributes
            transformed.add_node(edge_id, 
                            original_edge=(i, j),
                            spectrum_slots=np.ones(self.original_topology.graph['num_spectrum_resources']))

        # Add edges between transformed nodes if they're connected in paths
        # Use 'ksp' instead of 'k_paths'
        for src, dst in self.original_topology.graph['ksp'].keys():
            paths = self.original_topology.graph['ksp'][(src, dst)]
            for path in paths:
                # Get consecutive pairs of edges in path
                path_edges = list(zip(path.node_list[:-1], path.node_list[1:]))
                for idx in range(len(path_edges)-1):
                    edge1 = path_edges[idx]
                    edge2 = path_edges[idx+1]
                    if edge1 in self.edge_to_node_map and edge2 in self.edge_to_node_map:
                        node1 = self.edge_to_node_map[edge1]
                        node2 = self.edge_to_node_map[edge2]
                        transformed.add_edge(node1, node2)
        
        return transformed

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix of transformed topology"""
        return nx.adjacency_matrix(self.transformed_topology).todense()

    def get_node_features(self, spectrum_slots: np.ndarray) -> np.ndarray:
        """Get node features matrix (spectrum usage for transformed nodes)"""
        num_nodes = self.transformed_topology.number_of_nodes()
        features = np.zeros((num_nodes, spectrum_slots.shape[1]))
        
        for node in self.transformed_topology.nodes():
            i, j = self.node_to_edge_map[node]
            edge_idx = self.original_topology[i][j]['index']
            features[node] = spectrum_slots[edge_idx]
            
        return features

    def get_path_nodes(self, path_node_list: List) -> List[int]:
        """Convert path from original topology to sequence of nodes in transformed topology"""
        path_edges = list(zip(path_node_list[:-1], path_node_list[1:]))
        return [self.edge_to_node_map[edge] for edge in path_edges if edge in self.edge_to_node_map]

class SpectrumAvailability:
    def __init__(self, num_spectrum_resources: int):
        self.num_spectrum_resources = num_spectrum_resources

    def get_metrics(self, path: Path, topology: nx.Graph, required_slots: int) -> dict:
        """Calculate spectrum availability metrics for a path"""
        # Get common available slots across path
        available_slots = self._get_path_slots(path, topology)
        
        # Find contiguous blocks
        blocks = self._get_contiguous_blocks(available_slots)
        
        # Calculate metrics
        nfs = np.sum(available_slots)  # Total number of free slots
        nfsb = len(blocks)  # Number of free slot blocks
        
        # Count valid blocks (satisfying required_slots)
        valid_blocks = [block for block in blocks if block[1] >= required_slots]
        nfsb_valid = len(valid_blocks)
        
        # Get first-fit block properties
        if valid_blocks:
            istart = valid_blocks[0][0]  # Starting index of first valid block
            sfirst = valid_blocks[0][1]  # Size of first valid block
        else:
            istart = -1
            sfirst = -1
            
        # Calculate average block size
        sfsb = np.mean([block[1] for block in blocks]) if blocks else 0
        
        return {
            'nfs': nfs,
            'nfsb': nfsb,
            'nfsb_valid': nfsb_valid,
            'istart': istart,
            'sfirst': sfirst,
            'sfsb': sfsb
        }
    
    def _get_path_slots(self, path: Path, topology: nx.Graph) -> np.ndarray:
        """Get common available slots across all links in path"""
        common_slots = np.ones(self.num_spectrum_resources, dtype=bool)
        path_edges = list(zip(path.node_list[:-1], path.node_list[1:]))
        
        for i, j in path_edges:
            edge_idx = topology[i][j]['index']
            edge_slots = topology.graph['available_slots'][edge_idx]
            common_slots &= edge_slots
            
        return common_slots
    
    def _get_contiguous_blocks(self, slots: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous blocks of available spectrum"""
        blocks = []
        block_start = None
        
        for i in range(len(slots)):
            if slots[i]:  # Slot is available
                if block_start is None:
                    block_start = i
            elif block_start is not None:  # End of block
                block_size = i - block_start
                blocks.append((block_start, block_size))
                block_start = None
                
        # Handle last block if it extends to the end
        if block_start is not None:
            block_size = len(slots) - block_start
            blocks.append((block_start, block_size))
            
        return blocks