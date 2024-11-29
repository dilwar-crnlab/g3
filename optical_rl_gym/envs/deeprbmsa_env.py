'''
This environment attributes dynamic operation to RBMLSAEnv
'''

import gym
import numpy as np

from .rbmsa_env import RBMSAEnv
from .optical_network_env import OpticalNetworkEnv


class DeepRBMSAEnv(RBMSAEnv):

    def __init__(self, scenario, topology=None, j=1,
                 episode_length=1000,
                 load=100,
                 mean_service_holding_time=500,
                 node_request_probabilities=None,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False):
        super().__init__(scenario=scenario, topology=topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         k_paths=k_paths,
                         allow_rejection=allow_rejection,
                         reset=False)

        self.j = j
        shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 4) * self.k_paths * self.scenario
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(self.k_paths  * self.scenario * self.j + self.reject_action)
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        self.reset(only_counters=False)




    def step(self, action: int):
        parent_step_result = None

        valid_action = False
        if action < self.k_paths * self.j * self.scenario:  # action is for assigning a path
            valid_action = True
            path, band, block = self._get_path_block_id(action)  # agent decides path, band, block
                
            # getting initial indices and length of the block where the request will be allocated
            initial_indices, lengths = self.get_available_blocks(path, self.scenario, band, self.modulations_c1, self.modulations_c2, self.modulations_l2,
                                                                 self.modulations_c3, self.modulations_l3, self.modulations_s3, self.modulations_c4, self.modulations_l4,
                                                                 self.modulations_s4, self.modulations_e4)

            slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                          self.scenario, band, self.modulations_c1, self.modulations_c2, self.modulations_l2,
                                          self.modulations_c3, self.modulations_l3, self.modulations_s3, self.modulations_c4, self.modulations_l4,
                                          self.modulations_s4, self.modulations_e4)
                                          
            # checks if there are enough blocks to allocate the request
            if block < len(initial_indices):
                parent_step_result = super().step([path, band, initial_indices[block]])
            else:
                parent_step_result = super().step([self.k_paths, self.scenario, self.num_spectrum_resources])
        else:
            parent_step_result = super().step([self.k_paths, self.scenario, self.num_spectrum_resources])

        obs, rw, _, info = parent_step_result

        # add slots info in training logs
        info['slots'] = slots if valid_action else -1
        return parent_step_result

    # def observation(self):
    #     # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSCA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSCA_Agent.py#L384
    #     source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
    #     min_node = min(self.service.source_id, self.service.destination_id)
    #     max_node = max(self.service.source_id, self.service.destination_id)
    #     source_destination_tau[0, min_node] = 1
    #     source_destination_tau[1, max_node] = 1
    #     spectrum_obs = np.full((self.k_paths * self.scenario, 2 * self.j + 3), fill_value=-1.)
    #     # for the k-path ranges all possible bands to take the best decision
    #     for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
    #       for band in range(self.scenario):
    #         available_slots = self.get_available_slots(path, band)
    #         num_slots = self.get_number_slots(path, self.scenario, band, self.modulations_c1, self.modulations_c2, self.modulations_l2,
    #                                       self.modulations_c3, self.modulations_l3, self.modulations_s3, self.modulations_c4, self.modulations_l4,
    #                                       self.modulations_s4, self.modulations_e4)
    #         initial_indices, lengths = self.get_available_blocks(idp, self.scenario, band, self.modulations_c1, self.modulations_c2, self.modulations_l2,
    #                                       self.modulations_c3, self.modulations_l3, self.modulations_s3, self.modulations_c4, self.modulations_l4,
    #                                       self.modulations_s4, self.modulations_e4)
    #         for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
    #                     # initial slot index
    #             spectrum_obs[idp + (self.k_paths * band), idb * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources

    #                     # number of contiguous FS available
    #             spectrum_obs[idp + (self.k_paths * band), idb * 2 + 1] = (length - 8) / 8
    #         spectrum_obs[idp + (self.k_paths * band), self.j * 2] = (num_slots - 5.5) / 3.5 # number of FSs necessary

    #         idx, values, lengths = DeepRBMSAEnv.rle(available_slots)

    #         av_indices = np.argwhere(values == 1) # getting indices which have value 1
    #         # spectrum_obs = matrix with shape k_routes x s_bands in the scenario
    #         spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # total number of available FSs
    #         spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 # avg. number of FS blocks available
    #     bit_rate_obs = np.zeros((1, 1))
    #     bit_rate_obs[0, 0] = self.service.bit_rate / 100

    #     return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
    #                            spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
    #         .reshape(self.observation_space.shape)

    def observation(self):
        # Source-destination encoding
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        source_destination_tau[0, self.service.source_id] = 1
        source_destination_tau[1, self.service.destination_id] = 1
        
        # Extended spectrum observation matrix to include OSNR threshold
        # Add one more column for OSNR threshold
        spectrum_obs = np.full((self.k_paths * 2, 2 * self.j + 4), fill_value=-1.)  # +4 instead of +3
        
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
            for band in range(2):  # C and L bands only
                # Get modulation format based on path length
                # modulation = self.get_modulation_format(path, 2, band,
                #     self.modulations_c2 if band == 0 else self.modulations_l2)
                
                modulation = self.get_modulation_format(path, 2, band, self.modulations_c1, self.modulations_c2, 
                                                self.modulations_l2, self.modulations_c3, self.modulations_l3, self.modulations_s3,
                                                 self.modulations_c4, self.modulations_l4, self.modulations_s4, self.modulations_e4)
                
                # Get OSNR threshold for this modulation
                osnr_threshold = self.osnr_thresholds[modulation['modulation']]
                
                # Normalize OSNR threshold (assuming max threshold is 18.6 dB)
                normalized_osnr = osnr_threshold / 18.6
                
                available_slots = self.get_available_slots(path, band)
                num_slots = self.get_number_slots(path, self.scenario, band, self.modulations_c1, self.modulations_c2, self.modulations_l2, 
                                           self.modulations_c3, self.modulations_l3, self.modulations_s3, self.modulations_c4, self.modulations_l4,
                                           self.modulations_s4, self.modulations_e4)
                
                initial_indices, lengths = self.get_available_blocks(idp, self.scenario, band, self.modulations_c1, self.modulations_c2, self.modulations_l2,
                                           self.modulations_c3, self.modulations_l3, self.modulations_s3, self.modulations_c4, self.modulations_l4,
                                          self.modulations_s4, self.modulations_e4)
                
                for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
                    spectrum_obs[idp + (self.k_paths * band), idb * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources
                    spectrum_obs[idp + (self.k_paths * band), idb * 2 + 1] = (length - 8) / 8
                    
                spectrum_obs[idp + (self.k_paths * band), self.j * 2] = (num_slots - 5.5) / 3.5
                
                idx, values, lengths = DeepRBMSAEnv.rle(available_slots)
                av_indices = np.argwhere(values == 1)
                
                spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources
                spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 if len(av_indices) > 0 else -1
                
                # Add normalized OSNR threshold
                spectrum_obs[idp + (self.k_paths * band), self.j * 2 + 3] = normalized_osnr
        
        # Bit rate observation
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.service.bit_rate / 100
        
        return np.concatenate((bit_rate_obs, 
                            source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                            spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), 
                            axis=1).reshape(self.observation_space.shape)

    def reward(self, band, path_selected):
        return 1 if self.service.accepted else -1

    def reset(self, only_counters=True):
        return super().reset(only_counters=only_counters)

    def _get_path_block_id(self, action: int) -> (int, int):
        # decoding the action
        path = action // (self.j * self.scenario)
        band = action // (self.j * self.k_paths)
        block = action % self.j
        return path, band, block


