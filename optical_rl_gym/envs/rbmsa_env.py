'''
RBMLSAEnv extends the functionalities of EON Environment implementing basic properties of RBMLSA problem
'''

import gym
import copy
import math
import heapq
import logging
import functools
import numpy as np
import random
import csv
import pprint
from dataclasses import dataclass, field
from math import exp, pi, atan, asinh
from typing import Optional, Sequence, Tuple



import logging


# logging.basicConfig(
#     level=logging.DEBUG,  # Set logging level to DEBUG
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log message format
#     datefmt="%Y-%m-%d %H:%M:%S",  # Date format
# )


from optical_rl_gym.utils import Service, Path, PhysicalParameters
from .optical_network_env import OpticalNetworkEnv
from optical_rl_gym.osnr_calculator import OSNRCalculator

class RBMSAEnv(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'service_blocking_rate_since_reset', 
                    'bit_rate_blocking_rate', 'bit_rate_blocking_rate_since_reset', 'external_fragmentation']
    }

    def __init__(self, scenario=None, topology=None,
                 episode_length=100,
                 load=100,
                 mean_service_holding_time=10.0,
                 node_request_probabilities=None,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False,
                 reset=True
                 ):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed, allow_rejection=allow_rejection,
                         k_paths=k_paths)

        self.scenario = scenario
        #print("scenario", scenario)

        self.physical_params = PhysicalParameters() # for using PhysicalParameters data class
        # Initialize OSNR calculator
        self.osnr_calculator = OSNRCalculator()


        #xd950
        # Frequency ranges for C and L bands (in THz)
        self.band_frequencies = {
            0: {  # C-band
                'start': 191.69,  # THz
                'end': 196.08,    # THz
            },
            1: {  # L-band
                'start': 184.64,  # THz
                'end': 191.69,    # THz
            }
        }

        # OSNR thresholds for each modulation format
        self.osnr_thresholds = {
            'BPSK': 9,   # dB
            'QPSK': 12,   # dB
            '8QAM': 16,  # dB
            '16QAM': 18.6, # dB
        }
        # Frequency slot width (in GHz)
        self.slot_width = 12.5  # GHz

        # logging.basicConfig(
        #     filename="debug.log",  # Log file name
        #     level=logging.DEBUG,
        #     format="%(asctime)s - %(levelname)s - %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S"
        # )


        # specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.bit_rate_requested_since_reset = 0
        self.bit_rate_provisioned_since_reset = 0


        

        # self.episode_bit_rate_requested = 0
        # self.episode_bit_rate_provisioned = 0

        # self.bit_rate_requested = 0
        # self.bit_rate_provisioned = 0
        # self.episode_bit_rate_requested = 0
        # self.episode_bit_rate_provisioned = 0

        

        # depending on the scenario configured will be the total of spectrum resources available (the more bands considered the more available resources)
        # these values are based on the physical layer model presented in https://arxiv.org/pdf/2011.03671.pdf
        total_spectrum_resources = [358, 916, 1584, 2720]
        if self.scenario == 1:
            self.num_spectrum_resources = total_spectrum_resources[0]
        elif self.scenario == 2:
            self.num_spectrum_resources = total_spectrum_resources[1]
        elif self.scenario == 3:
            self.num_spectrum_resources = total_spectrum_resources[2]
        elif self.scenario == 4:
            self.num_spectrum_resources = total_spectrum_resources[3]

        # matrix to store the spectrum allocation
        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges() * self.scenario, self.num_spectrum_resources),
                                                 fill_value=-1, dtype=int)

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros((self.k_paths + 1, 
                                        self.scenario + 1,
                                       self.num_spectrum_resources + 1),
                                       dtype=int)
        self.actions_output_since_reset = np.zeros((self.k_paths + 1, 
                                                    self.scenario + 1,
                                                   self.num_spectrum_resources + 1),
                                                   dtype=int)
        self.actions_taken = np.zeros((self.k_paths + 1, 
                                       self.scenario + 1,
                                      self.num_spectrum_resources + 1),
                                      dtype=int)
        self.actions_taken_since_reset = np.zeros((self.k_paths + 1,
                                                   self.scenario + 1,
                                                  self.num_spectrum_resources + 1),
                                                  dtype=int)
        self.action_space = gym.spaces.MultiDiscrete((self.k_paths + self.reject_action,
                                                      self.scenario + self.reject_action,
                                                     self.num_spectrum_resources + self.reject_action))
        # defining observation and action spaces
        self.observation_space = gym.spaces.Dict(
            {'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)}
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rbmsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action: [int]):
        self.logger.debug("-----------------------------------------------------------------------------------------")
        # agent decides path, band and initial indices of the selected block
        path, band, initial_slot = action[0], action[1], action[2]  # action is for assigning a path
        self.actions_output[path, band, initial_slot] += 1
        if path < self.k_paths and band < self.scenario and initial_slot < self.num_spectrum_resources:
            temp_path = self.k_shortest_paths[self.service.source, self.service.destination][path]
            #print("Temp path", temp_path)
            if temp_path.length <= 4000:
                # if there are enough resources for the agent to select, then compute the number of slots that the request needs
                slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                            self.scenario, band, self.modulations_c1, self.modulations_c2, self.modulations_l2,
                                            self.modulations_c3, self.modulations_l3, self.modulations_s3, self.modulations_c4, self.modulations_l4,
                                            self.modulations_s4, self.modulations_e4)
                self.logger.debug(
                    '{} processing action {} path {} and initial slot {} for {} slots'.format(self.service.service_id,
                                                                                            action, path, initial_slot,
                                                                                            slots))
                # checking if the path selected is really free
                if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                    initial_slot, slots, band):
                    
                    #check for OSNR
                    temp_service = copy.deepcopy(self.service)
                    temp_service.bandwidth = slots * self.slot_width # in GHz
                    temp_service.band = band
                    temp_service.initial_slot = initial_slot
                    temp_service.number_slots = slots
                    temp_service.path = self.k_shortest_paths[self.service.source, self.service.destination][path]

                    temp_service.center_frequency = self._calculate_center_frequency(temp_service)

                    temp_service.modulation_format = self.get_modulation_format(temp_path, self.scenario, band, self.modulations_c1, self.modulations_c2, 
                                                self.modulations_l2, self.modulations_c3, self.modulations_l3, self.modulations_s3,
                                                 self.modulations_c4, self.modulations_l4, self.modulations_s4, self.modulations_e4)['modulation']
                    
                    #print("Temp serive:", temp_service)
                    osnr_db = self.osnr_calculator.calculate_osnr(temp_service, self.topology)
                    if osnr_db >= self.osnr_thresholds[temp_service.modulation_format]:
                        #print("OSNR", osnr)
                        self.service.current_OSNR = osnr_db
                        self.service.OSNR_th = self.osnr_thresholds[temp_service.modulation_format]       
                        # if so, provision it (write zeros the position os the selected block in the available slots matrix
                        self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                            initial_slot, slots, band, self.service.arrival_time)
                        self.service.accepted = True  # the request was accepted
                        self.actions_taken[path, band, initial_slot] += 1
                        self._add_release(self.service)
                else:
                    self.service.accepted = False  # the request was rejected (blocked), the path is not free
        else:
            self.service.accepted = False # the request was rejected (blocked), the path is not free

        if not self.service.accepted:
            self.actions_taken[self.k_paths, self.scenario, self.num_spectrum_resources] += 1
            self.logger.debug(f"Service Rejected: {self.service}")

        self.services_processed += 1
        self.services_processed_since_reset += 1
        self.bit_rate_requested += self.service.bit_rate
        self.bit_rate_requested_since_reset += self.service.bit_rate
        #self.bit_rate_provisioned += self.service.bit_rate


        self.topology.graph['services'].append(self.service)

        k_paths = self.k_shortest_paths[self.service.source, self.service.destination];
        path_selected = k_paths[path] if path < self.k_paths else None
        reward = self.reward(band, path_selected) # computing the reward

        # metrics to be monitor
        info = {    # computing metrics to be monitored
                   #'band': band if self.service.accepted else -1,
                   'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
                   'service_blocking_rate_since_reset': (self.services_processed_since_reset - self.services_accepted_since_reset) / self.services_processed_since_reset,
                   'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
                   'bit_rate_blocking_rate_since_reset': (self.bit_rate_requested_since_reset - self.bit_rate_provisioned_since_reset) / self.bit_rate_requested_since_reset,

               
               }

        self._new_service = False
        self._next_service()
        return self.observation(), reward, self.services_processed_since_reset == self.episode_length, info

    def reward(self, band, path_selected):
        return super().reward()

    def reset(self, only_counters=True):
        self.bit_rate_requested_since_reset = 0
        self.bit_rate_provisioned_since_reset = 0
        self.services_processed_since_reset = 0
        self.services_accepted_since_reset = 0



        self.actions_output_since_reset = np.zeros((self.k_paths + self.reject_action,
                                                    self.scenario + self.reject_action,
                                                   self.num_spectrum_resources + self.reject_action),
                                                   dtype=int)
        self.actions_taken_since_reset = np.zeros((self.k_paths + self.reject_action,
                                                   self.scenario + self.reject_action,
                                                  self.num_spectrum_resources + self.reject_action),
                                                  dtype=int)

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        # defining the available_slots matrix
        self.topology.graph["available_slots"] = np.ones((self.topology.number_of_edges() * self.scenario, self.num_spectrum_resources), dtype=int)

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges() * self.scenario, self.num_spectrum_resources),
                                                 fill_value=-1, dtype=int)

        for idx, lnk in enumerate(self.topology.edges()):
                    self.topology[lnk[0]][lnk[1]]['external_fragmentation'] = 0.
        self.topology.graph["compactness"] = 0.
        self.topology.graph["throughput"] = 0.
        for idx, lnk in enumerate(self.topology.edges()):
            self.topology[lnk[0]][lnk[1]]['fragmentation'] = 0.
            self.topology[lnk[0]][lnk[1]]['compactness'] = 0.

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human'):
        return

    def _provision_path(self, path: Path, initial_slot, number_slots, band, at ):
        # usage
        if not self.is_path_free(path, initial_slot, number_slots, band):
            self.logger.debug(f"Service rejcetd: {self.service.service_id} reason: path mot free")
            raise ValueError("Path {} has not enough capacity on slots {}-{}".format(path.node_list, path, initial_slot,
                                                                                     initial_slot + number_slots))
            

        #self.logger.debug('{} assigning path {} on initial slot {} for {} slots'.format(self.service.service_id, path.node_list, initial_slot, number_slots))

        # computing the horizontal shift in the available slots matrix
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        # for i in range(len(path.node_list) - 1):
        #     # provisioning resources for the given path (write zeros)
        #     self.topology.graph['available_slots'][((self.topology[path.node_list[i]][path.node_list[i + 1]]['index']) + (self.topology.number_of_edges() * band)),
        #                                                                     initial_slot_shift:initial_slot_shift + number_slots] = 0
        #     self.spectrum_slots_allocation[((self.topology[path.node_list[i]][path.node_list[i + 1]]['index']) + (self.topology.number_of_edges() * band)),
        #                                                 initial_slot_shift:initial_slot_shift + number_slots] = self.service.service_id
        #     self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service)
        #     self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service)
        #     self._update_link_stats(path.node_list[i], path.node_list[i + 1])

        # Modified to handle directed edges
        for i in range(len(path.node_list) - 1):
            edge_index = self.topology[path.node_list[i]][path.node_list[i + 1]]['index']
            
            self.topology.graph['available_slots'][
                (edge_index + (self.topology.number_of_edges() * band)),
                initial_slot_shift:initial_slot_shift + number_slots] = 0
                
            self.spectrum_slots_allocation[
                (edge_index + (self.topology.number_of_edges() * band)),
                initial_slot_shift:initial_slot_shift + number_slots] = self.service.service_id
                
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])
        
        
        self.topology.graph['running_services'].append(self.service)
        self.service.route = path
        self.service.scenario = band
        self.service.band = band
        self.service.initial_slot = initial_slot_shift
        self.service.number_slots = number_slots
        self.service.bandwidth = number_slots * self.slot_width
        #self.service.modulation_format = 

        self.service.termination_time = self.current_time + at
        self.service.center_frequency = self._calculate_center_frequency(self.service) #Error with the band
        self.service.accepted = True  # the request was accepted
        self.logger.debug(f"Service Accepted: {self.service}")

        #print(self.service)
        
        self._update_network_stats()

        self.services_accepted += 1
        self.services_accepted_since_reset += 1
        self.bit_rate_provisioned += self.service.bit_rate
        self.bit_rate_provisioned_since_reset += self.service.bit_rate

        #self.services_accepted += 1
        #self.episode_services_accepted += 1
        #self.bit_rate_provisioned += self.service.bit_rate
        #self.episode_bit_rate_provisioned += self.service.bit_rate

    
    def _calculate_center_frequency(self, service: Service):
        """
        Calculate center frequency for an allocation
        Args:
            Service
        Returns:
            Center frequency in THz
        """
        # Get band frequency range
        band_start = self.band_frequencies[service.band]['start']
        
        # Calculate the starting frequency of the block
        # Each slot is 12.5 GHz = 0.0125 THz
        start_freq = band_start + (service.initial_slot * 0.0125)
        
        # Calculate center frequency by adding half of the block width
        center_freq = start_freq + ((service.number_slots * 0.0125) / 2)
        
        return center_freq


    def _release_path(self, service: Service):
        '''
        This method release a path when a service complets by writing ones in the available slots matrix.
        '''
        # for i in range(len(service.route.node_list) - 1):
        #     self.topology.graph['available_slots'][
        #         ((self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index']) + (self.topology.number_of_edges() * service.scenario)),
        #                                 service.initial_slot:service.initial_slot + service.number_slots] = 1
        #     self.spectrum_slots_allocation[((self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index']) + (self.topology.number_of_edges() * service.scenario)),
        #                                 service.initial_slot:service.initial_slot + service.number_slots] = -1
        #     self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service)
        #     self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        """Modified release path for directed edges"""
        for i in range(len(service.route.node_list) - 1):
            edge_index = self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index']
            
            self.topology.graph['available_slots'][
                (edge_index + (self.topology.number_of_edges() * service.scenario)),
                service.initial_slot:service.initial_slot + service.number_slots] = 1
                
            self.spectrum_slots_allocation[
                (edge_index + (self.topology.number_of_edges() * service.scenario)),
                service.initial_slot:service.initial_slot + service.number_slots] = -1
            
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])

        self.topology.graph['running_services'].remove(service)

    def _update_network_stats(self):
        last_update = self.topology.graph['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph['throughput']
            last_compactness = self.topology.graph['compactness']

            cur_throughtput = 0.
            sum_slots_paths = 0 # this accounts for the sum of all Bi * Hi

            for service in self.topology.graph["running_services"]:
                cur_throughtput += service.bit_rate
                sum_slots_paths += service.number_slots * service.route.hops

            throughput = ((last_throughput * last_update) + (cur_throughtput * time_diff)) / self.current_time
            self.topology.graph['throughput'] = throughput

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6476152

            # TODO: implement fragmentation
            sum_occupied = 0
            sum_unused_spectrum_blocks = 0
            for n1, n2 in self.topology.edges():
                # link = self.topology.graph['available_slots'][self.topology[n1][n2]['index'],:]
                # getting the blocks
                initial_indices, values, lengths = RBMSAEnv.rle(self.topology.graph['available_slots'][self.topology[n1][n2]['index'], :])
                used_blocks = [i for i, x in enumerate(values) if x == 0]
                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                    sum_occupied += lambda_max - lambda_min # we do not put the "+1" because we use zero-indexed arrays

                    # evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RBMSAEnv.rle(
                        self.topology.graph['available_slots'][self.topology[n1][n2]['index'], lambda_min:lambda_max])
                    sum_unused_spectrum_blocks += np.sum(1 - internal_values)

            if sum_unused_spectrum_blocks > 0:
                cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (self.topology.number_of_edges() / sum_unused_spectrum_blocks)
                compactness = ((last_compactness * last_update) + (cur_spectrum_compactness * time_diff)) / self.current_time
                self.topology.graph['compactness'] = compactness

        self.topology.graph['last_update'] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']
        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (self.num_spectrum_resources - np.sum(
                self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :])) / self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

            slot_allocation = self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2]['external_fragmentation']

            cur_external_fragmentation = 0.
            cur_link_compactness = 0.
            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RBMSAEnv.rle(slot_allocation)

                # computing external fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1. - (float(max_empty) / float(np.sum(slot_allocation)))

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_fragmentation = self.topology[node1][node2]['fragmentation']
            last_compactness = self.topology[node1][node2]['compactness']

            cur_fragmentation = 0.
            cur_compactness = 0.
            if np.sum(slot_allocation) > 0:
                blocks = np.split(slot_allocation, np.where(np.diff(slot_allocation) != 0)[0] + 1)
                max_empty = 0
                for block in blocks:
                    if np.all(block == 1):
                        max_empty = max(max_empty, len(block))
                cur_fragmentation = 1. - (float(max_empty) / float(np.sum(slot_allocation)))

                lambdas = np.where(slot_allocation == 0)
                if len(lambdas) > 1:
                    lambda_min = np.min(lambdas)
                    lambda_max = np.max(lambdas)
                    # alloc = slot_allocation[lambda_min:lambda_max]
                    blocks = np.split(slot_allocation[lambda_min:lambda_max],
                                      np.where(np.diff(slot_allocation[lambda_min:lambda_max]) != 0)[0] + 1)
                    k = 0
                    for block in blocks:
                        if np.all(block == 1):
                            k += 1
                    # number of blocks of free slots between first and last slot used
                    if k > 0:
                        cur_compactness = ((lambda_max - lambda_min + 1) / len(lambdas)) * (1 / k)
                    else:
                        cur_compactness = 1.
                else:
                    cur_compactness = 1.

            external_fragmentation = ((last_external_fragmentation * last_update) + (cur_external_fragmentation * time_diff)) / self.current_time
            self.topology[node1][node2]['external_fragmentation'] = external_fragmentation
            
            fragmentation = ((last_fragmentation * last_update) + (cur_fragmentation * time_diff)) / self.current_time
            self.topology[node1][node2]['fragmentation'] = fragmentation

            link_compactness = ((last_compactness * last_update) + (cur_compactness * time_diff)) / self.current_time
            self.topology[node1][node2]['compactness'] = link_compactness

        self.topology[node1][node2]['last_update'] = self.current_time

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # list of possible bit-rates for the request
        #BitRate = [100, 200, 400]
        BitRate = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        bit_rate = random.choice(BitRate)

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

        self.service = Service(self.services_processed_since_reset, src, src_id,
                               destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, bit_rate=bit_rate)
        self._new_service = True

    def _get_path_slot_id(self, action: int) -> (int, int):
        """
        Decodes the single action index into the path index and the slot index to be used.
        :param action: the single action index
        :return: path index and initial slot index encoded in the action
        """
        path = int(action / self.num_spectrum_resources)
        initial_slot = action % self.num_spectrum_resources
        return path, initial_slot

    def get_number_slots(self, path: Path, scenario, band, modulations_c1, modulations_c2, modulations_l2,
                              modulations_c3, modulations_l3, modulations_s3, modulations_c4, modulations_l4,
                              modulations_s4, modulations_e4) -> int: # Calculo del numero de FSUs
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband. First, it assigns best modulation format depending of the
        distance and then does the computation of necessary slots.
        """

        # assigning a modulation format for the request
        modulation = self.get_modulation_format(path, scenario, band, modulations_c1, modulations_c2, modulations_l2,
                                                modulations_c3, modulations_l3, modulations_s3, modulations_c4,
                                                modulations_l4, modulations_s4, modulations_e4)
        # computing the number of slots
        service_bit_rate = self.service.bit_rate
        
        number_of_slots = math.ceil(service_bit_rate / modulation['capacity']) + 1
        return number_of_slots

    def get_shift(self, band):
        '''
        Method that computes a shift (depending on the band associated to the request) in the the available_slots matrix.
        This gurantees that zeros/ones can be written in the correct positions.
        '''
        x = 0  # Starting index to shift in columns in the available_slots matrix
        y = 0  # Column shift in the available slots matrix
        if band == 0:
            x = 0
            y = 357
        elif band == 1:
            x = 357
            y = 916
        elif band == 2:
            x = 824
            y = 1584
        elif band == 3:
            x = 1584
            y = 2721
        return x, y

    # def is_path_free(self, path: Path, initial_slot: int, number_slots: int, band) -> bool:
    #     # shifting the initial slots in the columns of the available slots matrix
    #     x = self.get_shift(band)[0]
    #     initial_slot_shift = initial_slot + x
    #     if initial_slot + number_slots > self.num_spectrum_resources:
    #         # logging.debug('error index' + env.parameters.rsa_algorithm)
    #         return False
    #     # (self.topology.number_of_edges() * band) will give a shift by rows in the available_slots matrix
    #     for i in range(len(path.node_list) - 1):
    #         if np.any(self.topology.graph['available_slots'][
    #                   ((self.topology[path.node_list[i]][path.node_list[i + 1]]['index']) + (self.topology.number_of_edges() * band)),
    #                   initial_slot_shift:initial_slot_shift + number_slots] == 0):
    #             return False
    #     return True

    def is_path_free(self, path: Path, initial_slot: int, number_slots: int, band) -> bool:
        """Modified to check spectrum availability in directed edges"""
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        
        if initial_slot + number_slots > self.num_spectrum_resources:
            return False
            
        # Check only forward direction
        for i in range(len(path.node_list) - 1):
            edge_index = self.topology[path.node_list[i]][path.node_list[i + 1]]['index']
            if np.any(self.topology.graph['available_slots'][
                (edge_index + (self.topology.number_of_edges() * band)),
                initial_slot_shift:initial_slot_shift + number_slots] == 0):
                return False
        return True

    def get_available_slots(self, path: Path, band):
        # This ensures that spectrum resources available for the assigned band can be read (shift by columns in the available slot matrix)
        x = self.get_shift(band)[0]
        y = self.get_shift(band)[1]
        # (self.topology.number_of_edges() * band) will give a shift by rows in the available_slots matrix
        available_slots = functools.reduce(np.multiply,
            self.topology.graph["available_slots"][[((self.topology[path.node_list[i]][path.node_list[i + 1]]['id']) + (self.topology.number_of_edges() * band))
                                                    for i in range(len(path.node_list) - 1)], x:y])
        return available_slots

    def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path, scenario, band, modulations_c1, modulations_c2, modulations_l2,
                             modulations_c3, modulations_l3, modulations_s3, modulations_c4, modulations_l4,
                             modulations_s4, modulations_e4):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[self.service.source, self.service.destination][path], band)

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                      scenario, band, modulations_c1, modulations_c2, modulations_l2,
                                      modulations_c3, modulations_l3, modulations_s3, modulations_c4, modulations_l4,
                                      modulations_s4, modulations_e4)

        # getting the blocks
        initial_indices, values, lengths = RBMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        # finds all possible consecutives 1s in the spectrum
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        # Finds all possible blocks (of 0s or 1s) with sufficient slots to accomodate the request
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        # interesection of available indices and sufficient indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[:self.j]

        return initial_indices[final_indices], lengths[final_indices]

    '''
    Physical Layer Model. 
    # The given values corresponds to a BER of 4.7*10^-3 as indicated in https://arxiv.org/pdf/2011.03671.pdf
    '''

    # [BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM, 256QAM]
    #capacity = [23, 46, 69, 92, 115, 140, 186] # bit-rate achieved by a request in single slot for different modulation formats
    capacity = [12.5, 25, 37.5, 50, 62.5, 75, 87.5] # bit-rate achieved by a request in single slot for different modulation formats

    # Scenario 1: Band C
    modulations_c1 = list()

    modulations_c1.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 4000})
    modulations_c1.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 2000})
    modulations_c1.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 1000})
    modulations_c1.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 500})
    # modulations_c1.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 1300})
    # modulations_c1.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 700})
    # modulations_c1.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

    # Scenario 2: Band C+L # modified by xd950
    modulations_c2 = list()  # Band C

    modulations_c2.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 4000})
    modulations_c2.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 2000})
    modulations_c2.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 1000})
    modulations_c2.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 500})
    # modulations_c2.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 1300})
    # modulations_c2.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 700})
    # modulations_c2.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

#     modulations_osnr_C_L = [
#     {'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 4000, 'osnr_threshold': self.osnr_thresholds['BPSK']},
#     {'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 2000, 'osnr_threshold': self.osnr_thresholds['QPSK']},
#     {'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 1000, 'osnr_threshold': self.osnr_thresholds['8QAM']},
#     {'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 500, 'osnr_threshold': self.osnr_thresholds['16QAM']}
# ]

    modulations_l2 = list()  # Band L # modified by xd950

    modulations_l2.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 4000})
    modulations_l2.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 2000})
    modulations_l2.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 1000})
    modulations_l2.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 500})
    # modulations_l2.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 1100})
    # modulations_l2.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 600})
    # modulations_l2.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

    # Scenario 3: Band C+L+S
    modulations_c3 = list()  # Band C

    modulations_c3.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 17400})
    modulations_c3.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 8700})
    modulations_c3.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 4700})
    modulations_c3.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 2300})
    modulations_c3.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 1200})
    modulations_c3.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 600})
    modulations_c3.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

    modulations_l3 = list()  # Band L

    modulations_l3.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 16700})
    modulations_l3.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 8400})
    modulations_l3.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 4600})
    modulations_l3.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 2200})
    modulations_l3.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 1100})
    modulations_l3.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 600})
    modulations_l3.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

    modulations_s3 = list()  # Band S

    modulations_s3.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 14800})
    modulations_s3.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 7400})
    modulations_s3.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 4100})
    modulations_s3.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 2000})
    modulations_s3.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 1000})
    modulations_s3.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 500})
    modulations_s3.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

    # Scenario 4: Band C+L+S+E
    modulations_c4 = list()  # Band C

    modulations_c4.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 13000})
    modulations_c4.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 6500})
    modulations_c4.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 3500})
    modulations_c4.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 1700})
    modulations_c4.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 800})
    modulations_c4.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 400})
    modulations_c4.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

    modulations_l4 = list()  # Band L

    modulations_l4.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 14400})
    modulations_l4.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 7200})
    modulations_l4.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 3900})
    modulations_l4.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 1900})
    modulations_l4.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 900})
    modulations_l4.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 500})
    modulations_l4.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 100})

    modulations_s4 = list()  # Band S

    modulations_s4.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 10200})
    modulations_s4.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 5100})
    modulations_s4.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 2900})
    modulations_s4.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 1400})
    modulations_s4.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 700})
    modulations_s4.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 300})
    modulations_s4.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 0})

    modulations_e4 = list()  # Band E

    modulations_e4.append({'modulation': 'BPSK', 'capacity': capacity[0], 'maximum_length': 3100})
    modulations_e4.append({'modulation': 'QPSK', 'capacity': capacity[1], 'maximum_length': 1500})
    modulations_e4.append({'modulation': '8QAM', 'capacity': capacity[2], 'maximum_length': 900})
    modulations_e4.append({'modulation': '16QAM', 'capacity': capacity[3], 'maximum_length': 400})
    modulations_e4.append({'modulation': '32QAM', 'capacity': capacity[4], 'maximum_length': 200})
    modulations_e4.append({'modulation': '64QAM', 'capacity': capacity[5], 'maximum_length': 100})
    modulations_e4.append({'modulation': '256QAM', 'capacity': capacity[6], 'maximum_length': 0})

    '''
      Methods for assigning a modulation format given the information offered by the pyshical layer model.
      '''
    def calculation_ml(self, modulations, length):
        for i in range(len(modulations) - 1):
            if length > modulations[i + 1]['maximum_length']:
                if length <= modulations[i]['maximum_length']:
                    return modulations[i]
        return modulations[len(modulations) - 1]

    def get_modulation_format(self, path: Path, scenario, band, modulations_c1, modulations_c2, modulations_l2,
                              modulations_c3, modulations_l3, modulations_s3, modulations_c4, modulations_l4,
                              modulations_s4, modulations_e4):

        length = path.length # gets the distance of the request
        if scenario == 1:  # Scenario C
            modulation_format = self.calculation_ml(modulations_c1, length)
        elif scenario == 2:  # Scenario C+L
            if band == 0:  # Band C
                modulation_format = self.calculation_ml(modulations_c2, length)
            elif band == 1:  # Band L
                modulation_format = self.calculation_ml(modulations_l2, length)
        elif scenario == 3:  # Scenario C+L+S
            if band == 0:  # Band C
                modulation_format = self.calculation_ml(modulations_c3, length)
            elif band == 1:  # Band L
                modulation_format = self.calculation_ml(modulations_l3, length)
            elif band == 2:  # Band S
                modulation_format = self.calculation_ml(modulations_s3, length)
        elif scenario == 4:  # Scenario C+L+S+E
            if band == 0:  # Band C
                modulation_format = self.calculation_ml(modulations_c4, length)
            elif band == 1:  # Band L
                modulation_format = self.calculation_ml(modulations_l4, length)
            elif band == 2:  # Band S
                modulation_format = self.calculation_ml(modulations_s4, length)
            elif band == 3:  # Band E
                modulation_format = self.calculation_ml(modulations_e4, length)

        return modulation_format
    


def shortest_available_path_first_fit(env: RBMSAEnv) -> Tuple[int, int]:
    print("called")
    for idp, path in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(
            0, env.topology.graph["num_spectrum_resources"] - num_slots
        ):
            if env.is_path_free(path, initial_slot, num_slots):
                return (idp, initial_slot)
    return (env.topology.graph["k_paths"], env.topology.graph["num_spectrum_resources"])




