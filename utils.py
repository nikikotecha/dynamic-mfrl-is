"""
Helper functions for getting agents and rl configs
"""

import os
import numpy as np
import copy
import torch.nn as nn 
import torch

def init_weights(m):
    """
    Define the initialization function for the layers.

    Parameters
    ----------
    m
        Type of the layer.
    """
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_connections(connections):
    for node, children in connections.items():
        if children:
            for child in children:
                if child < node:
                    raise Exception("Downstream node cannot have a smaller index number than upstream node")

def create_network(connections):
    num_nodes = max(connections.keys())
    network = np.zeros((num_nodes + 1, num_nodes + 1))
    for parent, children in connections.items():
        if children:
            for child in children:
                network[parent][child] = 1

    return network

def create_adjacency_matrix(connec, num_nodes, num_products):
    # Initialize the adjacency matrix with zeros
    matrix_size = num_nodes * num_products
    adj_matrix = np.zeros((matrix_size, matrix_size))
    
    # Iterate through the connections dictionary
    for src_node, dest_nodes in connec.items():
        for dest_node in dest_nodes:
            # Each node has num_products products, so we need to add connections for each product
            for product in range(num_products):
                src_index = src_node * num_products + product
                dest_index = dest_node * num_products + product
                adj_matrix[src_index, dest_index] = 1
    
    return adj_matrix

def find_connections(agent_handle, adjacency_matrix, agent_handles):
    # Get the index for the given agent handle
    if agent_handle not in agent_handles:
        raise ValueError("Agent handle not found.")

    agent_index = agent_handles.index(agent_handle)

    # Get the connections for this agent
    connections_out = adjacency_matrix[agent_index]
    connections_in = adjacency_matrix[:, agent_index]

    # Find connected agents (not connected to self)
    connected_out_indices = np.where(connections_out > 0)[0]
    connected_in_indices = np.where(connections_in > 0)[0]

    connected_out_agents = [agent_handles[i] for i in connected_out_indices if i != agent_index]
    connected_in_agents = [agent_handles[i] for i in connected_in_indices if i != agent_index]
    one_hop_connections = connected_in_agents + connected_out_agents
    return one_hop_connections

def get_stage(node, network):
    reached_root = False
    stage = 0
    counter = 0
    if node == 0:
        return 0
    while not reached_root:
        for i in range(len(network)):
            if network[i][node] == 1:
                stage += 1
                node = i
                if node == 0:
                    return stage
        counter += 1
        if counter > len(network):
            raise Exception("Infinite Loop")



def get_retailers(network):
    retailers = []
    for i in range(len(network)):
        if not any(network[i]):
            retailers.append(i)

    return retailers
