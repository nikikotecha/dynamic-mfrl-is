import numpy as np 
import torch

file = '/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_50agents_k=300/saved/000018399.tar'

def load_data(file):
    """
    Load saved data and update the networks.

    Parameters
    ----------
    path: str
    name: str
    networks: networks_ssd.Networks
    args: argparse.Namespace
    """
    checkpoint = torch.load(file, map_location="cpu")  # Load onto CPU by default

    return checkpoint['outcomes'], checkpoint['episode_trained']


outcomes, episode_trained = load_data('/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_50agents_k=300/saved/000018399.tar')

print(episode_trained)
print(outcomes['collective_rewards'])
print(outcomes['collective_rewards'][0][episode_trained-30:episode_trained-1])

"""import pickle

# Load the buffer

def dump_buffer(b_file):
    with open(b_file, 'rb') as f:
        buffer = pickle.load(f)
    return buffer

"""