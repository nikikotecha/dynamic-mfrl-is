import random
import sys
import time
import torch

import numpy as np

from networks import Networks
from parsed_args_ssd import args
#from sequential_social_dilemma_games.social_dilemmas.envs.cleanup import CleanupEnvMultiType, CleanupEnvReward
#from sequential_social_dilemma_games.social_dilemmas.envs.harvest import HarvestEnvReward
from env3rundivproduct import MultiAgentInvManagementDiv
import utils_all
import utils_ssd

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def roll_out(env, networks, agent_ids, agent_types, is_train, beta, GREEDY_BETA, image_path, prev_steps, epi_num, is_draw):
    collective_reward = {agent_type: 0 for agent_type in agent_types}
    collective_feature = {agent_type: 0 for agent_type in agent_types}
    samples = {}

    obs = env.reset()
    obs = {agent_id: torch.tensor(obs[agent_id]).to(device) for agent_id in agent_ids}

    for i in range(env.max_steps):
        prev_m_act = {agent_id: torch.tensor(env.prev_m_act[agent_id]).to(device) for agent_id in agent_ids}

        if is_train:
            act, act_probs = networks.get_actions(obs, prev_m_act, beta, is_target=False)
        else:
            act, act_probs = networks.get_actions(obs, prev_m_act, GREEDY_BETA, is_target=False)

        act_probs = {agent_id: torch.tensor(np.log(pdf_value)).to(device) for agent_id, pdf_value in act_probs.items()}

        if is_draw:
            if i == 0:
                print("Run the episode with saving figures...")
            filename = image_path + "frame" + str(prev_steps + i).zfill(9) + ".png"
            env.render(filename=filename, i=prev_steps + i, epi_num=epi_num, act_probs=act_probs)

        obs, act, rew, m_act, n_obs, fea = env.step(act)

        obs = {agent_id: torch.tensor(obs[agent_id]).to(device) for agent_id in agent_ids}
        n_obs = {agent_id: torch.tensor(n_obs[agent_id]).to(device) for agent_id in agent_ids}
        rew = {agent_id: torch.tensor(rew[agent_id]).to(device) for agent_id in agent_ids}
        m_act = {agent_id: torch.tensor(m_act[agent_id]).to(device) for agent_id in agent_ids}
        fea = {agent_id: torch.tensor(fea[agent_id]).to(device) for agent_id in agent_ids}

        if is_train:
            samples[i] = (obs, act, act_probs, rew, m_act, n_obs, fea, beta)
            print("act_probs: ", act_probs)

        for idx, agent_id in enumerate(agent_ids):
            agent_type = agent_types[idx]
            collective_reward[agent_type] += rew[agent_id].item()
            collective_feature[agent_type] += fea[agent_id].item()

    return samples, collective_reward, collective_feature

def main():
    image_path, video_path, saved_path = paths

    epi_length = args.episode_length
    fps = args.fps
    num_types = env.num_types

    agent_ids = env.node_names
    agent_types = list(env.agent_types.values())
    prev_steps = 0
    samples = [None] * epi_length
    collective_reward = [0 for _ in range(num_types)]
    collective_feature = [np.zeros(np.prod(env.observation_space.shape)) for _ in range(num_types)]

    obs = init_set['obs']
    prev_m_act = init_set['m_act']

    for i in range(epi_length):
        epsilon = explore_params['epsilon']
        beta = explore_params['beta']
        if args.mode_ac:
            rand_prob = np.random.rand(1)[0]

        # Add your existing code logic here

if __name__ == "__main__":
    main()