import random
import sys
import time

import numpy as np

from networks_backup import Networks
from parsed_args_ssd import args
#from sequential_social_dilemma_games.social_dilemmas.envs.cleanup import CleanupEnvMultiType, CleanupEnvReward
#from sequential_social_dilemma_games.social_dilemmas.envs.harvest import HarvestEnvReward
from env_execute import MultiAgentInvManagementDiv
import utils_all
import utils_ssd
import torch 
import os 
import json 
from collections import defaultdict

GREEDY_BETA = 0.001

# === File Paths: 30 Agents, IS ===
#br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_30_is_true/saved/"
#br_num = "000029999.tar"
#base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_30_2_nn/saved/"
#base_num = "000029999.tar"
#br_index = 8  # Assuming agent 8 is the BR agent

# === File Paths: 30 Agents, MF ===
#br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_30_mf/saved/"
#br_num = "000004999.tar"
#base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_30_2/saved/"
#base_num = "000015399.tar"
#br_index = 8  # Assuming agent 8 is the BR agent

# === File Paths: 50 Agents, IS ===
#br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_50_is_true/saved/"
#br_num = "000021199.tar"
#base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_50_2/saved/"
#base_num = "000014599.tar"
#br_index = 17  # Assuming agent 8 is the BR agent

# === File Paths: 50 Agents, MF ===
#br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_50_mf_true/saved/"
#br_num = "000021599.tar"
#base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_50_2/saved/"
#base_num = "000014999.tar"
#br_index = 17  # Assuming agent 8 is the BR agent

# === File Paths: 70 Agents, IS ===
#br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_70_is_true/saved/"
#br_num = "000013999.tar"
#base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_70agents_restart2/saved/"
#base_num = "000014399.tar"
#br_index = 34  # Assuming agent 8 is the BR agent

# === File Paths: 70 Agents, MF ===
#br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_70_mf_true/saved/"
#br_num = "000013199.tar"
#base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_70_2/saved/"
#base_num = "000007599.tar"
#br_index = 34  # Assuming agent 8 is the BR agent

# === File Paths: 100 Agents, IS ===
#br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_100_is_true/saved/"
#br_num = "000007199.tar"
#base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_100_restart2_nn/saved/"
#base_num = "000003799.tar"
#br_index = 34  # Assuming agent 34 is the BR agent

# === File Paths: 100 Agents, MF ===
br_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_100_mf_true/saved/"
br_num = "000007399.tar"
base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_100_restart2_nn/saved/"
base_num = "000007399.tar"

br_index = 34  # Assuming agent 34 is the BR agent

# === Helper functions ===

def kl_divergence_normal(mu_p, std_p, mu_q, std_q):
    std_p = std_p.clamp(min=1e-6)  # Avoid division by zero
    std_q = std_q.clamp(min=1e-6)  # Avoid division by zero
    var_p = std_p ** 2
    var_q = std_q ** 2
    print("mu_p:", mu_p, "std_p:", std_p, "mu_q:", mu_q, "std_q:", std_q)
    print("var_p:", var_p, "var_q:", var_q)
    kl = torch.log(std_q / std_p) + (var_p + (mu_p - mu_q)**2) / (2 * var_q) - 0.5
    return kl.sum(dim=-1)  # sum over action dims

def wasserstein_1d(mu_p, std_p, mu_q, std_q):
    # For multidim, sum over dims
    eps = 1e-6  # Small value to avoid division by zero
    std_p = std_p.clamp(min=eps)
    std_q = std_q.clamp(min=eps)
    return (torch.abs(mu_p - mu_q) + torch.abs(std_p - std_q)).sum(dim=-1)

def compute_divergences(obs_tensor, br_actor, base_actor):
    mu_br, std_br = br_actor.get_distribution_params(obs_tensor)
    mu_base, std_base = base_actor.get_distribution_params(obs_tensor)
    kl = kl_divergence_normal(mu_br, std_br, mu_base, std_base)
    wass = wasserstein_1d(mu_br, std_br, mu_base, std_base)
    return kl.detach().cpu().numpy(), wass.detach().cpu().numpy()

def load_data(path, name, networks, env):
    """
    Load saved data and update the networks.

    Parameters
    ----------
    path: str
    name: str
    networks: networks_ssd.Networks
    args: argparse.Namespace
    """
    checkpoint = torch.load(path + name, map_location="cpu")  # Load onto CPU by default
    # Restore network parameters
    for agent_type in range(env.num_types-1):
        print("Agent type:", agent_type)
        if args.mode_ac:
            networks.actor[agent_type].load_state_dict(checkpoint['actor'][agent_type])
            networks.actor_opt[agent_type].load_state_dict(checkpoint['actor_opt'][agent_type])
            networks.actor_target[agent_type].load_state_dict(checkpoint['actor'][agent_type])
        if args.mode_psi:
            networks.psi[agent_type].load_state_dict(checkpoint['psi'][agent_type])
            networks.psi_opt[agent_type].load_state_dict(checkpoint['psi_opt'][agent_type])
        else:
            networks.critic[agent_type].load_state_dict(checkpoint['critic'][agent_type])
            networks.critic_opt[agent_type].load_state_dict(checkpoint['critic_opt'][agent_type])
            networks.critic_target[agent_type].load_state_dict(checkpoint['critic'][agent_type])
    return checkpoint['args'], checkpoint['explore_params'], checkpoint['episode_trained'], checkpoint['time_trained'], checkpoint['outcomes']


def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    else:
        return obj

def roll_out(networks, env, args, init_set, epi_num, explore_params, paths, is_draw=False, is_train=True):
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

    actions_dict = {agent_id: [] for agent_id in agent_ids}
    log_probs_dict = {agent_id: [] for agent_id in agent_ids}
    mean_actions_dict = {f"type_{i}": [] for i in range(num_types)}
    obs_dict = {agent_id: [] for agent_id in agent_ids}
    all_infos = []
    all_profits = []
    all_backlog = []
    all_inv = []

    for i in range(epi_length):
        epsilon = explore_params['epsilon']
        beta = explore_params['beta']
        if args.mode_ac:
            rand_prob = np.random.rand(1)[0]
            if is_train and rand_prob < epsilon:
                act = {agent_id: np.random.uniform(low=env.action_space.low,
                                                   high=env.action_space.high,
                                                   size=env.action_space.shape) for agent_id in agent_ids}
                low = env.action_space.low
                high = env.action_space.high
                pdf_value = 1 / (high - low)
                if pdf_value == 0:
                    pdf_value = 1e-8
                act_probs = {agent_id: np.log(pdf_value) for agent_id in agent_ids}
            else:
                act, act_probs = networks.get_actions(obs, prev_m_act, GREEDY_BETA, is_target=False)
        else:
            if is_train:
                act, act_probs = networks.get_actions(obs, prev_m_act, beta, is_target=False)
            else:
                act, act_probs = networks.get_actions(obs, prev_m_act, GREEDY_BETA, is_target=False)


        n_obs, act, rew, m_act, n_obs, fea, infos = env.step(act)
        all_infos.append(infos)
        all_profits.append(infos['overall_profit'])
        all_backlog.append(infos['total_backlog'])
        all_inv.append(infos['total_inventory'])

        for agent_id in agent_ids:
            if isinstance(act[agent_id], float):
                actions_dict[agent_id].append(act[agent_id])  # Just append the scalar
            else:
                actions_dict[agent_id].append(act[agent_id].tolist())
            log_probs_dict[agent_id].append(act_probs[agent_id] if isinstance(act_probs[agent_id], float)
                                            else float(act_probs[agent_id]))
            obs_dict[agent_id].append(obs[agent_id].tolist())

        for type_id in range(num_types):
            val = m_act[type_id]
            if isinstance(val, (np.ndarray, list)):
                mean_actions_dict[f"type_{type_id}"].append(val.tolist() if hasattr(val, "tolist") else val)
            else:
                # assume val is dict or scalar, append as is
                mean_actions_dict[f"type_{type_id}"].append(val)

        
        samples[i] = (obs, act, act_probs, rew, m_act, n_obs, fea, beta)
        current_obs = samples[i][0]  # obs is the first element in the tuple

        for idx, agent_id in enumerate(agent_ids):
            agent_type = agent_types[idx]
            collective_reward[agent_type] += rew[agent_id]
            collective_feature[agent_type] += fea[agent_id]

        obs = n_obs
        prev_m_act = m_act


    init_set: dict = env.reset()

    return samples, init_set, collective_reward, all_infos, all_profits, all_backlog, all_inv


if __name__ == "__main__":
    # Seed setting.
    utils_all.set_random_seed(args.random_seed)

    # Build the environment.
    env = utils_ssd.get_env(args)
    init_set = env.reset()
    # Build networks
    base_networks = Networks(env, args)
    args_, explore_params, episode_trained, time_trained, outcome=load_data(base_path, base_num, base_networks, env)
    
    br_networks = Networks(env, args)
    args_br, explore_params_br, episode_trained_br, time_trained_br, outcome_br = load_data(br_path, br_num, br_networks, env)
    
    epsilon = explore_params['epsilon']
    beta = explore_params['beta']

    # Build paths for saving images.
    path, image_path, video_path, saved_path = utils_ssd.make_dirs(args)
    paths = [image_path, video_path, saved_path]

    # Metrics
    args.num_episodes = 2
    collective_rewards_test_base, collective_rewards_test_br = [np.zeros([args.num_types, args.num_episodes]) for _ in range(2)]
    objectives_test_base, objectives_test_br = [np.zeros(args.num_episodes) for _ in range(2)]

    time_start = time.time()

    # Save current setting(args) to txt for easy check
    utils_all.make_setting_txt(args, path)


    kl_list = []
    wass_list = []
    exploit_list = []
    obs_list = []
    # Run
    
    for i in range(args.num_episodes):
        # Decayed exploration parameters.
        explore_params = utils_ssd.get_explore_params(explore_params, i, args)
        samples_base, init_set_base, collective_reward_base, all_infos_base, all_profits_base, all_backlog_base, all_inv_base = roll_out(networks=base_networks,
                                                        env=env,
                                                        args=args,
                                                        init_set=init_set,
                                                        epi_num=i,
                                                        explore_params=explore_params,
                                                        paths=paths,
                                                        is_draw=False,
                                                        is_train=False,
                                                        )
        samples_br, init_set_br, collective_reward_br, all_infos_br, all_profits_br, all_backlog_br, all_inv_br = roll_out(networks=br_networks,
                                                        env=env,
                                                        args=args,
                                                        init_set=init_set,
                                                        epi_num=i,
                                                        explore_params=explore_params,
                                                        paths=paths,
                                                        is_draw=False,
                                                        is_train=False,
                                                        )

        for agent_type in range(args.num_types):
            collective_rewards_test_base[agent_type, i] = collective_reward_base[agent_type]
            objectives_test_base[i] = sum(collective_rewards_test_base[:, i])

        for agent_type in range(args.num_types):
            collective_rewards_test_br[agent_type, i] = collective_reward_br[agent_type]
            objectives_test_br[i] = sum(collective_rewards_test_br[:, i])
        
        # Compute divergences
        br_agent_id = br_index  # Assuming agent 8 is the BR agent
        br_actor = br_networks.actor[br_agent_id]
        base_actor = base_networks.actor[br_agent_id]  
        agent_list = list(init_set_base['obs'].keys())
        print("Agent list:", agent_list)
        agent_index = agent_list[br_agent_id]
        #obs_tensor = torch.stack([
        #    torch.tensor(init_set_base['obs'][agent_index], dtype=torch.float32)
        #])

        for epi in range(args.episode_length):
            # Compute KL divergence and Wasserstein distance
            samples_base_obs = samples_base[epi][0]
            obs = samples_base_obs[agent_index]
            obs_list.append(obs)  # Store the observation for later use
            obs_tensor = torch.stack([torch.tensor(obs, dtype=torch.float32)])
            kl, wass = compute_divergences(obs_tensor, br_actor, base_actor)
            kl_list.append(kl)
            wass_list.append(wass)
            exploitability = np.mean(np.abs(objectives_test_base[i] - objectives_test_br[i]))
            exploit_list.append(exploitability)
            print(f"Per time step {epi}, KL: {kl}, Wasserstein: {wass}, Exploitability: {exploitability}")

        # Save metrics to JSON file for this episode
        metrics = {
            'episode': i,
            'kl_divergences': convert_np(kl),
            'wasserstein_distances': convert_np(wass),
            'exploitability': exploitability,
            'collective_reward_base': objectives_test_base,
            'collective_reward_br': objectives_test_br,
            'all_infos_base': all_infos_base,
            'all_profits_base': all_profits_base,
            'all_backlog_base': all_backlog_base,
            'all_inv_base': all_inv_base,
            }

        metrics = convert_np(metrics)  # Convert numpy arrays to lists for JSON serialization
        metrics_file = os.path.join(saved_path, f"metrics_episode_{i}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Print progress
        print(f"Episode {i}/{args.num_episodes} - "
                  f"KL Divergence: {kl}, "
                  f"Wasserstein Distance: {wass}, "
                  f"Exploitability: {exploitability:.4f}")
            
        # Print progress
        print(f"Process : {i}/{args.num_episodes}, "
                  f"Time : {time.time() - time_start:.2f}, "
                  f"Collective reward (all types) : {sum(collective_rewards_test_base[:, i]):.2f}, "
                  f"Objective : {objectives_test_base[i]:.2f}, "
                  f"Test")

    print("Base", collective_rewards_test_base, np.mean(collective_rewards_test_base, axis = 1), np.std(collective_rewards_test_base, axis = 1))
    print("BR", collective_rewards_test_br, np.mean(collective_rewards_test_br, axis = 1), np.std(collective_rewards_test_br, axis = 1))
    