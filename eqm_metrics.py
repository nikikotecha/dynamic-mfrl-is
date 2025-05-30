import torch
import numpy as np
from torch.distributions import Normal
import utils 
import utils_ssd
from networks_backup import Networks
from parsed_args_ssd import args

# === File Paths ===
base_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/eqm_30_is_true/saved/"
base_num = "000029999.tar"

file_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_30_2_nn/saved/"
file_num = "000029999.tar"

# === Helper functions ===

def kl_divergence_normal(mu_p, std_p, mu_q, std_q):
    var_p = std_p ** 2
    var_q = std_q ** 2
    kl = torch.log(std_q / std_p) + (var_p + (mu_p - mu_q)**2) / (2 * var_q) - 0.5
    return kl.sum(dim=-1)  # sum over action dims

def wasserstein_1d(mu_p, std_p, mu_q, std_q):
    # For multidim, sum over dims
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
        #"Agent type:", agent_type)
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

# === Modified roll_out to include divergence and exploitability ===

def roll_out_with_metrics(networks, base_networks, env, args, init_set, epi_num, explore_params, paths, br_agent_id, is_draw=False, is_train=True):
    image_path, video_path, saved_path = paths
    epi_length = args.episode_length
    agent_ids = env.node_names
    agent_types = list(env.agent_types.values())

    obs = init_set['obs']
    prev_m_act = init_set['m_act']

    kl_list = []
    wass_list = []
    rewards_br = 0.0
    rewards_base = 0.0

    # For baseline rollout, we do a separate environment reset to simulate baseline policy on same seed
    #baseline_env = env.clone() if hasattr(env, 'clone') else env  # clone env if possible, else reuse (may be inaccurate)
    baseline_env = utils_ssd.get_env(args)  # Create a new environment instance for baseline
    baseline_env.reset()  # Reset baseline environment
    # Initialize baseline observations
    baseline_obs = init_set['obs']

    for i in range(epi_length):
        # --- BR policy actions ---
        # === Build mixed actions: BR agent uses BR policy, others use base ===

        # --- Compute divergences for BR agent ---
        agent_id = agent_ids[br_agent_id]
        obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)

        br_actor = networks.actor[br_agent_id]
        base_actor = base_networks.actor[br_agent_id]

        kl, wass = compute_divergences(obs_tensor, br_actor, base_actor)
        kl_list.append(float(kl[0]))
        wass_list.append(float(wass[0]))

        # --- Step environment with BR actions ---
        n_obs, act, rew, m_act, n_obs, fea, infos = env.step(act)
        rewards_br += rew[agent_id]

        # --- Baseline policy action for same agent ---
        base_act, _ = base_networks.get_actions(baseline_obs, prev_m_act, explore_params['beta'], is_target=False)
        n_base_obs, base_act, base_rew, base_m_act, _, _, _ = baseline_env.step(base_act)
        rewards_base += base_rew[agent_id]

        # Update observations for next step
        obs = n_obs
        prev_m_act = m_act
        baseline_obs = n_base_obs

    exploitability = rewards_br - rewards_base

    # Save metrics to JSON file for this episode
    import json
    metrics = {
        'episode': epi_num,
        'kl_divergences': kl_list,
        'wasserstein_distances': wass_list,
        'exploitability': exploitability,
        'cumulative_reward_br': rewards_br,
        'cumulative_reward_base': rewards_base
    }
    metrics_path = f"{saved_path}/metrics_episode_{str(epi_num).zfill(5)}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Episode {epi_num} - Exploitability: {exploitability:.4f}, Mean KL: {np.mean(kl_list):.4f}, Mean Wasserstein: {np.mean(wass_list):.4f}")

    return kl_list, wass_list, exploitability, rewards_br, rewards_base


# === Main run with baseline and BR networks ===

if __name__ == "__main__":
    # Load environment and networks (BR)
    env = utils_ssd.get_env(args)
    init_set = env.reset()
    networks = Networks(env, args)  # BR networks

    # Load baseline networks from file (make sure you have these saved)
    base_networks = Networks(env, args)
    base_args, base_params, base_episode, base_time, base_outcomes = load_data(base_path, base_num, base_networks, env)

    # Load BR checkpoint
    args_, explore_params, episode_trained, time_trained, outcome = load_data(file_path, file_num, networks, env)

    # Select BR agent index (e.g., agent 0)
    br_agent_id = 8

    # Build paths
    path, image_path, video_path, saved_path = utils_ssd.make_dirs(args)
    paths = [image_path, video_path, saved_path]

    args.num_episodes = 1
    for p1, p2 in zip(base_networks.actor[br_agent_id].parameters(), networks.actor[br_agent_id].parameters()):
        print(torch.allclose(p1, p2))  # Should be False if policies are different

    for i in range(args.num_episodes):
        explore_params = utils_ssd.get_explore_params(explore_params, i, args)

        kl_list, wass_list, exploitability, r_br, r_base = roll_out_with_metrics(
            networks=networks,
            base_networks=base_networks,
            env=env,
            args=args,
            init_set=init_set,
            epi_num=i,
            explore_params=explore_params,
            paths=paths,
            br_agent_id=br_agent_id,
            is_draw=False,
            is_train=False
        )

        # Reset env after each episode
        init_set = env.reset()

