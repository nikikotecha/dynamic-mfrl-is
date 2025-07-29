import os
import json
import matplotlib.pyplot as plt
import numpy as np
# Paths for each method and agent count
paths = {
    "IS": {
        30: "results_ssd/eqm_BR/execute/30_is_execute/saved",
        50: "results_ssd/eqm_BR/execute/50_is_execute/saved",
        70: "results_ssd/eqm_BR/execute/z_eqm_archive_run2/70_is_execute/saved",
        100: "results_ssd/eqm_BR/execute/z_eqm_archive_run2/100_is_execute/saved"
    },
    "MF": {
        30: "results_ssd/eqm_BR/execute/30_mf_execute/saved",
        50: "results_ssd/eqm_BR/execute/50_mf_execute/saved",
        70: "results_ssd/eqm_BR/execute/70_mf_execute/saved",
        100: "results_ssd/eqm_BR/execute/100_mf_execute/saved"
    }
}

def extract_metrics(directory):
    exploitability, kl_vals, wass_vals = [], [], []
    inventory, backlog = [], []
    rewards = []
    for filename in os.listdir(directory):
        if filename.startswith("metrics_episode_") and filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                # Extract metrics if they exist
                if (e := data.get("exploitability")) is not None:
                    exploitability.append(e)
                # KL
                k = data.get("kl", [])
                if isinstance(k, list):
                    kl_vals.extend(k)
                elif k is not None:
                    kl_vals.append(k)

                # Wasserstein
                w = data.get("wasserstein", [])
                if isinstance(w, list):
                    wass_vals.extend(w)
                elif w is not None:
                    wass_vals.append(w)

                rewards.append(data.get("collective_reward_base", [0]))
    
    return exploitability, kl_vals, wass_vals, inventory, backlog, rewards

agent_counts = sorted(paths["IS"].keys())

# Prepare containers
results = {
    "IS": {"agent_counts": {n: {"exploit": [], "kl": [], "wass": [], "exploit_std": [], "kl_std": [], "wass_std": [], "inventory": [], "inventory_std": [], "backlog": [], "backlog_std":[], "rewards":[]} for n in agent_counts}},
    "MF": {"agent_counts": {n: {"exploit": [], "kl": [], "wass": [], "exploit_std": [], "kl_std": [], "wass_std": [], "inventory": [], "inventory_std": [], "backlog": [], "backlog_std":[], "rewards":[]} for n in agent_counts}}
}

# Extract data for each setting
for method in ["IS", "MF"]:
    for n in agent_counts:
        exploit, kl, wass, inventory, av_backlog, rewards = extract_metrics(paths[method][n])
        results[method]["agent_counts"][n]["exploit"] = np.mean(exploit)
        results[method]["agent_counts"][n]["kl"] = np.mean(kl)
        results[method]["agent_counts"][n]["wass"] = np.mean(wass)
        results[method]["agent_counts"][n]["exploit_std"] = np.std(exploit)
        results[method]["agent_counts"][n]["kl_std"] = np.std(kl)
        results[method]["agent_counts"][n]["wass_std"] = np.std(wass)
        print("rewards", rewards)
        print("rewards length", len(rewards))
        last_collective_rewards = [rewards[-1] for rewards in rewards]
        print("last_collective_rewards", last_collective_rewards)
        results[method]["agent_counts"][n]["rewards"] = last_collective_rewards

print("Data extraction complete.") 
print("Results:", results)
def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    else:
        return obj

results = convert_np(results)
# Save results to JSON
output_file = "eqm_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

# ---------- PLOTTING FUNCTION ----------
def plot_metric(metric_name, ylabel, filename):
    plt.figure(figsize=(8, 6))
    for method, color in zip(["IS", "MF"], ["steelblue", "darkorange"]):
        plt.plot(agent_counts, results[method][metric_name], label=method, color=color, linewidth=2)
        plt.fill_between(agent_counts,
                         np.array(results[method][metric_name]) - np.array(results[method][metric_name + "_std"]),
                         np.array(results[method][metric_name]) + np.array(results[method][metric_name + "_std"]),
                         alpha=0.3, color=color)
    plt.xlabel("Number of Agents", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} vs Agent Count", fontsize=14)
    plt.legend(title="Method")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.autoscale()  # Automatically adjust axis limits
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# ---------- PLOT EACH METRIC ----------
plot_metric("exploit", "Exploitability", "exploitability_vs_agents.png")
plot_metric("kl", "KL Divergence", "kl_vs_agents.png")
plot_metric("wass", "Wasserstein Distance", "wasserstein_vs_agents.png")
