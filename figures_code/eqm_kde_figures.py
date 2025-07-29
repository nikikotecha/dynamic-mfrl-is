import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats 


def compute_analytical_ci(data, confidence=0.95):
    """Compute analytical confidence interval for the mean using t-distribution."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    # t critical value for two-tailed test
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = sem * t_crit  # margin of error
    
    # Return mean and margin of error (so CI = mean ± margin)
    return mean, margin


def load_episode_json_files(file_patterns, base_path="/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/execute/"):
    """Load episode JSON files for base vs BR comparison"""
    data = {}
    for pattern in file_patterns:
        file_path = f"{base_path}{pattern}_execute/saved/metrics_episode_99.json"
        if pattern == '70_mf':
            file_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/execute/70_mf_execute/saved/metrics_episode_76.json"
        if pattern == "100_is":
            file_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/execute/100_is_execute/saved/metrics_episode_47.json"
        if pattern == "100_mf":
            file_path = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/eqm_BR/execute/100_mf_execute/saved/metrics_episode_47.json"
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                # Extract number of agents and method from pattern
                parts = pattern.split('_')
                num_agents = int(parts[0])
                method = parts[1]
                
                data[pattern] = {
                    'num_agents': num_agents,
                    'method': method,
                    'collective_reward_base': content['collective_reward_base'],
                    'collective_reward_br': content['collective_reward_br']
                }
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data

def plot_base_br_comparison_grid(data):
    """Create grid plot comparing base vs BR rewards for each agent/method combination"""
    # Organize data by agents and methods
    agents_methods = {}
    for pattern, info in data.items():
        num_agents = info['num_agents']
        method = info['method']
        if num_agents not in agents_methods:
            agents_methods[num_agents] = {}
        agents_methods[num_agents][method] = info
    
    # Set up the plot style for AAAI paper
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Calculate grid dimensions (2 rows, enough columns for all combinations)
    num_agents_list = sorted(agents_methods.keys())
    methods_list = ['is', 'mf']  # assuming these are the methods
    total_plots = len(num_agents_list) * len(methods_list)
    cols = len(num_agents_list)
    rows = len(methods_list)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Color palette for methods (consistent with previous plots)
    method_colors = {'is': '#2E86AB', 'mf': '#A23B72'}
    
    for row, method in enumerate(methods_list):
        for col, num_agents in enumerate(num_agents_list):
            ax = axes[row, col]
            
            # Check if this combination exists
            if num_agents in agents_methods and method in agents_methods[num_agents]:
                info = agents_methods[num_agents][method]
                method_color = method_colors[method]
                
                # Compute confidence intervals
                base_mean, base_ci = compute_analytical_ci(info['collective_reward_base'])
                br_mean, br_ci = compute_analytical_ci(info['collective_reward_br'])
                

                # Plot KDE for base rewards (solid line, lighter fill)
                sns.kdeplot(
                    data=info['collective_reward_base'], 
                    ax=ax,
                    label='Base Policy',
                    color=method_color,
                    fill=False,
                    alpha=0.5,  # Lighter fill for base
                    linewidth=2.5,
                    linestyle='-'  # Solid line
                )
                
                # Add confidence interval lines for base policy
                ax.axvline(base_mean - base_ci, color=method_color, linestyle=':', alpha=0.7, linewidth=1.5)
                ax.axvline(base_mean + base_ci, color=method_color, linestyle=':', alpha=0.7, linewidth=1.5)

                # Plot KDE for BR rewards (dashed line, darker fill)
                sns.kdeplot(
                    data=info['collective_reward_br'], 
                    ax=ax,
                    label='Best Response',
                    color=method_color,
                    fill=True,
                    alpha=0.2,  # Darker fill for BR
                    linewidth=2.5,
                    linestyle='--'  # Dashed line
                )
                
                # Formatting for AAAI paper standards
                ax.set_xlabel('Collective Rewards', fontsize=11, fontweight='bold')
                ax.set_ylabel('Density', fontsize=11, fontweight='bold')
                ax.set_title(f'{num_agents} Agents - {method.upper()}', fontsize=12, fontweight='bold', pad=10)
                
            else:
                # Empty subplot if combination doesn't exist
                ax.set_title(f'{num_agents} Agents - {method.upper()}\n(No Data)', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, style='italic')
            
            # Clean up the plot
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            
            # Legend styling (only for plots with data)
            if num_agents in agents_methods and method in agents_methods[num_agents]:
                legend = ax.legend(frameon=False,
                                  fontsize=9, loc='upper left')
                """legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)"""
            
            # Tick formatting
            ax.tick_params(axis='both', which='major', labelsize=9)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('z_base_br_comparison_grid.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def main():
    # Define file patterns to load (adjust based on your actual files)
    file_patterns = ['30_is', '30_mf', '50_is', '50_mf', '70_is', '70_mf', '100_is', '100_mf']
    
    # Load episode data
    episode_data = load_episode_json_files(file_patterns)
    
    if not episode_data:
        print("No episode data loaded. Please check file paths and formats.")
        return
    
    # Generate grid plot
    plot_base_br_comparison_grid(episode_data)
    
    # Print summary statistics
    print("\nBase vs BR Summary Statistics:")
    print("-" * 60)
    for pattern, info in episode_data.items():
        base_rewards = info['collective_reward_base']
        br_rewards = info['collective_reward_br']
        print(f"{pattern}: Base Mean={np.mean(base_rewards):.1f}, "
              f"BR Mean={np.mean(br_rewards):.1f}, "
              f"Improvement={np.mean(br_rewards) - np.mean(base_rewards):.1f}")

if __name__ == "__main__":
    main()