import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats

def load_json_files(file_patterns):
    """Load mul    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add significance legend as simple text
    sig_text = "*** p<0.001, ** p<0.01, * p<    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add significance legend as simple text
    sig_text = "*** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    ax.text(0.02, 0.02, sig_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom')
    
    plt.tight_layout(pad=2.0)
    plt.savefig('z_backlog_bar_chart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('z_backlog_bar_chart.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')significant"
    ax.text(0.02, 0.02, sig_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom')
    
    plt.tight_layout(pad=2.0)
    plt.savefig('z_inventory_bar_chart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('z_inventory_bar_chart.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')s based on patterns like '70_is', '70_mf', etc."""
    data = {}
    
    for pattern in file_patterns:
        file_path = "results_ssd/execute/" + f"{pattern}_metrics_results.json"
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                # Extract number of agents and method from filename
                parts = pattern.split('_')
                num_agents = int(parts[0])
                method = parts[1]
                
                # Store raw data for confidence interval calculation
                inventory_data = content['inventory'].get('raw_data', [])
                backlog_data = content['backlog'].get('raw_data', [])
                
                data[pattern] = {
                    'num_agents': num_agents,
                    'method': method,
                    'rewards': content['collective_rewards_test'][0],
                    'inventory_avg': content['inventory']['average_inv'],
                    'inventory_std': content['inventory']['std_deviation_inv'],
                    'inventory_raw': inventory_data,
                    'backlog_avg': content['backlog']['average_backlog'],
                    'backlog_std': content['backlog']['std_deviation_backlog'],
                    'backlog_raw': backlog_data
                }
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for given data"""
    if len(data) == 0:
        return 0, 0
    
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    
    return interval[0], interval[1]

def perform_statistical_tests(data1, data2, method1_name, method2_name):
    """Perform statistical significance tests between two datasets"""
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(data1, data2)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
    
    # Perform Mann-Whitney U test (non-parametric alternative)
    u_stat, u_p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    # Determine significance level
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significance': significance,
        'u_stat': u_stat,
        'u_p_value': u_p_value,
        'method1': method1_name,
        'method2': method2_name,
        'mean1': np.mean(data1),
        'mean2': np.mean(data2)
    }

def plot_inventory_bar_chart(data):
    """Create bar chart for inventory with 95% confidence intervals and significance testing - AAAI paper quality"""
    # Organize data for plotting and significance testing
    df_list = []
    significance_results = {}
    
    for pattern, info in data.items():
        # Calculate 95% confidence interval
        if info['inventory_raw']:
            ci_lower, ci_upper = calculate_confidence_interval(info['inventory_raw'])
            ci_error = [info['inventory_avg'] - ci_lower, ci_upper - info['inventory_avg']]
        else:
            # Fallback to standard error if raw data not available
            ci_error = [1.96 * info['inventory_std'] / np.sqrt(30)] * 2  # Assuming n=30
        
        df_list.append({
            'num_agents': info['num_agents'],
            'method': info['method'].upper(),
            'inventory_avg': info['inventory_avg'],
            'ci_error': ci_error,
            'raw_data': info['inventory_raw']
        })
    
    df = pd.DataFrame(df_list)
    
    # Perform significance tests between IS and MF for each agent count
    agents = sorted(df['num_agents'].unique())
    for agent_count in agents:
        is_data = df[(df['num_agents'] == agent_count) & (df['method'] == 'IS')]
        mf_data = df[(df['num_agents'] == agent_count) & (df['method'] == 'MF')]
        
        if len(is_data) > 0 and len(mf_data) > 0 and len(is_data['raw_data'].iloc[0]) > 0 and len(mf_data['raw_data'].iloc[0]) > 0:
            result = perform_statistical_tests(
                is_data['raw_data'].iloc[0], 
                mf_data['raw_data'].iloc[0], 
                'IS', 'MF'
            )
            significance_results[agent_count] = result
    
    # Set up AAAI paper style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color palette consistent with KDE plots
    colors = {'IS': '#2E86AB', 'MF': '#A23B72'}
    
    # Get unique number of agents and methods
    methods = sorted(df['method'].unique())
    
    x = np.arange(len(agents))
    width = 0.35
    
    # Plot bars for each method
    bars_dict = {}
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        
        # Align data with agents order
        avg_values = []
        error_values = []
        for agent_count in agents:
            method_agent_data = method_data[method_data['num_agents'] == agent_count]
            if len(method_agent_data) > 0:
                avg_values.append(method_agent_data['inventory_avg'].iloc[0])
                error_values.append(method_agent_data['ci_error'].iloc[0])
            else:
                avg_values.append(0)
                error_values.append([0, 0])
        
        # Convert error values to format expected by matplotlib
        error_lower = [err[0] for err in error_values]
        error_upper = [err[1] for err in error_values]
        
        bars = ax.bar(x + i*width - width/2, avg_values, width, 
                     label=method, yerr=[error_lower, error_upper], capsize=5, 
                     color=colors[method], alpha=0.8, edgecolor='black', linewidth=0.8)
        bars_dict[method] = bars
    
    # Add significance annotations
    max_height = max([max(df[df['method'] == method]['inventory_avg']) for method in methods])
    for i, agent_count in enumerate(agents):
        if agent_count in significance_results:
            result = significance_results[agent_count]
            # Position significance marker above the bars
            """y_pos = max_height * 1.15
            ax.text(x[i], y_pos, result['significance'], 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')"""
            
            # Add connecting line between bars if significant
            if result['significance'] != 'ns':
                bar1_x = x[i] - width/2
                bar2_x = x[i] + width/2
                line_y = max_height * 1.08
                ax.plot([bar1_x, bar2_x], [line_y, line_y], 'k-', linewidth=1)
                ax.plot([bar1_x, bar1_x], [line_y-max_height*0.02, line_y], 'k-', linewidth=1)
                ax.plot([bar2_x, bar2_x], [line_y-max_height*0.02, line_y], 'k-', linewidth=1)
    
    # AAAI paper formatting
    ax.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Inventory', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=11)
    
    # Clean styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Adjust y-axis to accommodate significance markers
    ax.set_ylim(0, max_height * 1.25)
    
    # Simple legend without boxes
    ax.legend(frameon=False, loc='upper left', fontsize=11)
    
    # Add significance legend
    """sig_text = "Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
           facecolor='white', alpha=0.8))"""
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('z_inventory_bar_chart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('z_inventory_bar_chart.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Print significance results
    print("\nInventory Statistical Significance Results:")
    print("-" * 80)
    for agent_count, result in significance_results.items():
        print(f"{agent_count} agents: {result['method1']} vs {result['method2']}")
        print(f"  Mean {result['method1']}: {result['mean1']:.2f}, Mean {result['method2']}: {result['mean2']:.2f}")
        print(f"  t-statistic: {result['t_stat']:.3f}, p-value: {result['p_value']:.4f} {result['significance']}")
        print(f"  Cohen's d: {result['cohens_d']:.3f}, Mann-Whitney p: {result['u_p_value']:.4f}")
        print()
    
    plt.show()

def plot_backlog_bar_chart(data):
    """Create bar chart for backlog with 95% confidence intervals and significance testing - AAAI paper quality"""
    # Organize data for plotting and significance testing
    df_list = []
    significance_results = {}
    
    for pattern, info in data.items():
        # Calculate 95% confidence interval
        if info['backlog_raw']:
            ci_lower, ci_upper = calculate_confidence_interval(info['backlog_raw'])
            ci_error = [info['backlog_avg'] - ci_lower, ci_upper - info['backlog_avg']]
        else:
            # Fallback to standard error if raw data not available
            ci_error = [1.96 * info['backlog_std'] / np.sqrt(30)] * 2  # Assuming n=30
        
        df_list.append({
            'num_agents': info['num_agents'],
            'method': info['method'].upper(),
            'backlog_avg': info['backlog_avg'],
            'ci_error': ci_error,
            'raw_data': info['backlog_raw']
        })
    
    df = pd.DataFrame(df_list)
    
    # Perform significance tests between IS and MF for each agent count
    agents = sorted(df['num_agents'].unique())
    for agent_count in agents:
        is_data = df[(df['num_agents'] == agent_count) & (df['method'] == 'IS')]
        mf_data = df[(df['num_agents'] == agent_count) & (df['method'] == 'MF')]
        
        if len(is_data) > 0 and len(mf_data) > 0 and len(is_data['raw_data'].iloc[0]) > 0 and len(mf_data['raw_data'].iloc[0]) > 0:
            result = perform_statistical_tests(
                is_data['raw_data'].iloc[0], 
                mf_data['raw_data'].iloc[0], 
                'IS', 'MF'
            )
            significance_results[agent_count] = result
    
    # Set up AAAI paper style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color palette consistent with KDE plots
    colors = {'IS': '#2E86AB', 'MF': '#A23B72'}
    
    # Get unique number of agents and methods
    methods = sorted(df['method'].unique())
    
    x = np.arange(len(agents))
    width = 0.35
    
    # Plot bars for each method
    bars_dict = {}
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        
        # Align data with agents order
        avg_values = []
        error_values = []
        for agent_count in agents:
            method_agent_data = method_data[method_data['num_agents'] == agent_count]
            if len(method_agent_data) > 0:
                avg_values.append(method_agent_data['backlog_avg'].iloc[0])
                error_values.append(method_agent_data['ci_error'].iloc[0])
            else:
                avg_values.append(0)
                error_values.append([0, 0])
        
        # Convert error values to format expected by matplotlib
        error_lower = [err[0] for err in error_values]
        error_upper = [err[1] for err in error_values]
        
        bars = ax.bar(x + i*width - width/2, avg_values, width, 
                     label=method, yerr=[error_lower, error_upper], capsize=5, 
                     color=colors[method], alpha=0.8, edgecolor='black', linewidth=0.8)
        bars_dict[method] = bars
    
    # Add significance annotations
    max_height = max([max(df[df['method'] == method]['backlog_avg']) for method in methods])
    for i, agent_count in enumerate(agents):
        if agent_count in significance_results:
            result = significance_results[agent_count]
            # Position significance marker above the bars
            """y_pos = max_height * 1.15
            ax.text(x[i], y_pos, result['significance'], 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')"""
            
            # Add connecting line between bars if significant
            if result['significance'] != 'ns':
                bar1_x = x[i] - width/2
                bar2_x = x[i] + width/2
                line_y = max_height * 1.08
                ax.plot([bar1_x, bar2_x], [line_y, line_y], 'k-', linewidth=1)
                ax.plot([bar1_x, bar1_x], [line_y-max_height*0.02, line_y], 'k-', linewidth=1)
                ax.plot([bar2_x, bar2_x], [line_y-max_height*0.02, line_y], 'k-', linewidth=1)
    
    # AAAI paper formatting
    ax.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Backlog', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=11)
    
    # Clean styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Adjust y-axis to accommodate significance markers
    ax.set_ylim(0, max_height * 1.25)
    
    # Simple legend without boxes
    ax.legend(frameon=False, loc='upper left', fontsize=11)
    
    """    # Add significance legend
        sig_text = "Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', alpha=0.8))
    """    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('z_backlog_bar_chart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('z_backlog_bar_chart.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Print significance results
    print("\nBacklog Statistical Significance Results:")
    print("-" * 80)
    for agent_count, result in significance_results.items():
        print(f"{agent_count} agents: {result['method1']} vs {result['method2']}")
        print(f"  Mean {result['method1']}: {result['mean1']:.2f}, Mean {result['method2']}: {result['mean2']:.2f}")
        print(f"  t-statistic: {result['t_stat']:.3f}, p-value: {result['p_value']:.4f} {result['significance']}")
        print(f"  Cohen's d: {result['cohens_d']:.3f}, Mann-Whitney p: {result['u_p_value']:.4f}")
        print()
    
    plt.show()

def plot_rewards_kde(data):
    """Create KDE plot for rewards distribution with 95% confidence intervals and significance testing"""
    # Group data by number of agents
    agents_data = {}
    significance_results = {}
    
    for pattern, info in data.items():
        num_agents = info['num_agents']
        if num_agents not in agents_data:
            agents_data[num_agents] = {}
        agents_data[num_agents][info['method']] = info['rewards']
    
    # Perform significance tests for each agent count
    for num_agents, methods_data in agents_data.items():
        if 'is' in methods_data and 'mf' in methods_data:
            result = perform_statistical_tests(
                methods_data['is'], methods_data['mf'], 'IS', 'MF'
            )
            significance_results[num_agents] = result
    
    # Debug: Print what data we have
    print("\nDebug - Available data:")
    for num_agents, methods_data in agents_data.items():
        print(f"{num_agents} agents: {list(methods_data.keys())}")
        for method, rewards in methods_data.items():
            print(f"  {method}: {len(rewards)} data points")
    
    # Set up the plot style for AAAI paper
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create subplots for each number of agents
    num_plots = len(agents_data)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Color palette for methods
    colors = {'is': '#2E86AB', 'mf': '#A23B72'}
    
    for idx, (num_agents, methods_data) in enumerate(sorted(agents_data.items())):
        ax = axes[idx]
        
        for method, rewards in methods_data.items():
            # Skip if no data
            if len(rewards) == 0:
                print(f"Warning: No data for {num_agents} agents, {method} method")
                continue
            
            # Check if all values are the same (std = 0)
            if np.std(rewards) == 0:
                print(f"Warning: All values identical for {num_agents} agents, {method} method. Adding vertical line instead of KDE.")
                # Plot a vertical line at the constant value
                ax.axvline(x=np.mean(rewards), color=colors.get(method, f'C{idx}'), 
                          linewidth=3, alpha=0.8, label=f'{method.upper()} (constant)')
            else:
                # Plot KDE for variable data
                sns.kdeplot(
                    data=rewards, 
                    ax=ax,
                    label=method.upper(),
                    color=colors.get(method, f'C{idx}'),
                    fill=True,
                    alpha=0.6,
                    linewidth=2.5
                )
            
            # Add confidence interval markers
            ci_lower, ci_upper = calculate_confidence_interval(rewards)
            
            # Only add CI lines if data is not constant
            if np.std(rewards) > 0:
                # Add vertical lines for confidence interval (only label once for legend)
                if idx == 0 and method == 'is':  # Only add label for first plot and first method
                    ax.axvline(ci_lower, color=colors.get(method, f'C{idx}'), 
                              linestyle='--', alpha=0.8, linewidth=1.5)
                    ax.axvline(ci_upper, color=colors.get(method, f'C{idx}'), 
                              linestyle='--', alpha=0.8, linewidth=1.5)
                else:
                    ax.axvline(ci_lower, color=colors.get(method, f'C{idx}'), 
                              linestyle='--', alpha=0.8, linewidth=1.5)
                    ax.axvline(ci_upper, color=colors.get(method, f'C{idx}'), 
                              linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Add significance information as text
        if num_agents in significance_results:
            result = significance_results[num_agents]
            """textstr = f'p-value: {result["p_value"]:.3f} {result["significance"]}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top')"""
            significance_str = f'p = {result["p_value"]:.3f} {result["significance"]}'
            # Add a dummy line for legend entry
            ax.plot([], [], ' ', label=significance_str)  # Empty plot for legend entry

        # Formatting for AAAI paper standards
        ax.set_xlabel('Collective Rewards', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'{num_agents} Agents', fontsize=14, fontweight='bold', pad=15)
        
        # Clean up the plot
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Legend styling
        ax.legend(frameon=False, loc='upper right', fontsize=11)
        
        # Tick formatting
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    plt.tight_layout(pad=2.0)
    plt.savefig('z_rewards_kde_plot.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('z_rewards_kde_plot.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Print significance results
    print("\nRewards Statistical Significance Results:")
    print("-" * 80)
    for num_agents, result in significance_results.items():
        print(f"{num_agents} agents: {result['method1']} vs {result['method2']}")
        print(f"  Mean {result['method1']}: {result['mean1']:.2f}, Mean {result['method2']}: {result['mean2']:.2f}")
        print(f"  t-statistic: {result['t_stat']:.3f}, p-value: {result['p_value']:.4f} {result['significance']}")
        print(f"  Cohen's d: {result['cohens_d']:.3f} ({get_effect_size_interpretation(result['cohens_d'])})")
        print(f"  Mann-Whitney p: {result['u_p_value']:.4f}")
        print()
    
    plt.show()

def get_effect_size_interpretation(cohens_d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def main():
    # Define file patterns to load
    file_patterns = ['100_mf', '100_is', '70_is', '70_mf', '50_is', '50_mf', '30_is', '30_mf']
    
    # Load data from JSON files
    data = load_json_files(file_patterns)
    
    if not data:
        print("No data loaded. Please check file paths and formats.")
        return
    
    # Generate plots
    plot_rewards_kde(data)
    plot_inventory_bar_chart(data)
    plot_backlog_bar_chart(data)
    
    # Print summary statistics with significance testing
    print("\nSummary Statistics with Statistical Significance:")
    print("=" * 100)
    for pattern, info in data.items():
        rewards = info['rewards']
        ci_lower, ci_upper = calculate_confidence_interval(rewards)
        print(f"{pattern}: Rewards Mean={np.mean(rewards):.1f} (95% CI: {ci_lower:.1f}-{ci_upper:.1f}), "
              f"Std={np.std(rewards):.1f}, "
              f"Inventory Avg={info['inventory_avg']:.1f}")
    
    print("\nOverall Statistical Summary:")
    print("=" * 100)
    print("*** p<0.001 (highly significant)")
    print("**  p<0.01  (very significant)")  
    print("*   p<0.05  (significant)")
    print("ns         (not significant)")
    print("\nCohen's d interpretation:")
    print("negligible: |d| < 0.2, small: 0.2 ≤ |d| < 0.5, medium: 0.5 ≤ |d| < 0.8, large: |d| ≥ 0.8")

if __name__ == "__main__":
    main()