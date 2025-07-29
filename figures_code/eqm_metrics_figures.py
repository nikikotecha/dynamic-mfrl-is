import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd

# Data
agents = [30, 50, 70, 100]
methods = ['MF', 'IS']

kl_values = {
    'MF': [4.2970104, 0.7875463, 29.499552, 31.036346],
    'IS': [3.27e24, 2.99e22, 1.77e11, 520646]
}

wasserstein_values = {
    'MF': [24.080467, 0.8445944, 56.755985, 63.04894],
    'IS': [2557698, 244728, 4392210, 7547]
}

def plot_separate_log_heatmaps():
    """Create separate heatmaps with log-transformed values"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Prepare data and take log
    kl_matrix = np.array([kl_values['MF'], kl_values['IS']])
    ws_matrix = np.array([wasserstein_values['MF'], wasserstein_values['IS']])
    
    # Take log10 of the values
    log_kl_matrix = np.log10(kl_matrix)
    log_ws_matrix = np.log10(ws_matrix)
    
    # KL Divergence heatmap (separate plot)
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    im1 = sns.heatmap(log_kl_matrix, 
                      xticklabels=agents, 
                      yticklabels=methods,
                      annot=True, 
                      fmt='.1f',
                      cmap='viridis',
                      ax=ax1,
                      cbar_kws={'label': 'Log₁₀(KL Divergence)'},
                      annot_kws={"size":14})  # <-- add this

    
    ax1.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Method', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('z_kl_divergence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Wasserstein Distance heatmap (separate plot)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    im2 = sns.heatmap(log_ws_matrix,
                      xticklabels=agents,
                      yticklabels=methods, 
                      annot=True,
                      fmt='.1f',
                      cmap='plasma',
                      ax=ax2,
                      cbar_kws={'label': 'Log₁₀(Wasserstein Distance)'},
                      annot_kws={"size":14})
    
    ax2.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Method', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('z_wasserstein_distance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_separate_log_heatmaps()            