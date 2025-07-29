# Learning Under Changing Fields: Importance Sampling for Dynamic Mean-Field Agents

[![AAAI 2026](https://img.shields.io/badge/AAAI-2025-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Submitted to AAAI 2026** - *Conference on Artificial Intelligence*

## 📖 Abstract

This repository contains the implementation of our AAAI 2026 submission on **Learning Under Changing Fields: Importance Sampling for Dynamic Mean-Field Agents**. 

Mean-Field Reinforcement Learning (MFRL) enables scalable solutions for large-scale multi-agent systems by approximating agent interactions through population averages. However, a key limitation arises from temporal mismatches in the mean action distribution, which degrade learning performance as agent polices evolve. 

We address this by introducing an Importance Sampling (IS) correction framework that reweights past experiences to account for this distributional drift. We derive the IS correction term from first principles and provide a theoretical analysis establishing its asymptotic consistency. We also present finite-sample error bounds that reveal a trade-off between variance and scalability in large populations.

Empirical evaluations on large-scale inventory management tasks show that IS significantly improves profitability, inventory efficiency, and backlog reduction compared to standard MFRL. We further assess strategic robustness using approximate NashConv and distributional metrics, showing that IS-trained policies exhibit richer strategic behavior and multiple equilibria without sacrificing reward quality.

These results highlight the value of our proposed use of IS as a correction mechanism for dynamic shifts in mean action distributions in MFRL, going beyond its traditional role in off-policy learning.


## 🔬 Key Contributions

This work addresses temporal distribution shifts in mean-field multi-agent reinforcement learning (MFRL). Our main contributions are:

- **Importance Sampling Weight Correction**: We develop an importance sampling weight correction method to adjust for temporal discrepancies in the mean action distribution, enhancing learning consistency and reducing bias.

- **Finite-Sample Error Bounds**: We derive finite-sample error bounds for the IS-corrected value estimator, quantifying how estimation error behaves as a function of sample size and population scale. These results formally capture the trade-off between scalability and variance introduced by importance weighting, offering guidance on when the correction remains stable and reliable in practice.

- **Equilibrium Best-Response Analysis**: We conduct an equilibrium best-response analysis to characterize the convergence properties and robustness of the proposed method within the mean-field setting. The `eqm_analysis.py` module implements this by strategically excluding agent types and measuring best response gaps.

- **Empirical Validation**: Comprehensive evaluations on large-scale inventory management tasks demonstrating significant improvements in profitability, inventory efficiency, and backlog reduction compared to standard MFRL.

## 🏗️ Technical Framework

Our approach addresses the core challenge of temporal mismatches in mean action distributions that degrade MFRL performance as agent policies evolve. The framework combines:

- **Importance Sampling Correction**: Reweights past experiences to account for distributional drift in mean-field dynamics
- **Theoretical Foundation**: Derived from first principles with asymptotic consistency guarantees
- **Finite-Sample Analysis**: Error bounds revealing variance-scalability trade-offs in large populations
- **Strategic Robustness**: Assessment using approximate NashConv and distributional metrics

## 📊 Experimental Results

### Large-Scale Inventory Management
Our empirical evaluations demonstrate that IS significantly outperforms standard MFRL across multiple dimensions:

- **Profitability**: Enhanced collective rewards through better coordination
- **Inventory Efficiency**: Optimized stock levels reducing waste and shortages  
- **Backlog Reduction**: Minimized order delays and customer dissatisfaction

### Strategic Behavior Analysis
- **Richer Strategic Behavior**: IS-trained policies exhibit more diverse and robust strategies
- **Multiple Equilibria**: Discovery of alternative Nash equilibria without sacrificing reward quality
- **Distributional Robustness**: Enhanced performance under varying population dynamics
- **Best Response Analysis**: Comprehensive evaluation using `eqm_analysis.py` for Nash-conv assessment
- **Strategic Stability**: Policies demonstrate resilience against agent deviations and exploitability

### Statistical Validation
- **30-100 Agent Systems**: Scalability analysis across different population sizes
- **95% Confidence Intervals**: Rigorous statistical testing with significance markers
- **Distributional Analysis**: KDE plots showing reward distribution differences

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- PyTorch 1.12+ with CUDA 11.6
- Ray 2.41+ for distributed computing

### Environment Setup
```bash
# Clone the repository
git clone ....
cd mfmarl

# Create conda environment
conda env create -f environment.yml
conda activate mfmarll

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}')"
```

### Dependencies
```bash
# Core ML Libraries
torch==1.12.1
torchvision==0.13.1
torchaudio==0.12.1

# Multi-Agent RL
ray==2.41.0
gymnasium==0.26.3

# Data Analysis & Visualization
pandas==2.3.0
numpy==1.23.0
seaborn==0.13.2
matplotlib==3.8.4
scipy==1.9.2

# Additional Tools
h5py==3.12.1
pyyaml==6.0.2
```

## 🚀 Quick Start

### Training with Different Methods
```bash
# Train with Importance Sampling (IS) correction
python execute.py --method is --num_agents 50 --num_episodes 1000

# Train with standard Mean-Field (MF) approach  
python execute.py --method mf --num_agents 50 --num_episodes 1000
```
## 📁 Repository Structure

```text
mfmarl/
├── 📄 README.md                    # This file
├── 📄 environment.yml              # Conda environment
├── 📄 execute.py                   # Main training script
├── 📄 eqm_analysis.py              # Equilibrium & Nash-conv analysis
├── 📄 utils.py                     # Utility functions
├── 📄 utils_all.py                # Extended utilities
├── 📄 utils_ssd.py                # SSD-specific utilities
├── 📄 networks_backup.py          # Network architectures
├── 📄 parsed_args_ssd.py          # Argument parsing
├── 📄 run_restart.py              # Restart utilities
├── 📄 dump_pi.py                  # Policy dumping
├── 📄 env3rundivproduct.py        # Environment definition
└── 📄 __init__.py                 # Package initialization
```




## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

