import os
from unittest import result
import numpy as np
import matplotlib.pyplot as plt

def ensure_result_directory(agent_type):
    result_dir = f"results/{agent_type}"
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def plot_episode_returns(all_returns, agent_type):
    result_dir = ensure_result_directory(agent_type)
    plt.figure(figsize=(12, 8))
    min_length = min(len(returns) for returns in all_returns)
    all_returns = [returns[:min_length] for returns in all_returns]
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    
    episodes = np.arange(1, min_length + 1)
    
    for i, seed_returns in enumerate(all_returns):
        plt.plot(episodes, seed_returns[:min_length], color='red', alpha=0.1)
    
    plt.plot(episodes, mean_returns, color='red', label="Mean Returns")
    
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Returns", fontsize=14)
    plt.title(f"{agent_type} agent episodic returns", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(result_dir, f"{agent_type}_agent_returns.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return

def plot_eval_returns(all_returns, agent_type, eval_interval):
    result_dir = ensure_result_directory(agent_type)
    plt.figure(figsize=(12, 8))
    min_length = min(len(returns) for returns in all_returns)
    all_returns = [returns[:min_length] for returns in all_returns]
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    
    steps = np.arange(1, len(mean_returns) + 1) * eval_interval
    
    for i, seed_returns in enumerate(all_returns):
        plt.plot(steps, seed_returns, color='red', alpha=0.1)
    
    plt.plot(steps, mean_returns, color='red', label="Mean Returns")
    
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Evaluation Returns", fontsize=14)
    plt.title(f"{agent_type} agent training performance", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(result_dir, f"{agent_type}_agent_eval_returns.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return plt