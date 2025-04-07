import numpy as np
import matplotlib.pyplot as plt

def plot_episode_returns(all_returns):
    plt.figure(figsize=(12, 8))
    
    min_length = min(len(returns) for returns in all_returns)
    all_returns = [returns[:min_length] for returns in all_returns]
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    
    episodes = np.arange(1, min_length + 1)
    
    for i, seed_returns in enumerate(all_returns):
        plt.plot(episodes, seed_returns[:min_length], color='red', alpha=0.3)
    
    plt.plot(episodes, mean_returns, color='red', label="Mean Returns")
    
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Returns", fontsize=14)
    plt.title("Episodic Returns across Multiple Seeds", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_episode_returns.png", dpi=300)

    return

def plot_eval_returns(all_returns, eval_interval):
    plt.figure(figsize=(12, 8))

    min_length = min(len(returns) for returns in all_returns)
    all_returns = [returns[:min_length] for returns in all_returns]
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    
    steps = np.arange(1, len(mean_returns) + 1) * eval_interval
    
    for i, seed_returns in enumerate(all_returns):
        plt.plot(steps, seed_returns, color='red', alpha=0.3)
    
    plt.plot(steps, mean_returns, color='red', label="Mean Returns")
    
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Evaluation Returns", fontsize=14)
    plt.title("Training Performance across Multiple Seeds", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval_returns.png", dpi=300)
    
    return plt