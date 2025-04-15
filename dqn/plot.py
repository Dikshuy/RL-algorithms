import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def ensure_result_directory(agent_type):
    result_dir = f"results/{agent_type}"
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def save_seed_data(episode_returns, eval_returns, agent_type, seed):
    result_dir = ensure_result_directory(agent_type)
    
    seed_data = {
        "episode_returns": episode_returns,
        "eval_returns": eval_returns,
        "seed": seed
    }
    
    file_path = os.path.join(result_dir, f"{agent_type}_seed_{seed}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(seed_data, f)
    
    return file_path

def load_seed_data(agent_type, seed):
    result_dir = ensure_result_directory(agent_type)
    file_path = os.path.join(result_dir, f"{agent_type}_seed_{seed}.pkl")
    
    with open(file_path, 'rb') as f:
        seed_data = pickle.load(f)
    
    return seed_data

def plot_episode_returns(all_returns, agent_type):
    result_dir = ensure_result_directory(agent_type)
    
    plt.figure(figsize=(12, 8))
    
    min_length = min(len(returns) for returns in all_returns)
    all_returns = [returns[:min_length] for returns in all_returns]
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    
    episodes = np.arange(1, min_length + 1)
    
    for i, seed_returns in enumerate(all_returns):
        plt.plot(episodes, seed_returns, color='red', alpha=0.2, label=f"Seed {i}" if i == 0 else "")
    
    plt.plot(episodes, mean_returns, color='blue', linewidth=2, label="Mean Returns")
    
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Returns", fontsize=14)
    plt.title(f"{agent_type} Agent Episode Returns", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, f"{agent_type}_episode_returns.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_eval_returns(all_returns, agent_type, eval_interval):
    result_dir = ensure_result_directory(agent_type)
    
    plt.figure(figsize=(12, 8))
    
    min_length = min(len(returns) for returns in all_returns)
    all_returns = [returns[:min_length] for returns in all_returns]
    returns_array = np.array(all_returns)
    mean_returns = np.mean(returns_array, axis=0)
    
    steps = np.arange(1, min_length + 1) * eval_interval
    
    for i, seed_returns in enumerate(all_returns):
        plt.plot(steps, seed_returns[:min_length], color='red', alpha=0.2, label=f"Seed {i}" if i == 0 else "")
    
    plt.plot(steps, mean_returns, color='blue', linewidth=2, label="Mean Returns")
    
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Evaluation Returns", fontsize=14)
    plt.title(f"{agent_type} Agent Evaluation Performance", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, f"{agent_type}_eval_returns.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_from_saved_data(agent_type, seeds, plot_type="episode", eval_interval=None):
    all_returns = []
    
    for seed in seeds:
        seed_data = load_seed_data(agent_type, seed)
        
        if plot_type == "episode":
            all_returns.append(seed_data["episode_returns"])
        elif plot_type == "eval":
            all_returns.append(seed_data["eval_returns"])
    
    if plot_type == "episode":
        return plot_episode_returns(all_returns, agent_type)
    elif plot_type == "eval":
        if eval_interval is None:
            raise ValueError("eval_interval must be provided for eval returns plot")
        return plot_eval_returns(all_returns, agent_type, eval_interval)