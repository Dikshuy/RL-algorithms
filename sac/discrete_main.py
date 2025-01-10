import numpy as np
import gymnasium as gym
from sac_discrete import DiscreteSAC
import torch
import matplotlib.pyplot as plt
from scipy import stats

def train_sac(seed):
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_action = env.action_space.n
    n_state = env.observation_space.shape[0]
    alpha = 0.0003
    gamma = 0.99
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    capacity = 10000
    gradient_steps = 1
    batch_size = 128
    tau = 0.005
    iterations = 100000
    log_interval = 2000
    
    sac_agent = DiscreteSAC(env, n_state, n_action, alpha, gamma, capacity, gradient_steps, batch_size, tau, device)
    returns = []
    
    for i in range(iterations):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = sac_agent.choose_action(obs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            sac_agent.store_transition(obs, action, reward, next_state, done)
            
            if sac_agent.replay_buffer.pos >= sac_agent.batch_size:
                sac_agent.update()
                
            obs = next_state
            episode_reward += reward
            
        returns.append(episode_reward)
        
        if i % log_interval == 0:
            sac_agent.save_model()
            
        print(f"Seed {seed}, Episode {i}, Reward: {episode_reward}")
            
    return returns

def plot_returns(all_returns, seeds, window=100):
    plt.figure(figsize=(10, 6))

    returns_array = np.array([all_returns[seed] for seed in seeds])
    mean_returns = np.mean(returns_array, axis=0)

    n = len(seeds)
    std_error = stats.sem(returns_array, axis=0)
    ci_95 = std_error * stats.t.ppf((1 + 0.95) / 2, n-1)
    
    smoothed_mean = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
    smoothed_ci = np.convolve(ci_95, np.ones(window)/window, mode='valid')
    episodes = range(len(smoothed_mean))
    
    plt.plot(episodes, smoothed_mean, 'b-', label='Mean', linewidth=2)
    plt.fill_between(episodes, 
                     smoothed_mean - smoothed_ci, 
                     smoothed_mean + smoothed_ci, 
                     alpha=0.2, color='b',
                     label='95% CI')
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('returns for cartpole using SAC')
    plt.legend()
    plt.grid(True)
    plt.savefig('discrete_sac_returns.png')
    plt.show()

if __name__ == "__main__":
    num_seeds = 10
    seeds = [i for i in range(num_seeds)]
    all_returns = {}
    
    for seed in seeds:
        returns = train_sac(seed)
        all_returns[seed] = returns
        
    plot_returns(all_returns, seeds)