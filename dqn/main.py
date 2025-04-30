import numpy as np
import random
from plot import *
import argparse

import gymnasium as gym
import torch
from dqn import DQN
from ddqn import DDQN
from d3qn import D3QN
from buffer import ReplayBuffer, NStepReplayBuffer, PrioritizedReplayBuffer

def create_buffer(buffer_type, state_dim, buffer_size, batch_size, device, n_step, gamma, alpha, beta):
    if buffer_type == 'standard':
        return ReplayBuffer(
            capacity=buffer_size,
            obs_shape=(state_dim,),
            device=device,
            batch_size=batch_size,
            gamma=gamma
        )
    elif buffer_type == 'n_step':
        return NStepReplayBuffer(
            capacity=buffer_size,
            obs_shape=(state_dim,),
            device=device,
            batch_size=batch_size,
            n_step=n_step,
            gamma=gamma
        )
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(
            capacity=buffer_size,
            obs_shape=(state_dim,),
            device=device,
            batch_size=batch_size,
            n_step=n_step,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            beta_increment=0.0001, # yet to decide this annealing schedule
        )
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")
    
def create_agent(agent_type, state_dim, action_dim, buffer, lr, optimizer_eps, gamma, tau, target_update_freq, device):
    if agent_type == 'dqn':
        return DQN(state_dim, action_dim, buffer, lr, optimizer_eps, gamma, tau, target_update_freq, device)
    elif agent_type == 'ddqn':
        return DDQN(state_dim, action_dim, buffer, lr, optimizer_eps, gamma, tau, target_update_freq, device)
    elif agent_type == 'd3qn':
        return D3QN(state_dim, action_dim, buffer, lr, optimizer_eps, gamma, tau, target_update_freq, device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
def evaluate_policy(env, agent, eval_episodes=3):
	total_scores = 0
	for _ in range(eval_episodes):
		obs, _ = env.reset()
		done = False
		while not done:
			action = agent.choose_action(obs, epsilon=0.0)
			obs_next, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			total_scores += reward
			obs = obs_next     
	return total_scores/eval_episodes

def train_agent(agent_type, seed, eval_interval=500, n_step=1, buffer_type='standard'):
    env = gym.make('CartPole-v1')
    env_eval = gym.make('CartPole-v1')

    env_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    state_dim = env.observation_space.shape[0]      # state dimension
    action_dim = env.action_space.n                 # action dimension  
    buffer_size = 25000                             # replay buffer size
    batch_size = 256                                # batch size
    lr = 1e-4                                       # learning rate
    optimizer_eps = 1e-5                            # optimizer epsilon
    gamma = 0.99                                    # discount factor
    n_step = n_step                                 # n-step return
    tau = 5e-3                                      # soft update parameter (soft update)
    target_update_freq = 100                        # target network update frequency (hard update)
    eps_start = 1.0                                 # initial epsilon
    eps_end = 0.001                                 # final epsilon
    eps_decay_rate = 0.99                           # decay rate
    num_episodes = 1000                             # number of episodes
    alpha = 0.6                                     # prioritization exponent
    beta = 0.4                                      # importance sampling exponent

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    buffer = create_buffer(buffer_type, state_dim, buffer_size, batch_size, device, n_step, gamma, alpha, beta)
    agent = create_agent(agent_type, state_dim, action_dim, buffer, lr, optimizer_eps, gamma, tau, target_update_freq, device)

    eval_returns = []
    episode_returns = []
    total_steps = 0

    for i in range(num_episodes):
        obs, _ = env.reset(seed=env_seed)
        epsilon = max(eps_end, eps_start * (eps_decay_rate ** i))

        env_seed += 1
        done = False
        episodic_reward = 0

        while not done:
            action = agent.choose_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.add(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            episodic_reward += reward
            total_steps += 1
            
            if len(agent.memory) > batch_size:
                agent.learn()

            if total_steps % eval_interval == 0:
                eval_reward = evaluate_policy(env_eval, agent)
                eval_returns.append(eval_reward)
                # print(f"Step {total_steps}: Eval reward: {eval_reward:.2f}")

        episode_returns.append(episodic_reward)
        print(f"Episode {i+1}/{num_episodes}, Reward: {episodic_reward}")

    env.close()
    env_eval.close()

    return episode_returns, eval_returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agents on CartPole')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'ddqn', 'd3qn'], help='Agent type: dqn, ddqn, or d3qn')
    parser.add_argument('--n_step', type=int, default=1, help='Number of steps for n-step return')
    parser.add_argument('--buffer', type=str, default='standard', choices=['standard', 'n_step', 'prioritized'], help='Buffer type: standard, n_step, or prioritized')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to run')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval in steps')

    args = parser.parse_args()

    seed_list = list(range(args.seeds))

    all_episode_returns = []
    all_eval_returns = []
    
    for seed in seed_list:
        print(f"Training with seed: {seed}")
        episode_returns, eval_returns = train_agent(args.agent, seed, args.eval_interval, args.buffer)
        save_seed_data(episode_returns, eval_returns, f"{args.agent}", seed)
        all_episode_returns.append(episode_returns)
        all_eval_returns.append(eval_returns)
    
    plot_episode_returns(all_episode_returns, args.agent)
    plot_eval_returns(all_eval_returns, args.agent, args.eval_interval)
