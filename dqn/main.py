import numpy as np
import random
from plot import *
import argparse

import gym
import torch
from dqn import DQN
from ddqn import DDQN
from d3qn import D3QN


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

def train_dqn(agent_type, seed, eval_interval=500):
    env = gym.make('CartPole-v1')
    env_eval = gym.make('CartPole-v1')

    env_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    state_dim = env.observation_space.shape[0]      # state dimension
    action_dim = env.action_space.n                 # action dimension  
    buffer_size = 10000                             # replay buffer size
    batch_size = 128                                # batch size
    lr = 3e-4                                       # learning rate
    optimizer_eps = 1e-5                            # optimizer epsilon
    gamma = 0.99                                    # discount factor
    n_step = 1                                      # n-step return
    tau = 1e-3                                      # soft update parameter
    eps_start = 1.0                                 # initial epsilon
    eps_end = 0.001                                 # final epsilon
    eps_decay_rate = 0.99                           # decay rate
    num_episodes = 1000                             # number of episodes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if agent_type == 'dqn':
        agent = DQN(state_dim, action_dim, buffer_size, batch_size, lr, optimizer_eps, gamma, n_step, tau, device)
    elif agent_type == 'ddqn':
        agent = DDQN(state_dim, action_dim, buffer_size, batch_size, lr, optimizer_eps, gamma, n_step, tau, device)
    elif agent_type == 'd3qn':
        agent = D3QN(state_dim, action_dim, buffer_size, batch_size, lr, optimizer_eps, gamma, n_step, tau, device)

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
            
            if len(agent.memory) >= batch_size:
                experiences = agent.memory.sample()
                agent.learn(experiences)

            if total_steps % eval_interval == 0:
                eval_reward = evaluate_policy(env_eval, agent)
                eval_returns.append(eval_reward)
                # print(f"Evaluation at step {total_steps}: {eval_reward}")

        episode_returns.append(episodic_reward)
        print(f"Episode {i+1}/{num_episodes}, Reward: {episodic_reward}")

    env.close()
    env_eval.close()

    return episode_returns, eval_returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agents on CartPole')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'ddqn', 'd3qn'], help='Agent type: dqn, ddqn, or d3qn')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to run')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval in steps')

    args = parser.parse_args()

    seeds = [i for i in range(args.seeds)]
    
    all_episode_returns = []
    all_eval_returns = []
    
    for seed in seeds:
        print(f"seed: {seed}")
        episode_returns, eval_returns = train_dqn(args.agent, seed, args.eval_interval)
        all_episode_returns.append(episode_returns)
        all_eval_returns.append(eval_returns)

    plot_episode_returns(all_episode_returns, args.agent)
    # plot_eval_returns(all_eval_returns, args.agent, args.eval_interval)
