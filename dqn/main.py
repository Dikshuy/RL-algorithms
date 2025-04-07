import numpy as np
import random
from plot import *

import gym
import torch
from dqn import DQN
from ddqn import DDQN


def evaluate_policy(env, agent, eval_episodes=5):
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

def train_dqn(seed, eval_interval=100):
    env = gym.make('CartPole-v1')
    env_eval = gym.make('CartPole-v1')

    env_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    buffer_size = 10000
    batch_size = 128
    lr = 3e-4
    gamma = 0.99
    tau = 1e-3
    eps_start = 1.0
    eps_end = 0.001
    eps_decay_rate = 0.99
    num_episodes = 2000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = DQN(state_dim, action_dim, buffer_size, batch_size, lr, gamma, tau, device)
    # agent = DDQN(state_dim, action_dim, buffer_size, batch_size, lr, gamma, tau, device)

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
    num_seeds = 5               # number of seeds
    eval_interval = 100         # evaluation every 100 steps
    seeds = [i for i in range(num_seeds)]
    all_episode_returns = []
    all_eval_returns = []
    
    for seed in seeds:
        print(f"seed: {seed}")
        episode_returns, eval_returns = train_dqn(seed, eval_interval)
        all_episode_returns.append(episode_returns)
        all_eval_returns.append(eval_returns)

    plot_episode_returns(all_episode_returns)
    plot_eval_returns(all_eval_returns, eval_interval)
