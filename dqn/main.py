import numpy as np
from collections import deque
import gym
import torch
import random
from dqn import DQN
from ddqn import DDQN


def evaluate_policy(env, agent, turns = 3):
	total_scores = 0
	for _ in range(turns):
		obs, _ = env.reset()
		done = False
		while not done:
			action = agent.choose_action(obs, epsilon=0.0)
			obs_next, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated

			total_scores += reward
			obs = obs_next
               
	return int(total_scores/turns)

def train_dqn(seed):
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
    eps_end = 0.1
    eps_decay_rate = 0.99
    num_episodes = 500
    eval_interval = 100
    buffer = deque(maxlen=buffer_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = DQN(state_dim, action_dim, buffer_size, batch_size, lr, gamma, tau, device)
    # agent = DDQN(state_dim, action_dim, buffer_size, batch_size, lr, gamma, tau, device)

    returns = []

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
            
            buffer.append((obs, action, reward, next_obs, done))
            
            obs = next_obs
            episodic_reward += reward
            
            if len(buffer) >= batch_size:
                experiences = random.sample(buffer, batch_size)
                agent.learn(experiences, gamma)

            if total_steps % eval_interval == 0:
                eval_reward = evaluate_policy(env_eval, agent)
                print(f"Step: {total_steps}, Eval reward: {eval_reward}")

            total_steps += 1

        returns.append(episodic_reward)

    env.close()
    env_eval.close()

    return returns

if __name__ == "__main__":
    num_seeds = 2
    seeds = [i for i in range(num_seeds)]
    
    for seed in seeds:
        returns = train_dqn(seed)
