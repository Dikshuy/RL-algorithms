import numpy as np
import gymnasium as gym
from reinforce import agent

env=gym.make('CartPole-v1')
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

alpha=0.0003
gamma=0.99
device='cpu'

reinforce_agent = agent(n_state, n_action, alpha, gamma, device)

for episode in range(10000):
    t = 0
    transitions = []
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done and t < 300:
        action = reinforce_agent.choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        transitions.append((obs, action, reward))        

        obs = next_obs
        t = t + 1

    if transitions:
        reinforce_agent.learn(transitions)

    if episode % 100 == 0:
        print('--------', 'episode: ', episode, 'R: ', total_reward,'--------')