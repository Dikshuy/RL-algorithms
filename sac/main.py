import numpy as np
import gymnasium as gym
from sac import SAC

env=gym.make('Pendulum-v1')
n_action=env.action_space.shape[0]
n_state=env.observation_space.shape[0]

alpha=0.0003
gamma=0.99
device='cpu'
capacity=10000
gradient_steps=1
batch_size=128
tau=0.005

iterations = 100000
log_interval = 2000

load = False

sac_agent = SAC(n_state, n_action, alpha, gamma, capacity, gradient_steps, batch_size, tau, device)

if load:    
    sac_agent.load_models()

for i in range(iterations):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = sac_agent.choose_action(obs)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        sac_agent.store_transition(obs, action, reward, next_state, done)
        if sac_agent.num_transitions >= sac_agent.batch_size:
            sac_agent.update()
        obs = next_state
        episode_reward += reward
    if i % log_interval == 0:
        sac_agent.save_models()

    print(f"Episode {i} reward: {episode_reward}")
