import numpy as np
from collections import namedtuple
import gym

from dqn import DQN

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
memory_capacity = 10000
batch_size = 256
lr = 3e-4
gamma = 0.99
target_update_interval = 100
num_episodes = 2000

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

agent = DQN(env, state_dim, action_dim, memory_capacity, batch_size, lr, gamma, target_update_interval)

for i in range(num_episodes):
    state, _ = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(state, 0.9)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(Transition(state, action, reward, next_state))
        agent.learn()
        state = next_state
        score += reward

    print(f'Episode {i}, score: {score}')

env.close()
