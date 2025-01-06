import gymnasium as gym
from actor_critic import agent

env=gym.make('CartPole-v1')
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

alpha=0.0003
gamma=0.99
device='cpu'

ac_agent = agent(n_state, n_action, alpha, gamma, device)

for episode in range(10000):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = ac_agent.choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        transition = (obs, reward, action, next_obs, done)
        td_error = ac_agent.critic_learn(transition)
        ac_agent.actor_learn(obs, action, td_error)
        obs = next_obs

    if episode % 100 == 0:
        print('--------', 'episode: ', episode, 'R: ', total_reward,'--------')