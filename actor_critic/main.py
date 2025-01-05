import gymnasium as gym
import numpy as np
from actor_critic import Agent, NewAgent

def test_agent(agent_class, env_name='Pendulum-v1', episodes=5000):
    env = gym.make(env_name)
    
    if agent_class == Agent:
        agent = Agent(alpha=0.0003, beta=0.0003,
                     input_dims=[env.observation_space.shape[0]],
                     n_actions=env.action_space.shape[0])
    else:
        agent = NewAgent(alpha=0.0003,
                        input_dims=[env.observation_space.shape[0]],
                        n_actions=env.action_space.shape[0])
    
    scores = []
    
    for episode in range(episodes):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            score += reward
            agent.learn(observation, reward, new_observation, done)
            observation = new_observation
        
        scores.append(score)
        print(f'Episode: {episode}, Score: {score}')
    
    env.close()
    return np.mean(scores)

if __name__ == '__main__':
    # Test both agent implementations
    print("Testing original Agent:")
    mean_score_agent = test_agent(Agent)
    print(f"Mean score for Agent: {mean_score_agent}\n")
    
    print("Testing NewAgent:")
    mean_score_new_agent = test_agent(NewAgent)
    print(f"Mean score for NewAgent: {mean_score_new_agent}")
