import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
from datetime import datetime

import gymnasium as gym
from sac import SAC
import torch
from torch.utils.tensorboard import SummaryWriter

def evaluate_policy(env, agent: SAC, device, turns = 3):
	total_scores = 0
	for _ in range(turns):
		obs, _ = env.reset()
		done = False
		while not done:
			action = agent.greedy_action(obs)
			obs_next, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated

			total_scores += reward
			obs = obs_next
               
	return int(total_scores/turns)

def train_sac(seed):
    env = gym.make('Pendulum-v1')
    eval_env = gym.make('Pendulum-v1')

    env_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    write = True

    if write:
        time_now = str(datetime.now())[0:-10]
        time_now = ' ' + time_now[0:13] + '_' + time_now[-2::]
        writepath = 'runs/SAC_pendulum' + time_now
        if os.path.exists(writepath):   shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    lr = 0.0003
    gamma = 0.99
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    capacity = 100000
    gradient_steps = 1
    batch_size = 256
    tau = 0.005
    max_steps = 60000
    log_interval = 10000
    eval_interval = 2000
    random_steps = 10000

    save = True
    load = False
    model_index = 5

    if not os.path.exists('model'): os.mkdir('model')

    agent = SAC(env, n_state, n_action, lr, gamma, capacity, gradient_steps, batch_size, tau, device)

    if load: agent.load_model(model_index)
    returns = []

    total_steps = 0
    
    while total_steps < max_steps:
        obs, _ = env.reset(seed=env_seed)
        env_seed += 1
        done = False
        episode_reward = 0
        
        while not done:
            if total_steps < random_steps:    action = env.action_space.sample()
            else:   action = agent.choose_action(obs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_state, done)

            obs = next_state
            episode_reward += reward
            
            if total_steps >= random_steps:
                agent.update()

            if total_steps % eval_interval==0:
                exp_return = evaluate_policy(eval_env, agent, device, turns=5)
                if write:
                    writer.add_scalar('ep_r', exp_return, global_step=total_steps)
                print('seed:', seed, '| steps: {}'.format(total_steps), '| reward:', int(exp_return))

            if save and total_steps % log_interval == 0:
                agent.save_model()

            total_steps += 1
            
        returns.append(episode_reward)

    env.close()
    eval_env.close()
            
    return returns

if __name__ == "__main__":
    num_seeds = 1
    seeds = [i for i in range(num_seeds)]
    
    for seed in seeds:
        returns = train_sac(seed)