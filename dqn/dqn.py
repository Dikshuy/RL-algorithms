from gymnasium import ActionWrapper
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class QNet(nn.Module):
    def __init__(self, n_state, n_action):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value
    
class DQN:
    def __init__(self, env, state_dim, action_dim, memory_capacity, batch_size, lr, gamma, target_update_interval):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eval_net, self.target_net = QNet(state_dim, action_dim), QNet(state_dim, action_dim)

        self.memory = [None] * memory_capacity
        self.memory_counter = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.update_count = 0
        self.target_update_interval = target_update_interval

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, epsilon):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_value = self.eval_net(state)
        action_max_value, index = torch.max(action_value, 1)
        action = index.item()
        if np.random.rand(1) >= epsilon:
            action = np.random.choice(range(self.action_dim), 1).item()
        return action
    
    def store_transition(self, transition):
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = transition
        self.memory_counter += 1
        return self.memory_counter >= self.memory_capacity

    def learn(self):
        if self.memory_counter >= self.memory_capacity:
            state = torch.tensor([t.state for t in self.memory if t is not None], dtype=torch.float)
            action = torch.tensor([t.action for t in self.memory if t is not None], dtype=torch.long).view(-1, 1)
            reward = torch.tensor([t.reward for t in self.memory if t is not None], dtype=torch.float).view(-1, 1)
            next_state = torch.tensor([t.next_state for t in self.memory if t is not None], dtype=torch.float)

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            
            with torch.no_grad():
                q_target = reward + self.gamma * self.target_net(next_state).max(1)[0]

            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), self.batch_size, False):
                q_eval = self.eval_net(state).gather(1, action)[index]
                loss = self.loss_func(q_target[index].unsqueeze(1), q_eval)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.update_count % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.eval_net.state_dict())