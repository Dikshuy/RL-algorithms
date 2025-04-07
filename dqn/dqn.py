import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    
class QNet(nn.Module):
    def __init__(self, n_state, n_action, device):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_action)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value

class DQN:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, lr, gamma, tau, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eval_net = QNet(state_dim, action_dim, device).to(device)
        self.target_net =  QNet(state_dim, action_dim, device).to(device)

        self.memory = ReplayBuffer(buffer_size, batch_size, device)
        self.memory_counter = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        
    def choose_action(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.eval_net.eval()
        with torch.no_grad():
            action_value = self.eval_net(state)
        self.eval_net.train()

        if np.random.random() > epsilon:
            action = action_value.argmax(1).item()
        else:
            action = np.random.choice(self.action_dim)
        
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # updates for dqn
        q = self.eval_net(states).gather(1, actions)
        q_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_func(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(target_param.data * (1-self.tau) + param.data * self.tau)