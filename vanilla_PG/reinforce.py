import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class policy_net(nn.Module):
    def __init__(self, n_state, n_action):
        super(policy_net, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_action)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = self.softmax(x)
        return prob
    
class value_net(nn.Module):
    def __init__(self, n_state):
        super(value_net, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class agent():
    def __init__(self, state_dim, n_action, alpha=0.0003, gamma=0.99, device='cpu'):
        self.policy = policy_net(state_dim, n_action).to(device)
        self.value = value_net(state_dim).to(device)
        self.gamma = gamma
        self.device = device

        self.G = 0

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=alpha)
        self.value_optim = optim.Adam(self.value.parameters(), lr=alpha)

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device) 
        prob = self.policy(state)
        dist = T.distributions.Categorical(prob)
        action = dist.sample()
        return action.item()
    
    def learn(self, transitions):
        timestep = len(transitions)
        returns = T.zeros(timestep, 1).to(self.device)
        log_probs = T.zeros(timestep, 1).to(self.device)
        self.G = 0

        states = T.tensor(np.array([t[0] for t in transitions]), dtype=T.float).to(self.device)
        actions = T.tensor(np.array([t[1] for t in transitions]), dtype=T.long).to(self.device)
        rewards = T.tensor(np.array([t[2] for t in transitions]), dtype=T.float).to(self.device)

        values = self.value(states)

        for i in reversed(range(timestep)):
            self.G = rewards[i] + self.gamma * self.G
            returns[i] = self.G
            
            probs = self.policy(states[i].unsqueeze(0))
            log_probs[i] = T.log(probs.squeeze(0)[actions[i]]).unsqueeze(0)

        policy_loss = -(log_probs * (returns - values.detach())).mean()
        value_loss = F.mse_loss(values, returns)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()