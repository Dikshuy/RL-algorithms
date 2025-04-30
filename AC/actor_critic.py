import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class actor(nn.Module):
    def __init__(self, n_state, n_action):
        super(actor, self).__init__()
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
    
class critic(nn.Module):
    def __init__(self, n_state):
        super(critic, self).__init__()
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
        self.actor = actor(state_dim, n_action).to(device)
        self.critic = critic(state_dim).to(device)
        self.gamma = gamma
        self.device = device

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=alpha)

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device) 
        prob = self.actor(state)
        dist = T.distributions.Categorical(prob)
        action = dist.sample()
        return action.item()
    
    def actor_learn(self, state, action, td):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device) 
        prob = self.actor(state)
        prob = prob[0, action]
        self.actor_optim.zero_grad()
        loss = -(T.log(prob) * td.detach())
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, transition):
        state, reward, _, next_state, done = transition
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device) 
        next_state = T.tensor(next_state, dtype=T.float).unsqueeze(0).to(self.device) 
        reward = T.tensor([reward], dtype=T.float).to(self.device)
        done = T.tensor([done], dtype=T.float).to(self.device)

        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_error = td_target - self.critic(state)

        self.critic_optim.zero_grad()
        loss = td_error**2
        loss.backward()
        self.critic_optim.step()

        return td_error