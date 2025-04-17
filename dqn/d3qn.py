import numpy as np
from buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
class DuelingQNet(nn.Module):
    def __init__(self, n_state, n_action, device):
        super(DuelingQNet, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.fc2 = nn.Linear(128, 128)

        self.value_fc = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(128, n_action)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - 1/|A| * sum(A(s,a))
        action_value = value + advantage - advantage.mean(dim=1, keepdim=True)
        return action_value

class D3QN:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, lr, optimizer_eps, gamma, n_step, tau, target_update_freq, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eval_net = DuelingQNet(state_dim, action_dim, device).to(device)
        self.target_net =  DuelingQNet(state_dim, action_dim, device).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.memory = ReplayBuffer(buffer_size, batch_size, n_step, gamma, device)
        self.batch_size = batch_size
        self.gamma = gamma ** n_step
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.device = device

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr, eps=optimizer_eps)
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

        # updates for dueling double dqn
        q = self.eval_net(states).gather(1, actions)
        max_action_indices = self.eval_net(next_states).argmax(1).unsqueeze(1)
        q_next = self.target_net(next_states).gather(1, max_action_indices)
        q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_func(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # hard update - traditionally dqn performs hard updates
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # soft update - slowly changing target network alternative
        # for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
        #     target_param.data.copy_(target_param.data * (1-self.tau) + param.data * self.tau)