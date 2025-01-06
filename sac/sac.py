import os
import numpy as np
from collections import namedtuple
import gymnasium as gym

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])
    
class QNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(QNetwork, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.fc1 = nn.Linear(n_state+n_action, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        s = state.reshape(-1, self.n_state)
        a = action.reshape(-1, self.n_action)
        x = T.cat((s, a), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class ValueNetwork(nn.Module):
    def __init__(self, n_state):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value
    
class PolicyNetwork(nn.Module):
    def __init__(self, n_state, n_action, min_log_std=-20, max_log_std=2):
        super(PolicyNetwork, self).__init__()
        self.n_state = n_state

        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, n_action)
        self.log_std_head = nn.Linear(256, n_action)

        self.max_log_std = max_log_std
        self.min_log_std = min_log_std

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = T.clamp(log_std_head, self.min_log_std, self.max_log_std)

        return mu, log_std_head
    
class SAC():
    def __init__(self, state_dim, n_action, alpha=0.0003, gamma=0.99, capacity=10000, gradient_steps=1, batch_size=64, tau=0.005, device='cpu'):
        super(SAC, self).__init__()

        self.policy = PolicyNetwork(state_dim, n_action).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        self.target_value = ValueNetwork(state_dim).to(device)
        self.q1 = QNetwork(state_dim, n_action).to(device)
        self.q2 = QNetwork(state_dim, n_action).to(device)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=alpha)
        self.value_optim = optim.Adam(self.value.parameters(), lr=alpha)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=alpha)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=alpha)

        self.gamma = gamma
        self.device = device
        self.n_action = n_action

        self.value_criterion = nn.MSELoss()
        self.q1_criterion = nn.MSELoss()
        self.q2_criterion = nn.MSELoss()

        self.capacity = capacity
        self.replay_buffer = []
        self.num_transitions = 0
        self.num_training = 1

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(param.data)

        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.tau = tau

        os.makedirs('./model/', exist_ok=True)

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device)
        mu, log_std = self.policy(state)
        std = T.exp(log_std)
        dist = T.distributions.Normal(mu, std)
        z = dist.sample()
        action = T.tanh(z).detach().cpu().numpy().flatten()
        return action
    
    def store_transition(self, s, a, r, s_, d):
        transition = Transition(s, a, r, s_, d)
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[self.num_transitions % self.capacity] = transition
        self.num_transitions += 1

    def evaluate(self, state):
        batch_mu, batch_std = self.policy(state)
        batch_std = T.exp(batch_std)
        dist = T.distributions.Normal(batch_mu, batch_std)
        
        z = T.distributions.Normal(0, 1).sample()
        action = T.tanh(batch_mu + batch_std*z.to(self.device))
        log_prob = dist.log_prob(batch_mu + batch_std*z.to(self.device)) - T.log(1 - action**2 + 1e-6)

        return action, log_prob
    
    def update(self):
        if self.num_training % 500 == 0:
            print("Training .. {} times".format(self.num_training))

        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[idx] for idx in batch_indices]
        
        # Convert to numpy arrays first
        states = np.array([t.s for t in batch])
        actions = np.array([t.a for t in batch])
        rewards = np.array([t.r for t in batch])
        next_states = np.array([t.s_ for t in batch])
        dones = np.array([t.d for t in batch])
        
        # Convert to tensors
        states = T.FloatTensor(states).to(self.device)
        actions = T.FloatTensor(actions).reshape(-1, self.n_action).to(self.device)
        rewards = T.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        next_states = T.FloatTensor(next_states).to(self.device)
        dones = T.FloatTensor(dones).reshape(-1, 1).to(self.device)

        for _ in range(self.gradient_steps):
            with T.no_grad():
                target_value = self.target_value(next_states)
                q_next = rewards + self.gamma * (1 - dones) * target_value

            expected_value = self.value(states)
            expected_q1 = self.q1(states, actions)
            expected_q2 = self.q2(states, actions)

            q1_loss = self.q1_criterion(expected_q1, q_next.detach()).mean()
            q2_loss = self.q2_criterion(expected_q2, q_next.detach()).mean()

            self.q1_optim.zero_grad()
            q1_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.q1.parameters(), 0.5)
            self.q1_optim.step()

            self.q2_optim.zero_grad()
            q2_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.q2.parameters(), 0.5)
            self.q2_optim.step()

            sample_action, log_prob = self.evaluate(states)
            expected_new_q = T.min(self.q1(states, sample_action), self.q2(states, sample_action))
            next_value = expected_new_q - log_prob

            value_loss = self.value_criterion(expected_value, next_value.detach()).mean()

            self.value_optim.zero_grad()
            value_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optim.step()

            pi_loss = (log_prob - expected_new_q).mean()

            self.policy_optim.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optim.step()

            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.num_training += 1

    def save_model(self):
        T.save(self.policy.state_dict(), './model/policy.pth')
        T.save(self.value.state_dict(), './model/value.pth')
        T.save(self.target_value.state_dict(), './model/target_value.pth')
        T.save(self.q1.state_dict(), './model/q1.pth')
        T.save(self.q2.state_dict(), './model/q2.pth')
        print("Model saved ..")

    def load_model(self):
        self.policy.load_state_dict(T.load('./model/policy.pth'))
        self.value.load_state_dict(T.load('./model/value.pth'))
        self.target_value.load_state_dict(T.load('./model/target_value.pth'))
        self.q1.load_state_dict(T.load('./model/q1.pth'))
        self.q2.load_state_dict(T.load('./model/q2.pth'))
        print("Model loaded ..")