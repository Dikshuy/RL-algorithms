import os
import numpy as np
import random
from collections import deque, namedtuple
import gymnasium as gym

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

        states = T.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = T.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = T.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
class QNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_action)
        self.reset_params()

    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_vals = self.fc3(x)

        return q_vals
    
class PolicyNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_action)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))

        return action_probs
    
class DiscreteSAC():
    def __init__(self, env, state_dim, n_action, lr=0.0003, gamma=0.99, capacity=10000, gradient_steps=1, batch_size=64, tau=0.005, device='cpu', target_entropy_scale = 0.89):
        super(DiscreteSAC, self).__init__()

        self.policy = PolicyNetwork(state_dim, n_action).to(device)
        self.q1 = QNetwork(state_dim, n_action).to(device)
        self.q2 = QNetwork(state_dim, n_action).to(device)
        self.target_q1 = QNetwork(state_dim, n_action).to(device)
        self.target_q2 = QNetwork(state_dim, n_action).to(device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=lr, eps=1e-4)
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr, eps=1e-4)

        # entropy tuning
        self.target_entropy = -target_entropy_scale * T.log(1/T.tensor(n_action))
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

        self.gamma = gamma
        self.device = device
        self.n_action = n_action

        self.replay_buffer = ReplayBuffer(capacity, batch_size, device=device)

        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.tau = tau
        self.num_training = 1

        os.makedirs('./discrete_model/', exist_ok=True)

    def choose_action(self, state):
        action_probs = self.policy(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample().cpu()
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_probs = T.log(action_probs + z)
        action = action.detach().cpu().numpy()
        return action, action_probs, log_probs
    
    def greedy_action(self, state):
        action_probs = self.policy(state)
        action = action_probs.argmax(-1).item()      
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward,next_state, done)
    
    def update(self):        
        data = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones = data 

        for _ in range(self.gradient_steps):
            with T.no_grad():
                _, next_action_probs, next_log_probs = self.choose_action(next_states)
                next_q1_target = self.target_q1(next_states)
                next_q2_target = self.target_q2(next_states)
                next_q_target = T.min(next_q1_target, next_q2_target)

                min_q_next_target = (next_action_probs * (next_q_target - self.alpha * next_log_probs)).sum(dim=1).unsqueeze(-1)
                next_q = rewards + self.gamma * (1 - dones) * (min_q_next_target)

            current_q1 = self.q1(states).gather(1, actions.long())
            current_q2 = self.q2(states).gather(1, actions.long())

            q1_loss = F.mse_loss(current_q1, next_q)
            q2_loss = F.mse_loss(current_q2, next_q)
            q_loss = q1_loss + q2_loss

            self.q_optim.zero_grad()
            q_loss.backward(retain_graph=True)
            self.q_optim.step()

            _, action_probs, log_probs = self.choose_action(states)

            with T.no_grad():
                min_q = T.min(self.q1(states), self.q2(states))

            policy_loss = (action_probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_probs + self.target_entropy).detach())).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

            for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.num_training += 1

    def save_model(self):
        T.save(self.policy.state_dict(), './discrete_model/policy.pth')
        T.save(self.q1.state_dict(), './discrete_model/q1.pth')
        T.save(self.q2.state_dict(), './discrete_model/q2.pth')
        T.save(self.target_q1.state_dict(), './discrete_model/target_q1.pth')
        T.save(self.target_q2.state_dict(), './discrete_model/target_q2.pth')
        print("model saved ..")

    def load_model(self):
        self.policy.load_state_dict(T.load('./discrete_model/policy.pth'))
        self.q1.load_state_dict(T.load('./discrete_model/q1.pth'))
        self.q2.load_state_dict(T.load('./discrete_model/q2.pth'))
        self.target_q1.load_state_dict(T.load('./discrete_model/target_q1.pth'))
        self.target_q2.load_state_dict(T.load('./discrete_model/target_q2.pth'))
        print("model loaded..")