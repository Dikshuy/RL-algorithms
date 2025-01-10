import os
import numpy as np
import gymnasium as gym

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer
    
class QNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_action)

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

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)

        return action_probs
    
class DiscreteSAC():
    def __init__(self, env, state_dim, n_action, alpha=0.0003, gamma=0.99, capacity=10000, gradient_steps=1, batch_size=64, tau=0.005, device='cpu', target_entropy_scale = 0.89, target_entropy=None):
        super(DiscreteSAC, self).__init__()

        self.policy = PolicyNetwork(state_dim, n_action).to(device)
        self.q1 = QNetwork(state_dim, n_action).to(device)
        self.q2 = QNetwork(state_dim, n_action).to(device)
        self.target_q1 = QNetwork(state_dim, n_action).to(device)
        self.target_q2 = QNetwork(state_dim, n_action).to(device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=alpha, eps=1e-4)
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=alpha, eps=1e-4)

        # entrop tuning
        self.target_entropy = - target_entropy_scale * T.log(1/T.tensor(n_action))
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha, eps=1e-4)

        self.gamma = gamma
        self.device = device
        self.n_action = n_action

        self.replay_buffer = ReplayBuffer(capacity, env.observation_space, env.action_space, device=device, n_envs=1)

        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.tau = tau
        self.num_training = 1

        os.makedirs('./discrete_model/', exist_ok=True)

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.device)
        with T.no_grad():
            action_probs = self.policy(state)
            action = T.distributions.Categorical(action_probs).sample().item()
        return action
    
    def store_transition(self, obs, action, reward, next_obs, done):
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        action = np.array(action)
        
        self.replay_buffer.add(
            obs,
            next_obs,
            action,
            reward,
            done,
            [{}]
        )
    
    def update(self):
        if self.num_training % 500 == 0:
            print("Training .. {} times".format(self.num_training))

        if self.replay_buffer.pos < self.batch_size:
            return
        
        data = self.replay_buffer.sample(self.batch_size)
        
        states = data.observations
        next_states = data.next_observations
        actions = data.actions
        rewards = data.rewards.reshape(-1, 1)
        dones = data.dones.reshape(-1, 1)
        
        # Convert to tensors
        if not isinstance(states, T.Tensor):
            states = T.FloatTensor(states).to(self.device)
            actions = T.FloatTensor(actions).reshape(-1, self.n_action).to(self.device)
            rewards = T.FloatTensor(rewards).reshape(-1, 1).to(self.device)
            next_states = T.FloatTensor(next_states).to(self.device)
            dones = T.FloatTensor(dones).reshape(-1, 1).to(self.device)

        for _ in range(self.gradient_steps):
            with T.no_grad():
                next_action_probs = self.policy(next_states)
                next_log_probs = T.log(next_action_probs + 1e-10)
                next_q1 = self.target_q1(next_states)
                next_q2= self.target_q2(next_states)
                next_q = T.min(next_q1, next_q2)

                target_value = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
                target_q = rewards + self.gamma * (1 - dones) * target_value

            current_q1 = self.q1(states).gather(1, actions)
            current_q2 = self.q2(states).gather(1, actions)

            q1_loss = F.mse_loss(current_q1, target_q.detach())
            q2_loss = F.mse_loss(current_q2, target_q.detach())
            q_loss = q1_loss + q2_loss

            # update single optim not q1 and q2

            self.q_optim.zero_grad()
            q_loss.backward(retain_graph=True)
            self.q_optim.step()

            action_probs = self.policy(states)
            log_probs = T.log(action_probs + 1e-10)
            q_values = T.min(self.q1(states), self.q2(states))

            policy_loss = (action_probs * (self.alpha * log_probs - q_values)).sum(dim=1).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).mean())

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

            for target_param, param in zip(self.q1.parameters(), self.target_q1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.q2.parameters(), self.target_q2.parameters()):
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