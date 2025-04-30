import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
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

class DDQN:
    def __init__(self, state_dim, action_dim, buffer, lr, optimizer_eps, gamma, tau, target_update_freq, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = buffer
        self.gamma = gamma
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.device = device

        self.eval_net = QNet(state_dim, action_dim, device).to(device)
        self.target_net =  QNet(state_dim, action_dim, device).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

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

    def learn(self):
        batch = self.memory.sample()
        states = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_observations
        dones = batch.dones
        
        # default weights for standard and n-step buffer
        weights = torch.ones_like(rewards)
        indices = None

        # in case of PER buffer
        if hasattr(batch, 'weights') and hasattr(batch, 'indices'):
            weights = batch.weights
            indices = batch.indices

        # updates for double dqn
        q_values = self.eval_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.eval_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            q_targets = rewards + (1 - dones.float()) * self.gamma * next_q_values

        td_errors = q_targets - q_values
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities for prioritized replay buffer
        if hasattr(batch, 'indices') and indices is not None:
            new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, new_priorities)

        # hard update - traditionally dqn performs hard updates
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # soft update - slowly changing target network alternative
        # for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
        #     target_param.data.copy_(target_param.data * (1-self.tau) + param.data * self.tau)

        return loss.item()