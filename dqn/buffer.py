import random
from collections import namedtuple, deque
import re
import torch
import numpy as np


# replay buffer for n-step return
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, n_step, gamma, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.n_step = n_step
        self.gamma = gamma
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        reward, next_state, done = 0, None, False 
        for idx, experience in enumerate(self.n_step_buffer):
            if idx == 0:
                next_state = experience.next_state 
            reward += experience.reward * (self.gamma ** idx)
            if experience.done:
                done = True
                if idx != 0:
                    next_state = self.n_step_buffer[idx-1].next_state
                break
                
        return reward, next_state, done

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.n_step_buffer.append(e)
        if len(self.n_step_buffer) < self.n_step and not done:
            return
        if self.n_step > 1:
            reward, next_state, done = self._get_n_step_info()
        first_experience = self.n_step_buffer[0]
        e = self.experience(first_experience.state, first_experience.action, reward, next_state, done)
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

# logic from cleanrl 
class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[2 * parent + 1] + self.tree[2 * parent + 2]
            parent = (parent - 1) // 2

    def _update(self, idx, value):
        idx += self.capacity - 1
        self.tree[idx] = value
        self._propagate(idx)

    def total(self):
        return self.tree[0]
    
    def _retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            right = idx * 2 + 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        return idx - (self.capacity - 1)
    
class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.full(self.tree_size, float("inf"), dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(self.tree[2 * parent + 1], self.tree[2 * parent + 2])
            parent = (parent - 1) // 2

    def _update(self, idx, value):
        idx += self.capacity - 1
        self.tree[idx] = value
        self._propagate(idx)

    def min(self):
        return self.tree[0]

# priortized experience replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, device, n_step, gamma, alpha=0.6, beta=0.4, eps=1e-6):
        self.device = device
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.buffer_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.buffer_actions = np.zeros(capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer_next_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.buffer_dones = np.zeros(capacity, dtype=np.bool_)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for i in range(len(self.n_step_buffer)):
            reward += (self.gamma ** i) * self.n_step_buffer[i][2]
            if self.n_step_buffer[i][4]:
                next_obs = self.n_step_buffer[i][3]
                done = True
                break
        return reward, next_obs, done
    
    def add(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:   
            return
        
        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_next_obs[idx] = next_obs
        self.buffer_dones[idx] = done

        priority = self.max_priority ** self.alpha
        self.sum_tree._update(idx, priority)
        self.min_tree._update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        indices = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree._retrieve(upperbound)
            indices.append(idx)

        samples = {
            "observations": torch.from_numpy(self.buffer_obs[indices]).to(self.device),
            "actions": torch.from_numpy(self.buffer_actions[indices]).to(self.device).unsqueeze(1),
            "rewards": torch.from_numpy(self.buffer_rewards[indices]).to(self.device).unsqueeze(1),
            "next_observations": torch.from_numpy(self.buffer_next_obs[indices]).to(self.device),
            "dones": torch.from_numpy(self.buffer_dones[indices]).to(self.device).unsqueeze(1),
        }

        probs = np.array([self.sum_tree.tree[idx + self.capacity - 1] for idx in indices])
        weights = (self.size * probs / self.sum_tree.total()) ** (-self.beta)
        weights = weights / weights.max()
        samples["weights"] = torch.from_numpy(weights).to(self.device).unsqueeze(1)
        samples["indices"] = indices

        return PrioritizedBatch(**samples)
    
    def _update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indices, priorities):
            priority = priority ** self.alpha
            self.sum_tree._update(idx, priority)
            self.min_tree._update(idx, priority)

    def __len__(self):
        return self.size
    
PrioritizedBatch = namedtuple("PrioritizedBatch", ["observations", "actions", "rewards", "next_observations", "dones", "indices", "weights"])