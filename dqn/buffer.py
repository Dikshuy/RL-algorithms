import random
from collections import namedtuple, deque
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
    
# priortized experience replay buffer
class PERBuffer:
    pass