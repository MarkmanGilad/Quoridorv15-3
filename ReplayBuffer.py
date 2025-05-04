from collections import deque
import random
import torch
import numpy as np
from State import State

capacity = 100000
end_priority = 9

class ReplayBuffer:
    def __init__(self, capacity= capacity, path = None) -> None:
        if path:
            self.buffer = torch.load(path).buffer
        else:
            self.buffer = deque(maxlen=capacity)

    def push (self, state : State, action, reward, next_state: State, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample (self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        # states = torch.vstack(state_tensors)
        # actions = torch.vstack(action_tensor)
        # rewards = torch.vstack(reward_tensors)
        # next_states = torch.vstack(next_state_tensors)
        # done_tensor = torch.tensor(dones).long().reshape(-1,1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)