import torch
from environment import Environment
from State import State
state = State()

env = Environment(state)
obj = torch.load(f"Data/actions_20250505_004404.pth", weights_only=False)
actions = env.get_legal_actions(obj)
states = env.all_next_states(obj.to_tensor(), actions)
i = 1