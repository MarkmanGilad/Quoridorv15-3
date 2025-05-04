import torch
import random
import math
from DQN import DQN
from constants import *
from State import State
from environment import Environment

class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, env= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.player = player
        self.train = train
        self.setTrainMode()
        self.env : Environment = env
        

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def getAction (self, state: State, epoch = 0, events= None, train = True) -> tuple:
        actions = self.env.get_legal_actions(state)
        state_tensor = state.to_tensor()
        
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                idx = random.randrange(len(actions))
                action = actions[idx]
                return action, idx

        next_states = self.env.all_next_states(state_tensor=state_tensor, actions=actions, player=self.player)
        
        with torch.no_grad():
            Q_values = self.DQN(next_states)
        max_index = torch.argmax(Q_values)
        if train:
            return actions[max_index], Q_values[max_index]
        return actions[max_index]
        
    def get_Q_Values (self, states, actions):
        next_states_lst = []
        for i in range(len(states)):
            state_tensor = states[i].to_tensor()
            action = actions[i]
            next_state = self.env.next_state(state_tensor=state_tensor, action=action, player=self.player)
            next_states_lst.append(next_state)
        
        next_states = torch.stack(next_states_lst)
        Q_values = self.DQN(next_states)
        return Q_values

    def get_actions_values (self, states, dones):
        action_lst = []
        values_lst = []
        for i in range(len(states)):
            if dones[i]:
                action_lst.append((0,0,0))
                values_lst.append(torch.zeros(1, dtype=torch.float32).to(device=self.DQN.device))
                continue
            actions = self.env.get_legal_actions(states[i])
            state_tensor = states[i].to_tensor()
            next_states = self.env.all_next_states(state_tensor=state_tensor, actions=actions, player=self.player)
            with torch.no_grad():
                Q_values = self.DQN(next_states)
            max_index = torch.argmax(Q_values)
            action_lst.append(actions[max_index])
            values_lst.append(Q_values[max_index])
        
        Q_values = torch.stack(values_lst)
        return action_lst, Q_values





    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        # res = final + (start - final) * math.exp(-1 * epoch/decay)
        res = max(final, start - (start - final) * epoch / decay)
        return res
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)