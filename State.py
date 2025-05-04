from constants import *
import numpy as np
from graphics import *
import torch 
import copy

class State:
    '''
        0 - empty
        1 - black
        -1 - white
        2 - wall
    '''
    def __init__(self) -> None:
        self.board = np.zeros([9,9])
        self.board[0,4] = 1
        self.board[8,4] = -1
        self.horizontal_walls = np.zeros([9,9])
        self.vertical_walls = np.zeros([9,9])
        self.current_player = 1
        self.white_wall_counter = 10
        self.black_wall_counter = 10
        self.Graphics = Graphics
    
    def reset(self):
        self.__init__()
        
    def to_tensor (self):
        board = torch.tensor(self.board, dtype=torch.float32)
        h_walls = torch.tensor(self.horizontal_walls, dtype=torch.float32)
        v_walls = torch.tensor(self.vertical_walls, dtype=torch.float32)
        state_tensor = torch.stack([board, h_walls, v_walls])
        return state_tensor

    def copy(self):
        s = State()
        s.board = self.board.copy()
        s.horizontal_walls = self.horizontal_walls.copy()
        s.vertical_walls = self.vertical_walls.copy()
        s.current_player = self.current_player
        s.white_wall_counter = self.white_wall_counter
        s.black_wall_counter = self.black_wall_counter
        s.Graphics = Graphics
        return s
    
    def target (self):
        if self.current_player == 1:
            return 8
        return 0

    
        





        
