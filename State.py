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
        self.board = np.zeros([ROWS,COLS])
        self.board[0,COLS//2] = 1
        self.board[ROWS-1,COLS//2] = -1
        self.horizontal_walls = np.zeros([ROWS,COLS])
        self.vertical_walls = np.zeros([ROWS,COLS])
        self.current_player = 1
        self.white_wall_counter = WALLS
        self.black_wall_counter = WALLS
        self.Graphics = Graphics
    
    def reset(self):
        self.__init__()
        
    def to_tensor (self):
        player_pos = np.where(self.board == 1)
        player_row = int(player_pos[0][0])
        player_col = int(player_pos[1][0])
        player1_encode = 1
        opponent_pos = np.where(self.board == -1)
        opponent_row = int(opponent_pos[0][0])
        opponent_col = int(opponent_pos[1][0])
        opponent_encode = -1

        board = torch.zeros(self.board.shape, dtype=torch.float32)
        board[player_row, player_col] = player1_encode
        board[opponent_row, opponent_col] = opponent_encode

        # board = torch.tensor(self.board, dtype=torch.float32)
        h_walls = torch.tensor(self.horizontal_walls, dtype=torch.float32) / 2
        v_walls = torch.tensor(self.vertical_walls, dtype=torch.float32) / 2
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

    
        





        
