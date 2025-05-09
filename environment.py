import numpy as np
from State import State
import torch
import time
from constants import *
from collections import deque


class Environment:
     
    def __init__(self, state:State ) -> None:
        self.state = state
        #    self.vertical_walls = state.vertical_walls
        self.opponent_coe_reward = 0.1
        self.step_reward = -0.2
        self.forward_reward = 0.1
        self.done_reward = 10
        
    def move_piece(self,pos):
        """
        Move a piece from one position to another.
        """
        from_row, from_col = self.get_player_row_col( self.state.current_player,self.state)
        to_row, to_col = pos


        # Check if the destination position is valid
        if self.is_valid_move(pos):
            # Move the piece
            self.state.board[to_row, to_col] = self.state.board[from_row, from_col]
            self.state.board[from_row, from_col] = 0  # Clear the original position
            self.state.current_player = self.state.current_player * -1
   
    def move_piecexxx(self,pos,state):


        from_row, from_col = self.get_player_row_col(state.current_player,state)
        to_row, to_col = pos


        state.board[to_row, to_col] = state.board[from_row, from_col]
        state.board[from_row, from_col] = 0  
    
    def is_valid_move(self,pos, state = None):
        """
        Check if the move from `from_pos` to `to_pos` is valid.
        """
        if state is None:
            state = self.state
        from_row, from_col = self.get_player_row_col(state.current_player,state)
        to_row, to_col = pos
        if  to_col > COLS-1 or to_row > ROWS-1 or to_row < 0 or to_col < 0:
            return False
        if (from_row - to_row) == 2 and from_row > 1 and state.horizontal_walls[to_row,to_col] != 2 and state.horizontal_walls[from_row -1,to_col] != 2 and state.board[from_row - 1, from_col] == state.current_player * -1 and from_col == to_col:
            return True
        if (to_row - from_row) == 2 and to_row > 1 and state.horizontal_walls[from_row + 1,to_col] != 2 and state.horizontal_walls[from_row ,to_col] != 2 and state.board[from_row + 1, from_col] == state.current_player * -1 and from_col == to_col:
            return True
        if (to_col - from_col) == 2 and to_col > 1 and state.vertical_walls[from_row ,from_col + 1] != 2 and state.vertical_walls[from_row ,from_col] != 2 and state.board[from_row , from_col + 1] == state.current_player * -1 and from_row == to_row:
            return True
        if (from_col - to_col) == 2 and from_col > 1 and state.vertical_walls[from_row ,from_col - 1] != 2 and state.vertical_walls[from_row ,from_col -2 ] != 2 and state.board[from_row , from_col - 1] == state.current_player * -1 and from_row == to_row:
            return True
        if (from_row - to_row) == 1 and (from_col - to_col) == 1 and state.horizontal_walls[to_row-1,to_col+1] == 2 and state.vertical_walls[to_row,to_col] != 2 and state.board[from_row - 1, from_col] == state.current_player * -1:
            return True
        if (from_row - to_row) == 1 and (from_col - to_col) == -1 and state.horizontal_walls[to_row-1,to_col-1] == 2 and state.vertical_walls[to_row,to_col-1] != 2 and state.board[from_row - 1, from_col] == state.current_player * -1:
            return True
        if (from_row - to_row) == -1 and (from_col - to_col) == 1 and state.horizontal_walls[to_row,to_col+1] == 2 and state.vertical_walls[to_row,to_col] != 2 and state.board[from_row + 1, from_col] == state.current_player * -1:
            return True
        if (from_row - to_row) == -1 and (from_col - to_col) == -1 and state.horizontal_walls[to_row,to_col-1] == 2 and state.vertical_walls[to_row,to_col-1] != 2 and state.board[from_row + 1, from_col] == state.current_player * -1:
            return True
        
        if (from_row - to_row) == 1 and (from_col - to_col) == 1 and state.vertical_walls[to_row+1,to_col-1] == 2 and state.horizontal_walls[to_row,to_col] != 2 and state.board[from_row , from_col -1] == state.current_player * -1:
            return True
        if (from_row - to_row) == -1 and (from_col - to_col) == 1 and state.vertical_walls[to_row-1,to_col-1] == 2 and state.horizontal_walls[to_row-1,to_col] != 2 and state.board[from_row , from_col -1] == state.current_player * -1:
            return True
        if (from_row - to_row) == 1 and (from_col - to_col) == -1 and state.vertical_walls[to_row+1,to_col] == 2 and state.horizontal_walls[to_row,to_col] != 2 and state.board[from_row , from_col +1] == state.current_player * -1:
            return True
        if (from_row - to_row) == -1 and (from_col - to_col) == -1 and state.vertical_walls[to_row-1,to_col] == 2 and state.horizontal_walls[to_row-1,to_col] != 2 and state.board[from_row, from_col +1] == state.current_player * -1:
            return True

        if (from_col - to_col ) == 1 and  state.vertical_walls[to_row,to_col] == 2:
            return False
        if (to_col - from_col ) == 1 and  state.vertical_walls[from_row,from_col] == 2:
            return False
        if (to_row - from_row ) == 1 and state.horizontal_walls[from_row,from_col] == 2:
            return False
        if (from_row - to_row) == 1 and state.horizontal_walls[to_row,to_col] == 2:
            return False
        if state.board[to_row, to_col]  == 0 and state.board[to_row, to_col]  is not state.board[from_row, from_col] and abs(from_row - to_row) < 2 and abs(from_col - to_col) < 2 and (from_row == to_row or from_col == to_col) and state.board[from_row, from_col] == state.current_player and  to_col <= COLS-1 and to_row <= ROWS-1:
            return True
        else:
            return False
   
    def is_piece_there(self, from_pos):
        from_row, from_col = from_pos
       
        if self.state.board[from_row, from_col] != self.state.current_player:
            return False
        else:
            return True
   
    def is_valid_vertical_wall(self, wall1, state=None):
        if state is None:
            state = self.state

        wall1_row, wall1_col = wall1
        wall2_row, wall2_col = wall1
        wall2_row += 1
        if  wall2_row == -1 or wall1_col >= COLS-1  or wall1_row >= ROWS-1 or\
                state.vertical_walls[wall1_row, wall1_col] == 2   or state.vertical_walls[wall2_row, wall2_col] == 2 or\
                        (state.current_player == -1 and state.white_wall_counter == 0) or (state.current_player == 1 and state.black_wall_counter == 0) or \
                        (state.horizontal_walls[wall1_row, wall1_col] == 2 and state.horizontal_walls[wall2_row-1, wall2_col+1] == 2):
            return False
        visited=np.zeros([ROWS,COLS])
        visited.fill(0)
        state1 = state.copy()
        state1.vertical_walls[wall1_row, wall1_col] = 2
        state1.vertical_walls[wall2_row, wall2_col] = 2
        state2 = state1.copy()
        state1.current_player = -1
        state2.current_player = 1
        path = []
        self.find_path(0,state1,self.get_player_row_col(-1, state1),visited,path)
        path2 = []
        visited.fill(0)
        self.find_path(ROWS-1,state2,self.get_player_row_col(1, state2),visited,path2)
        if  path == [] or path2 == []:
            return False
        return True
   
    def add_vertical_walls(self,wall1):  
        wall1_row, wall1_col = wall1
        wall2_row, wall2_col = wall1
        wall2_row += 1
        self.state.vertical_walls[wall1_row, wall1_col] = 2
        self.state.vertical_walls[wall2_row, wall2_col] = 2
        if self.state.current_player == -1:
            self.state.white_wall_counter -= 1
        else:
            self.state.black_wall_counter -= 1
        self.state.current_player = self.state.current_player * -1
   
    def is_valid_horizontal_wall(self, wall1, state = None):
        if state is None:
            state = self.state
       
        wall1_row, wall1_col = wall1
        wall2_row, wall2_col = wall1
        wall2_col += 1
        if  wall2_col == -1 or wall1_col >= COLS-1  or wall1_row >= ROWS-1 or state.horizontal_walls[wall1_row, wall1_col] == 2 \
                or state.horizontal_walls[wall2_row, wall2_col] == 2   or  (state.current_player == -1 and state.white_wall_counter == 0) \
                or (state.current_player == 1 and state.black_wall_counter == 0) \
                or (state.vertical_walls[wall1_row, wall1_col] == 2 and state.vertical_walls[wall2_row+1, wall2_col-1] == 2):
            return False
        visited=np.zeros([ROWS,COLS])
        visited.fill(0)
        state1 = state.copy()
        state1.horizontal_walls[wall1_row, wall1_col] = 2
        state1.horizontal_walls[wall2_row, wall2_col] = 2
        state2 = state1.copy()
        state1.current_player = -1
        state2.current_player = 1
        path = []
        self.find_path(0,state1,self.get_player_row_col(-1,state1),visited,path)
        path2 = []
        visited.fill(0)
        self.find_path(ROWS-1,state2,self.get_player_row_col(1,state2),visited,path2)
        if path == [] or path2 == []:
            return False
        return True
   
    def add_horizontal_walls(self,wall1):  
        wall1_row, wall1_col = wall1
        wall2_row, wall2_col = wall1
        wall2_col += 1
        self.state.horizontal_walls[wall1_row, wall1_col] = 2
        self.state.horizontal_walls[wall2_row, wall2_col] = 2
        if self.state.current_player == -1:
            self.state.white_wall_counter -= 1
        else:
            self.state.black_wall_counter -= 1
        self.state.current_player = self.state.current_player * -1
   
    def win_game(self):
        for i in range (COLS):
            if self.state.board[0,i] == -1:
                return "white"
            if self.state.board[ROWS-1,i] == 1:
                return "black"
        return None

    def get_player_row_col (self, player,state):
        board = state.board
        indices = np.where(board == player)
        row, col = indices[0][0], indices[1][0]
        return row, col
   
    def isLegal(self,action):
        pos =  action[1],action[2]
        if action[0] == 0:
           
            return self.is_valid_move(pos)
        if action[0] == 1:
            return self.is_valid_horizontal_wall(pos)
        if action[0] == 2:
            return self.is_valid_vertical_wall(pos)
        return False

    def move(self,action):
        pos =  action[1],action[2]
        if action[0] == 0:
            self.move_piece(pos)
        if action[0] == 1:
            self.add_horizontal_walls(pos)
        if action[0] == 2:
            self.add_vertical_walls(pos)
   
    def validmovelistpiece(self,state : State):
        from_row, from_col = self.get_player_row_col(state.current_player,state)
        to_row, to_col = from_row, from_col
        arr = []
        to_row = -1
        to_col = -1
        for i in range(ROWS):
            to_row += 1
            to_col = -1
            for j in range (COLS):
                to_col += 1
                if self.is_valid_move((to_row,to_col), state):
                    arr.append((0,to_row,to_col))
        return arr
       
    def move_list_piece_path(self, state):
        from_row, from_col = self.get_player_row_col(state.current_player,state)
        to_row, to_col = from_row, from_col
        arr = []
        for i in range(4):
            if i == 0:
                to_row+=1
            elif i == 1:
                to_row-=2
            elif i == 2:
                to_row+= 1
                to_col +=1
            elif i == 3:
                to_col -=2
            if (from_col - to_col ) == 1 and  state.vertical_walls[to_row,to_col] == 2:
                continue
            if (to_col - from_col ) == 1 and  state.vertical_walls[from_row,from_col] == 2:
                continue
            if (to_row - from_row ) == 1 and state.horizontal_walls[from_row,from_col] == 2:
                continue
            if (from_row - to_row) == 1 and state.horizontal_walls[to_row,to_col] == 2:
                continue
            if to_col <= COLS-1 and to_row <= ROWS-1 and to_row >= 0 and to_col >= 0:
                arr.append((to_row,to_col))
        return arr
      
    def find_path(self,target, state, pos, visited, path=[]):
        if pos[0] == target:
            return True
        if visited[pos] == 1:
            return False    
        visited[pos] = 1
        found = False
        temp = self.move_list_piece_path(state)
        for new_pos in temp:
            if not found:
                self.move_piecexxx(new_pos,state)
                found = self.find_path(pos=new_pos,state=state,visited=visited,path=path,target=target)
                if found:
                    path.append(new_pos)
        return found
       
    def valid_move_list_vert_wall(self, state:State = None):
        if state is None:
            state = self.state
        if (state.current_player == -1 and state.white_wall_counter == 0) or (state.current_player == 1 and state.black_wall_counter == 0):
            return []
        wall1_row, wall1_col = -1,-1
        arr = []
        for i in range (ROWS-1):
            wall1_col+=1
            wall1_row = -1
            for j in range (COLS-1):
                wall1_row +=1
                if self.is_valid_vertical_wall((wall1_row,wall1_col), state):
                    arr.append((2,wall1_row, wall1_col))
   
        return arr

    def valid_move_list_hor_wall(self, state: State = None):
        if state is None:
            state = self.state
        
        if (state.current_player == -1 and state.white_wall_counter == 0) or (state.current_player == 1 and state.black_wall_counter == 0):
            return []
        wall1_row, wall1_col = -1,-1
        arr = []
        for i in range (ROWS-1):
            wall1_row +=1
            wall1_col = -1
            for j in range (COLS-1):
                wall1_col+=1
                if self.is_valid_horizontal_wall((wall1_row,wall1_col), state):
                    arr.append((1,wall1_row, wall1_col))
   
        return arr    

    def Reset(self):
        self.state.reset()
    
    #region def toTensor(self, device = torch.device('cpu')) -> tuple:
    #     x1,y1 = self.get_player_row_col(1,self.state)
    #     player1 = x1,y1,self.state.black_wall_counter
    #     x2,y2 = self.get_player_row_col(-1,self.state)
    #     player2 = x2,y2,self.state.white_wall_counter
    #     player1 = np.array(player1)
    #     player2 = np.array(player2)
    #     board2 = self.state.horizontal_walls.reshape(-1)
    #     board3 = self.state.vertical_walls.reshape(-1)
    #     board_np = np.concatenate((player1,player2,board2,board3))
    #     print(board_np)
    #     board_np = torch.from_numpy(board_np)
    #     board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)
    #     action1 = self.validmovelistpiece(self.state)
    #     action2 = self.valid_move_list_hor_wall()
    #     action3 = self.valid_move_list_vert_wall()
    #     actions_np = action1 + action2 + action3
    #     actions_np = np.array(actions_np)
    #     actions_tensor = torch.from_numpy(actions_np)
    #     return board_tensor, actions_tensor
    #endregion

    #####################################################################

    def get_legal_actions (self, state):
        action1 = self.validmovelistpiece(state)
        action2 = self.valid_move_list_hor_wall(state)
        action3 = self.valid_move_list_vert_wall(state)
        actions = action1 + action2 + action3
        if len(actions) == 0:
            self.save(state)
        return actions

    def next_state(self, state_tensor, action ,player = 1):
        next_state = state_tensor.clone()
        type, row, col =  action[0], action[1], action[2]
        board, h_walls, v_walls = next_state[0], next_state[1], next_state[2]

        if type == 0:       # piece
            player_pos = torch.where(board > 0 if player == 1 else board < 0)
            player_row = player_pos[0][0]
            player_col = player_pos[1][0]
            board[row, col] = board[player_row, player_col]
            board[player_row, player_col] = 0  # Clear the original position
        if type == 1:       # h_walls
            wall1_row, wall1_col = row, col
            wall2_row, wall2_col = row, col+1
            h_walls[wall1_row, wall1_col] = 2
            h_walls[wall2_row, wall2_col] = 2
        if type == 2:       # v_walls
            wall1_row, wall1_col = row, col
            wall2_row, wall2_col = row+1, col
            v_walls[wall1_row, wall1_col] = 2
            v_walls[wall2_row, wall2_col] = 2
        
        return next_state
    
    def all_next_states (self, state_tensor, actions, player = 1):
        next_state_lst = []
        for action in actions:
            next_state_lst.append(self.next_state(state_tensor, action, player))
        
        if len(next_state_lst) == 0:
            self.save(next_state_lst)
            self.save(actions)

        next_states = torch.stack(next_state_lst, dim=0)
        return next_states

    def reward (self, state: State, action, next_state: State, player):
        reward = self.step_reward
        player1_pos = np.where(state.board==1)
        player1_next_pos = np.where(next_state.board==1)
        opponent_pos = np.where(state.board==-1)
        opponent_next_pos = np.where(next_state.board==-1)

        player_advanced = (player1_next_pos[0]-player1_pos[0]) * self.forward_reward
        opponenet_advanced = (opponent_pos[0]- opponent_next_pos[0]) * self.forward_reward
        reward += player_advanced - self.opponent_coe_reward * opponenet_advanced
        
        win = self.win(next_state)
        if  win == 1:
            reward += self.done_reward
        elif win == -1:
            reward -= self.done_reward
        
        pos = tuple(map(int, player1_pos))
        path = self.shortest_path(state, pos, len(state.board)-1)

        return reward
    
    def win (self, state=None):
       if state is None:
           state = self.state
       if -1 in state.board[0]:
           return -1
       elif 1 in state.board[ROWS-1]:
           return 1
       else:
           return 0
       
    def is_done (self, state = None):
        return self.win(state) != 0

    def shortest_path(self, state, start_pos, row_target):
        queue = deque()
        visited = set()
        prev = {}

        queue.append(start_pos)
        visited.add(start_pos)

        while queue:
            pos = queue.popleft()

            if pos[0] == row_target:
                path = []
                while pos in prev:
                    path.append(pos)
                    pos = prev[pos]
                path.append(start_pos)
                path.reverse()
                return path  

            for next_pos in self.get_legal_moves(state, pos):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
                    prev[next_pos] = pos

        return None  

    def get_legal_moves(self, state, pos):
        state = state.copy()
        row, col = self.get_player_row_col(state.current_player, state) 
        state.board[row, col] = 0
        state.board[pos] = state.current_player
        moves = self.validmovelistpiece(state)
        moves = [m[1:] for m in moves]
        return moves

    def save (self, obj):
        torch.save(obj, f"Data/error_{str(type(obj))}_{time.strftime('%Y%m%d_%H%M%S')}.pth")