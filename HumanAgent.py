import pygame
from constants import *
from State import *
from graphics import Graphics


class HumanAgent:
    def __init__(self, player, env): 
        self.player = player
        self.env = env
        # self.state = state
        # self.destination_pos = None
        # self.source_pos = None
        # self.vert_wall1_pos = None
        # self.hor_wall1_pos = None
        # self.vert_wall2_pos = None
        # self.hor_wall2_pos = None
        # self.mouse_pos = None
        
    

    def getAction(self,state, events= None, train = None):
        action = (5,5,5)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[0] > 899: return None
                    action = self.pos_to_board(mouse_pos)
                    
                elif event.button == 3:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[0] > 820 or mouse_pos[1] > 820 or mouse_pos[1] < 50 or mouse_pos[0] < 50: return None 
                    action = self.pos_to_Wall(mouse_pos)
                    
        if self.env.isLegal(action):
            return action
        return None
    
            


    def pos_to_board (self,mouse_pos):
        x, y = mouse_pos 
        col = x // SQUARE_SIZE 
        row = y // SQUARE_SIZE 
        return 0, row, col

    def pos_to_Wall (self,mouse_pos):
        x, y = mouse_pos
        col = (x) // 100 
        if  (y + 80)% 100> 65:
            row = (y - 50) // 100
            
            return 1,row,col 
        row = (y) // 100
        col = (x -50) // 100 
        return 2,row,col 
            

    
    
    # def move(self,mouse_pos):
    #     mouse_pos = pygame.mouse.get_pos()
    #                     # Translate mouse position to board coordinates
    #     col = mouse_pos[0] // SQUARE_SIZE 
    #     row = mouse_pos[1] // SQUARE_SIZE 
    #                 # Get mouse position
    #     self.source_pos =    
    #                     # If the clicked position is empty, it's the destination position
                            
    #         self.destination_pos = (row, col)
    #                             # Move the piece from the source to the destination position
    #         if self.state.is_valid_move(self.source_pos, self.destination_pos) == True:
    #             self.state.move_piece(self.source_pos, self.destination_pos)
    #             self.source_pos = None
    #             self.vert_wall1_pos = None
    #             self.hor_wall1_pos = None
    #             self.vert_wall2_pos = None
    #             self.hor_wall2_pos = None
    #         self.destination_pos = None
                            
                            
    #     else:   
    #                     # Check if the clicked position contains a piece
    #         self.source_pos = (row, col)
    #         if self.state.is_piece_there(self.source_pos) == False:
    #                 self.source_pos = None

    # def wall(self,mouse_pos):
    #     mouse_pos = pygame.mouse.get_pos()
    #                     # Translate mouse position to board coordinates
    #     col = (mouse_pos[0] ) // 100 
    #     if  (mouse_pos[1] + 80)% 100> 70:
    #         hor_row = (mouse_pos[1] - 50) // 100
            
    #         if self.hor_wall1_pos == None and self.vert_wall1_pos == None:
    #             self.hor_wall1_pos = (hor_row, col)
            
    #         elif self.vert_wall1_pos == None:
    #             self.hor_wall2_pos = (hor_row, col)
    #     else:
    #         vert_row = (mouse_pos[1] ) // 100
    #         if self.vert_wall1_pos == None and self.hor_wall1_pos == None:
    #             self.vert_wall1_pos = (vert_row, (mouse_pos[0] -50) // 100 )
            
    #         elif self.hor_wall1_pos == None:
    #             self.vert_wall2_pos = (vert_row, (mouse_pos[0] -50) // 100 )    
        

    #     if self.vert_wall1_pos != None and self.vert_wall2_pos != None:
    #         if self.state.is_valid_vertical_wall(self.vert_wall1_pos,self.vert_wall2_pos):
    #             self.state.add_vertical_walls(self.vert_wall1_pos,self.vert_wall2_pos)
    #         self.vert_wall1_pos = None
    #         self.hor_wall1_pos = None
    #         self.vert_wall2_pos = None
    #         self.hor_wall2_pos = None
        
    #     if self.hor_wall1_pos != None and  self.hor_wall2_pos != None:
    #         if self.state.is_valid_horizontal_wall(self.hor_wall1_pos,self.hor_wall2_pos):
    #             self.state.add_horizontal_walls(self.hor_wall1_pos,self.hor_wall2_pos)
    #         self.vert_wall1_pos = None
    #         self.hor_wall1_pos = None
    #         self.vert_wall2_pos = None
    #         self.hor_wall2_pos = None

   
    
    def randomagent(self):
        import random
        x = random.randint(0,1)
        pos_x = random.randint(1, 900) 
        pos_y = random.randint(1, 900)
        self.mouse_pos = (pos_x, pos_y)
        if x == 0:
            self.move(self.mouse_pos)
        elif x == 1:
            self.wall(self.mouse_pos)

    