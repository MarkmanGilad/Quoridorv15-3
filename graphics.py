import pygame
from constants import *
from State import *
import numpy as np

class Graphics:
    def __init__(self) -> None:
        self.win_width = WIDTH + 2 + 400
        self.win_height = HEIGHT + 2 
        self.win = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption('Quridor')
        self.font = pygame.font.SysFont(None, 48)
        self.victory_font = pygame.font.SysFont(None, 72)
        self.arr = []
        self.arr2 = []

    def reset(self):
        self.arr = []
        self.arr2 = []

    def calc_pos(self, row,col):
        x = (SQUARE_SIZE * col + SQUARE_SIZE) - SQUARE_SIZE // 2
        y = (SQUARE_SIZE * row + SQUARE_SIZE) - SQUARE_SIZE // 2
        return x,y
    
    def draw (self, state):
        self.draw_board()
        self.draw_pieces(state)
        self.draw_current_player(state)
        self.draw_vertical_walls(state)
        self.draw_horizontal_walls(state)
    
    def draw_pieces (self, state):
        black_positions = np.where(state.board == 1)
        white_positions = np.where(state.board == -1)

        # Draw black piece
        for row, col in zip(black_positions[0], black_positions[1]):
            x, y = self.calc_pos(row, col)
            radius = SQUARE_SIZE // 2 - 5
            pygame.draw.circle(self.win, BLACK, (x, y), radius - 10)

        # Draw white piece
        for row, col in zip(white_positions[0], white_positions[1]):
            x, y = self.calc_pos(row, col)
            radius = SQUARE_SIZE // 2 - 5
            pygame.draw.circle(self.win, (255, 255, 255), (x, y), radius - 10)
    

    def draw_board(self):
        self.win.fill((SPACE))
        for row in range(ROWS + 1):
            pygame.draw.line(self.win, (90, 58, 56), (0, row * SQUARE_SIZE), (WIDTH, row * SQUARE_SIZE), LINE_WIDTH)  
        for col in range(COLS + 1):
            pygame.draw.line(self.win, (90, 58, 56), (col * SQUARE_SIZE, 0), (col * SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    
    def draw_vertical_walls(self, state,action = None):
        if  action != None and action[0] == 2:
            x, y = self.calc_pos(action[1],action[2])
            if state.current_player == 1:
                wall_color = (255, 255, 255)
            else :
                wall_color = (0,0,0)
            start_point = (x + SQUARE_SIZE // 2 , y - SQUARE_SIZE // 2 + 5)
            end_point = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2 * 3 - 5)
        
            self.arr.append((wall_color,start_point, end_point))
        for i in range(len(self.arr)):
            pygame.draw.line(self.win, self.arr[i][0],self.arr[i][1],self.arr[i][2], 15)
            # self.win.blit(wallimagevertsmall, (x + 42,y - 50))
    
    def draw_horizontal_walls(self, state,action = None):
        if  action != None and action[0] == 1:
            x, y = self.calc_pos(action[1],action[2])
            if state.current_player == 1:
                wall_color = (255, 255, 255)
            else :
                wall_color = (0,0,0)
            start_point = (x - SQUARE_SIZE // 2 + 5, y + SQUARE_SIZE // 2)
            end_point = (x + SQUARE_SIZE // 2 * 3 - 5, y + SQUARE_SIZE // 2)
        
            self.arr2.append((wall_color,start_point, end_point))
        for i in range(len(self.arr2)):
            pygame.draw.line(self.win, self.arr2[i][0],self.arr2[i][1],self.arr2[i][2], 15)
            # self.win.blit(wallimagevertsmall, (x + 42,y - 50))

    def draw_current_player(self, state):
        current_player = "Black" if state.current_player == 1 else "White"
        current_player_text = f'Current Player: {current_player}'
        text_surface = self.font.render(current_player_text, True, WHITE)
        self.win.blit(text_surface, (920, 425))
        white_wall_text = self.font.render("white walls left: " + str(state.white_wall_counter), True, WHITE)
        self.win.blit(white_wall_text, (920, 800))
        black_wall_text = self.font.render("black walls left: " + str(state.black_wall_counter), True, WHITE)
        self.win.blit(black_wall_text, (920, 50))
    
    def highlight_hor_walls(self):
        x, y = pygame.mouse.get_pos()
        wall_width = 10
        wall_color = (255, 0, 0)
        start_point = (x - 50, y + 50)
        end_point = (x + 50, y + 50)
        pygame.draw.line(self.win, wall_color, start_point, end_point, wall_width)
    
    def highlight_vert_walls(self):
        x, y = pygame.mouse.get_pos()
        wall_width = 10
        wall_color = (255, 0, 0)
        start_point = (x + 50 , y - 50)
        end_point = (x + 50, y + 50)
        pygame.draw.line(self.win, wall_color, start_point, end_point, wall_width)
    
    def draw_victory_screen(self, winner):
        self.win.fill((0, 0, 0)) 
        victory_text = f'{winner} Wins!'
        victory_surface = self.victory_font.render(victory_text, True, (255, 255, 255)) 
        text_rect = victory_surface.get_rect(center=(self.win_width // 2, self.win_height // 2))
        self.win.blit(victory_surface, text_rect)
        pygame.display.flip()
       
    