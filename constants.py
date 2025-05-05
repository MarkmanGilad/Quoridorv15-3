import pygame
FPS = 60
WALLS = 3
WIDTH, HEIGHT = 900, 900
ROWS, COLS = 5,5
LINES = 4
LINE_SIZE = 10
SQUARE_SIZE = WIDTH//COLS
LINE_WIDTH = 15
BOARD_PADDING = 50
H_WIDTH, H_HEIGHT = 300, 100
M_WIDTH, M_HEIGHT = 300, 300
epsilon_start = 1
epsilon_final = 0.01
epsiln_decay = 10

#RGB
SPACE = (124,61,46)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (211,211,211)
GREEN = (0, 128, 0)
CADETBLUE1 = (152,245,255)
wallimagehor = pygame.image.load('wall.PNG')
wallimagevert = pygame.transform.rotate(wallimagehor,90)
w = wallimagevert.get_width()
h = wallimagevert.get_height()
wallimagevertsmall = pygame.transform.scale_by(wallimagevert,(w*0.02,h * 0.00152))
w2 = wallimagehor.get_width()
h2 = wallimagehor.get_height()
wallimagehorsmall = pygame.transform.scale_by(wallimagehor,(w2*0.00152,h2 * 0.02))
class Graphics:
    pass