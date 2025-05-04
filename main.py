import pygame
from constants import *
from State import *
from graphics import Graphics
from HumanAgent import *
from environment import Environment
from RandomAgent import RandomAgent
from DQNAgent import DQN_Agent
import time
pygame.init()




clock = pygame.time.Clock()
env = Environment(State())
graphics = Graphics()
# player1 = HumanAgent(player=1, env=env)
player1 = DQN_Agent(player = 1,env = env)
# player2 = RandomAgent(-1,env)
player2 = HumanAgent(player=-1, env=env)
visited=np.zeros([9,9])




def main ():
    run = True
    source_pos = None
    vert_wall1_pos = None
    hor_wall1_pos = None
    vert_wall2_pos = None
    hor_wall2_pos = None
    victory_displayed = False
    player = player1



    step = 0
    while (run):
        print(step, end = "\r")
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
        action = player.getAction(state=env.state,events = events, train=False) # (type 0-2, row, col)
        if action != None:
            step+= 1
            env.move(action)
            graphics.draw_vertical_walls(env.state,action)
            graphics.draw_horizontal_walls(env.state,action)
            if player == player1:
                player = player2
            else:
                player = player1
            
 
        winner = env.win_game()
        if winner:
            graphics.draw_victory_screen(winner)
            env.state.reset()
            graphics.reset()
            print(step)
            step = 0
           
        else:
            graphics.draw(env.state)
       
       
        pygame.display.update()
        clock.tick(FPS)
    pygame.quit()  


if __name__ == '__main__':


    main()
