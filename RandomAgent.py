from State import State
from environment import Environment
import random


class RandomAgent:
    def __init__(self, player = None, env : Environment = None) -> None:
        self.env = env


    def getAction (self, events = None, graphics=None, state: State = None, epoch = 0, train = None):
        action1 = self.env.validmovelistpiece(self.env.state)
        action2 = self.env.valid_move_list_hor_wall()
        action3 = self.env.valid_move_list_vert_wall()
        actions = action2 + action3 + action1
        if random.randint(0,1) == 0:
            return random.choice(action1)
        return random.choice(actions)
