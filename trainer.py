import pygame
from constants import *
from State import *
from graphics import Graphics
from HumanAgent import *
from environment import Environment
from RandomAgent import RandomAgent
from DQNAgent import DQN_Agent
from ReplayBuffer import ReplayBuffer
import os
import wandb


def main (chkpt):
    
    epochs = 1000000
    C = 10
    learning_rate = 0.01
    batch_size = 32
    
    pygame.init()
    graphics = Graphics()
    env = Environment(State())
    
    player1 = DQN_Agent(player=1, env=env)
    player1_hat = DQN_Agent(player=1, env=env)
    player1_hat.DQN = player1.DQN.copy()
    player1_hat.DQN.train = False
    buffer = ReplayBuffer(path=None)
    optim = torch.optim.Adam(player1.DQN.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,10000, gamma=0.9)
    path = f"Quoridor{chkpt}.pth"
    player2 = RandomAgent(player=-1, env=env)
    
    for epoch in range(epochs):
        env.Reset()
        graphics.reset()
        step = 0
        end_of_game = False
        state = env.state.copy()
        while not end_of_game and step < 200:
            step += 1
            print(step, end="\r")

            # pygame.event.pump()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            ################# Sample Environement #################
            
            action, _ = player1.getAction(state=env.state)
            env.move(action)
            after_state = env.state.copy()
            reward = env.reward(state, after_state, player=player1.player)
            end_of_game = env.is_done()
            if end_of_game:
                buffer.push(state, action, reward, after_state, True)
                break
            
            after_action = player2.getAction(state=after_state)
            env.move(after_action)
            next_state = env.state.copy()
            reward += env.reward(state, next_state, player=player1.player)
            end_of_game = env.is_done()
            buffer.push(state, action, reward, next_state, end_of_game)
            state = next_state
            
            graphics.draw_vertical_walls(env.state,action)
            graphics.draw_horizontal_walls(env.state,action)
            graphics.draw(env.state)
            pygame.display.update()

            if len(buffer) < 100:
                continue
            
            ################ Train NN #######################
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = player1.get_Q_Values(states, actions)
            _, Q_hat_Values = player1.get_actions_values(next_states, dones)
            
            loss = player1.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if epoch % C == 0:
                player1_hat.DQN.load_state_dict(player1.DQN.state_dict())
            # scheduler.step()

        print(f"{chkpt}: steps: {step} win: {env.win()}")

    player1.save_param(path)
    
        
if __name__ == "__main__":
    if not os.path.exists("Data/checkpoit_num"):
        torch.save(1, "Data/checkpoit_num")    
    
    chkpt = torch.load("Data/checkpoit_num", weights_only=False)
    chkpt += 1
    torch.save(chkpt, "Data/checkpoit_num")    
    main (chkpt)
    pygame.quit()