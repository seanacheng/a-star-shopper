#Author: Hang Yu

import json
import random
import socket

import gymnasium as gym
from env import SupermarketEnv
from utils import recv_socket_data

from Q_Learning_agent import QLAgent  # Make sure to import your QLAgent class
import pickle
import pandas as pd


if __name__ == "__main__":
    

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT', 'RESET']
    # Initialize Q-learning agent
    action_space = len(action_commands) - 1   # Assuming your action space size is equal to the number of action commands
    agent = QLAgent(action_space)
    agent.qtable = pd.read_json('qtable.json')
    
    
    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    step_cnts = []
    success_cnt = 0
    training_time = 100
    episode_length = 1000
    for i in range(training_time):
        sock_game.send(str.encode("0 RESET"))  # reset the game
        state = recv_socket_data(sock_game)
        state = json.loads(state)
        cnt = 0
        total_reward = 0
        while not state['gameOver']:
            cnt += 1
            # Choose an action based on the current state
            action_index = agent.choose_action(state)
            action = "0 " + action_commands[action_index]

            print("Sending action: ", action)
            sock_game.send(str.encode(action))  # send action to env

            next_state = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(next_state)

            previous_agent_position = state['observation']['players'][0]['position']
            current_agent_position = next_state['observation']['players'][0]['position']
            current_holding_food = next_state['observation']['players'][0]['holding_food']
            previous_direction = state['observation']['players'][0]['direction']
            current_direction = next_state['observation']['players'][0]['direction']

            # Update state
            state = next_state
            agent.qtable.to_json('qtable.json')

            # defining success as picked up carrot and stopped for at 1 turn
            if current_holding_food == 'carrot' and previous_direction == current_direction and previous_agent_position == current_agent_position: 
                step_cnts.append(cnt)
                success_cnt += 1
                break
            elif cnt > episode_length:
                step_cnts.append(cnt)
                break
        # Additional code for end of episode if needed

    print("Success Rate:", success_cnt / training_time)
    print("Efficiency:", sum(step_cnts)/ training_time)
    print("Step Counts:", step_cnts)

    # Close socket connection
    sock_game.close()

