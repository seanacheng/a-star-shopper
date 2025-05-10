import pandas as pd
from utils import recv_socket_data
import json
from Q_Learning_agent_pick_and_place_in_basket import QLAgent
from astar_path_planner_random_walk import Agent

import random
import json
import socket

from utils import recv_socket_data

# SETUP FOR TESTING Q-LEARNING AGENT

objs = [
    {'height': 2.5, 'width': 3, 'position': [0.2, 4.5], 're_centered_position': [2.125, 5.75]},
    {'height': 2.5, 'width': 3, 'position': [0.2, 9.5], 're_centered_position': [2.125, 10.75]},
    {'height': 1, 'width': 2, 'position': [5.5, 1.5], 're_centered_position': [6.5, 2]},
    {'height': 1, 'width': 2, 'position': [7.5, 1.5], 're_centered_position': [8.5, 2]},
    {'height': 1, 'width': 2, 'position': [9.5, 1.5], 're_centered_position': [10.5, 2]},
    {'height': 1, 'width': 2, 'position': [11.5, 1.5], 're_centered_position': [12.5, 2]},
    {'height': 1, 'width': 2, 'position': [13.5, 1.5], 're_centered_position': [14.5, 2]},
    {'height': 1, 'width': 2, 'position': [5.5, 5.5], 're_centered_position': [6.5, 6]},
    {'height': 1, 'width': 2, 'position': [7.5, 5.5], 're_centered_position': [8.5, 6]},
    {'height': 1, 'width': 2, 'position': [9.5, 5.5], 're_centered_position': [10.5, 6]},
    {'height': 1, 'width': 2, 'position': [11.5, 5.5], 're_centered_position': [12.5, 6]},
    {'height': 1, 'width': 2, 'position': [13.5, 5.5], 're_centered_position': [14.5, 6]},
    {'height': 1, 'width': 2, 'position': [5.5, 9.5], 're_centered_position': [6.5, 10]},
    {'height': 1, 'width': 2, 'position': [7.5, 9.5], 're_centered_position': [8.5, 10]},
    {'height': 1, 'width': 2, 'position': [9.5, 9.5], 're_centered_position': [10.5, 10]},
    {'height': 1, 'width': 2, 'position': [11.5, 9.5], 're_centered_position': [12.5, 10]},
    {'height': 1, 'width': 2, 'position': [13.5, 9.5], 're_centered_position': [14.5, 10]},
    {'height': 1, 'width': 2, 'position': [5.5, 13.5], 're_centered_position': [6.5, 14]},
    {'height': 1, 'width': 2, 'position': [7.5, 13.5], 're_centered_position': [8.5, 14]},
    {'height': 1, 'width': 2, 'position': [9.5, 13.5], 're_centered_position': [10.5, 14]},
    {'height': 1, 'width': 2, 'position': [11.5, 13.5], 're_centered_position': [12.5, 14]},
    {'height': 1, 'width': 2, 'position': [13.5, 13.5], 're_centered_position': [14.5, 14]},
    {'height': 1, 'width': 2, 'position': [5.5, 17.5], 're_centered_position': [6.5, 18]},
    {'height': 1, 'width': 2, 'position': [7.5, 17.5], 're_centered_position': [8.5, 18]},
    {'height': 1, 'width': 2, 'position': [9.5, 17.5], 're_centered_position': [10.5, 18]},
    {'height': 1, 'width': 2, 'position': [11.5, 17.5], 're_centered_position': [12.5, 18]},
    {'height': 1, 'width': 2, 'position': [13.5, 17.5], 're_centered_position': [14.5, 18]},
    {'height': 1, 'width': 2, 'position': [5.5, 21.5], 're_centered_position': [6.5, 22]},
    {'height': 1, 'width': 2, 'position': [7.5, 21.5], 're_centered_position': [8.5, 22]},
    {'height': 1, 'width': 2, 'position': [9.5, 21.5], 're_centered_position': [10.5, 22]},
    {'height': 1, 'width': 2, 'position': [11.5, 21.5], 're_centered_position': [12.5, 22]},
    {'height': 1, 'width': 2, 'position': [13.5, 21.5], 're_centered_position': [14.5, 22]},
    {'height': 6, 'width': 0.7, 'position': [1, 18.5], 're_centered_position': [1.35, 21.5]},
    {'height': 6, 'width': 0.7, 'position': [2, 18.5], 're_centered_position': [2.35, 21.5]},
    {'height': 0.8, 'width': 0.8, 'position': [3.5, 18.5], 're_centered_position': [4.15, 19.4]},
    {'height': 2.25, 'width': 1.5, 'position': [18.25, 4.75], 're_centered_position': [19.125, 5.875]},
    {'height': 2.25, 'width': 1.5, 'position': [18.25, 10.75], 're_centered_position': [19.125, 11.875]}
]

def pick_random_shopping_item(data):
    item_options = []
    # Loop through each shelf in the data
    for shelf in data['observation']['shelves']:
        item_options.append(shelf['food_name'])
    for counter in data['observation']['counters']:
        item_options.append(counter['food'])
    
    return random.choice(item_options)

def find_item_position(data, item_name):
    """
    Finds the position of an item based on its name within the shelves section of the data structure.

    Parameters:
        data (dict): The complete data structure containing various game elements including shelves.
        item_name (str): The name of the item to find.

    Returns:
        list or None: The position of the item as [x, y] or None if the item is not found.
    """
    # Loop through each shelf in the data
    for shelf in data['observation']['shelves']:
        if shelf['food_name'] == item_name:
            return shelf['position']
    for counter in data['observation']['counters']:
        if counter['food'] == item_name:
            return counter['position']

if __name__ == "__main__":

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']
    action_space = len(action_commands)

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    sock_game.send(str.encode("0 NOP"))  # reset the game
    state = recv_socket_data(sock_game)
    game_state = json.loads(state)
    player = QLAgent(action_space=action_space, socket_game=sock_game, env=game_state)
    player.qtable = pd.read_json('pick_and_place_in_basket_qtable.json')

    sock_game.send(str.encode("1 NOP"))
    player_2_state = recv_socket_data(sock_game)
    player_2_state = json.loads(player_2_state)
    player_2 = Agent(socket_game=sock_game, env=player_2_state, player_idx=1)

    step_cnts = []
    success_cnt = 0
    training_time = 100
    for i in range(training_time):
        print("Start episode:", i)
        game_state = player.step('RESET')
        player_2.step('NOP')
        basketReturns = [3.5, 18.5]

        path_to_basket = player.astar((player.player['position'][0], player.player['position'][1]), (basketReturns[0], basketReturns[1]), objs, player.map_width, player.map_height)
        player.perform_actions(player.from_path_to_actions(path_to_basket))
        player.face_item(basketReturns[0], basketReturns[1])
        player.step('INTERACT')
        player.step('INTERACT')


        item = pick_random_shopping_item(game_state)
        player.shopping_list = [item]
        player.shopping_quant = [random.randint(1, 3)]
        print("go for item: ", item)
        item_pos = find_item_position(game_state, item)

        x_offset = 1
        y_offset = -0.9
        player_2_y_offset = 0
        if item == 'milk' or item == 'chocolate milk' or item == 'strawberry milk':
            y_offset = 2.4  # adjust y position to visit the milk shelves from the bottom
            player_2_y_offset = 3

        # Move player 2 to the item location
        # Player 2 gets stuck behind Q learning agent at basket return if going to southmost shelves
        if item_pos[1] > 20:
            player_2_path  = player_2.astar((player_2.player['position'][0] - 0.2, player_2.player['position'][1]),
                                (4.5, 15.6), objs, player_2.map_width, player_2.map_height)
            player_2.perform_actions(player_2.from_path_to_actions(player_2_path))
        player_2_path  = player_2.astar((player_2.player['position'][0] - 0.2, player_2.player['position'][1]),
                                (item_pos[0] + x_offset, item_pos[1] + player_2_y_offset), objs, player_2.map_width, player_2.map_height)
        player_2.perform_actions(player_2.from_path_to_actions(player_2_path))
        player_2.face_item(item_pos[0] + x_offset, item_pos[1])
        
        # Compute path to next item in optimal item order list
        path = player.astar((player.player['position'][0], player.player['position'][1]),
                            (item_pos[0] + x_offset, item_pos[1] + y_offset), objs, player.map_width, player.map_height)
        player.perform_actions(player.from_path_to_actions(path))
        player.face_item(item_pos[0] + x_offset, item_pos[1])
        game_state = player.state
        player.state['observation']['players'][0]['next_item'] = item  # pass on to Q-learning agent what the next item is
        player.state['observation']['players'][0]['curr_basket'] = 0  # pass on to Q-learning agent what the current basket is
        
        # Use Q-learning to pick and place the desired quantity of the item in the basket
        episode_length = 1000
        cnt = 0
        total_reward = 0
        holding_basket = True
        while not game_state['gameOver']:
            cnt += 1
            
            # After 40 counts, move player 2 out of the way
            if cnt == 40:
                # if player_2 is on the far right, move left, otherwise move right
                if player_2.player['position'][0] < 12:
                    x_offset += 4
                else:
                    x_offset -= 4
                if player_2.player['position'][1] < 20:
                    player_2_y_offset += 4
                else:
                    player_2_y_offset -= 4

                player_2_path  = player_2.astar((player_2.player['position'][0], player_2.player['position'][1]),
                                        (item_pos[0] + x_offset, item_pos[1] + player_2_y_offset), objs, player_2.map_width, player_2.map_height)

                player_2.perform_actions(player_2.from_path_to_actions(player_2_path))
                player_2.face_item(item_pos[0] + x_offset, item_pos[1])

            # Choose an action based on the current state
            action_index = player.choose_action(game_state, True)
            print(action_commands[action_index])
            print("------------------------------------------")

            # If direction does not match action, double action to turn 
            # to the correct direction before stepping in that direction
            # NORTH: direction = "0", action_index = 1
            # SOUTH: direction = "1", action_index = 2
            # EAST: direction = "2", action_index = 3
            # WEST: direction = "3", action_index = 4
            if 1 <= action_index <= 4 and int(player.current_direction) + 1 != action_index:
                player.step(action_commands[action_index])

            action = "0 " + action_commands[action_index]
            sock_game.send(str.encode(action))  # send action to env
            next_state = recv_socket_data(sock_game)  # get observation from env
            next_state = json.loads(next_state)
            next_state['observation']['players'][0]['next_item'] = item
            next_state['observation']['players'][0]['curr_basket'] = 0

            # if toggle basket, assume basket is dropped
            if action_index == 5 or not holding_basket:
                holding_basket = False
                next_state['observation']['players'][0]['curr_basket'] = -1

            # only if basket moves, assume player is holding the basket
            if next_state['observation']['baskets'][0]['position'] != game_state['observation']['baskets'][0]['position']:
                holding_basket = True
                next_state['observation']['players'][0]['curr_basket'] = 0

            # Update state
            game_state = next_state
            player.qtable.to_json('pick_and_place_in_basket_qtable.json')
            player.state = game_state

            basket_contents = player.obs['baskets'][0]['contents']
            basket_quant = player.obs['baskets'][0]['contents_quant']

            # If basket contains the item and the quantity is correct, move to the next item
            if item in basket_contents and int(basket_quant[basket_contents.index(item)]) == player.shopping_quant[0] and holding_basket:
                print("Episode success")
                step_cnts.append(cnt)
                success_cnt += 1
                break
            elif cnt > episode_length:  # If the episode length is reached, break
                print("Episode length exceeded")
                step_cnts.append(cnt)
                break

        
    print("Success Rate:", success_cnt / training_time)
    print("Efficiency:", sum(step_cnts)/ training_time)
    print("Step Counts:", step_cnts)

    # Close socket connection
    sock_game.close()


