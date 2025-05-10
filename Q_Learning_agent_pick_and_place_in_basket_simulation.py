import pandas as pd
from utils import recv_socket_data
from Q_Learning_agent_pick_and_place_in_basket import QLAgent

import json
import socket
import time

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
    sock_game.send(str.encode("0 RESET"))  # reset the game
    state = recv_socket_data(sock_game)
    game_state = json.loads(state)
    player = QLAgent(action_space=action_space, socket_game=sock_game, env=game_state)
    player.qtable = pd.read_json('pick_and_place_in_basket_qtable.json')

    shopping_list = game_state['observation']['players'][0]['shopping_list']
    shopping_quant = game_state['observation']['players'][0]['list_quant']
    basketReturns = [3.5, 18.5]

    # Allow a maximum of 6 items in the shopping list
    count = 0
    for i in range(len(shopping_quant)):
        if count + shopping_quant[i] == 6:
            shopping_list = shopping_list[:i+1]
            shopping_quant = shopping_quant[:i+1]
            break
        elif count + shopping_quant[i] > 6:
            shopping_quant[i] = 6 - count
            shopping_list = shopping_list[:i+1]
            shopping_quant = shopping_quant[:i+1]
            break
        else:
            count += shopping_quant[i]
    player.shopping_list = shopping_list
    player.shopping_quant = shopping_quant
    print("shopping list: ", shopping_list)
    print("shopping quant: ", shopping_quant)

    path_to_basket = player.astar((player.player['position'][0], player.player['position'][1]), (basketReturns[0], basketReturns[1]), objs, player.map_width, player.map_height)
    player.perform_actions(player.from_path_to_actions(path_to_basket))
    player.face_item(basketReturns[0], basketReturns[1])
    player.step('INTERACT')
    player.step('INTERACT')

    x_offset = 1
    shelf_positions = []
    for item in shopping_list:
        y_offset = -0.9
        item_pos = find_item_position(game_state, item)

        if item == 'milk' or item == 'chocolate milk' or item == 'strawberry milk':
            y_offset = 2.4  # adjust y position to visit the milk shelves from the bottom

        shelf_positions.append((item_pos[0] + x_offset, item_pos[1] + y_offset))

    # Compute optimal shelf order
    optimal_shelf_order = player.compute_optimal_shelf_order(
        start=(player.player['position'][0], player.player['position'][1]),
        shelves=shelf_positions,
        objs=objs,
        map_width=player.map_width,
        map_height=player.map_height
    )
    position_to_item = {tuple(pos): item for item, pos in zip(shopping_list, shelf_positions)}
    optimal_item_order = [position_to_item[tuple(pos)] for pos in optimal_shelf_order]
    print("optimal item order: ", optimal_item_order)
    
    for item in optimal_item_order:
        print("go for item: ", item)
        item_pos = find_item_position(game_state, item)

        y_offset = -0.9
        if item == 'milk' or item == 'chocolate milk' or item == 'strawberry milk':
            y_offset = 2.4  # adjust y position to visit the milk shelves from the bottom

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
        holding_basket = True
        while not game_state['gameOver']:
            cnt += 1

            # Choose an action based on the current state
            action_index = player.choose_action(game_state, True)

            # If direction does not match action, double action to turn 
            # to the correct direction before stepping in that direction
            # NORTH: direction = "0", action_index = 1
            # SOUTH: direction = "1", action_index = 2
            # EAST: direction = "2", action_index = 3
            # WEST: direction = "3", action_index = 4
            if 1 <= action_index <= 4 and int(player.current_direction) + 1 != action_index:
                player.step(action_commands[action_index])

            print("Sending action: ", action_commands[action_index])
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

            basket_contents = player.obs['baskets'][0]['contents']
            basket_quant = player.obs['baskets'][0]['contents_quant']

            # If basket contains the item and the quantity is correct, move to the next item
            if item in basket_contents and basket_quant[basket_contents.index(item)] == shopping_quant[shopping_list.index(item)] and holding_basket:
                print("Picked up", basket_quant[basket_contents.index(item)], item)
                break
            elif cnt > episode_length:  # If the episode length is reached, break
                print("Failed to pick up", shopping_quant[shopping_list.index(item)], item)
                print("holding basket", holding_basket)
                # find the basket before going to next item
                if not holding_basket:
                    path_to_basket = player.astar((player.player['position'][0], player.player['position'][1]), (player.obs['baskets'][0]['position'][0] + 0.5, player.obs['baskets'][0]['position'][1]), objs, player.map_width, player.map_height)
                    player.perform_actions(player.from_path_to_actions(path_to_basket))
                    player.face_item(player.obs['baskets'][0]['position'][0] + 0.5, player.obs['baskets'][0]['position'][1])
                    player.step('TOGGLE_CART')
                    print("picking up basket")
                    time.sleep(10)
                break
        
        print("basket_contents: ", basket_contents)
        print("basket_quant: ", basket_quant)
        

    # Close socket connection
    sock_game.close()


