# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

import json
import random
import socket

from env import SupermarketEnv
from utils import recv_socket_data


if __name__ == "__main__":

    # Make the env
    # env_id = 'Supermarket-v0'
    # env = gym.make(env_id)

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']

    print("action_commands: ", action_commands)

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))

    # Begin going south for a cart
    action = "0 SOUTH"
    confirmed = False # true when confirming the interaction with an object
    picking_leeks = False
    picking_carrots = False
    exiting = False
    dropped_cart = False # true when arrived at the shelf and let go of the cart
    while True:

        print("Sending action: ", action)
        sock_game.send(str.encode(action))  # send action to env

        output = recv_socket_data(sock_game)  # get observation from env
        output = json.loads(output)

        print("Observations: ", output["observation"])
        print("Violations", output["violations"])

        # Determines next action when each given violation is committed
        if len(output["violations"]) > 0: 
            if output["violations"][0] == 'Player 0 ran into the cart return':
                action = "0 INTERACT"
                continue
            elif output["violations"][0] == 'Player 0 ran into the garlic shelf with a cart':
                action = "0 SOUTH"
                continue
            elif output["violations"][0] == 'Player 0 ran into the leek shelf':
                action = "0 INTERACT"
                continue
            elif output["violations"][0] == 'Player 0 ran into the carrot shelf':
                action = "0 INTERACT"
                continue

        # interactions happen twice, first time only sets the confirmed flag to true
        # second time the player interacts, it will decide what to do next based on 
        # where in the process of grabbing a cart, or picking two items the player is
        if action == "0 INTERACT" and confirmed:
            confirmed = False
            # if grabbing a cart, turn right toward the food shelves
            if not picking_leeks and not picking_carrots:
                action = "0 EAST"
                picking_leeks = True
            # if picking leeks and holding a leek, go back down to the cart
            # otherwise, grab the cart and keep going right to get carrots
            elif picking_leeks:
                if output["observation"]["players"][0]["holding_food"] == "leek":
                    action = "0 SOUTH"
                    continue
                action = "0 TOGGLE_CART"
                dropped_cart = False
                picking_leeks = False
                picking_carrots = True
                continue
            # if picking carrots and holding a carrot, go back down to the cart
            # otherwise, grab the cart and keep going right to exit
            elif picking_carrots:
                if output["observation"]["players"][0]["holding_food"] == "carrot":
                    action = "0 SOUTH"
                    continue
                action = "0 TOGGLE_CART"
                dropped_cart = False
                picking_carrots = False
                exiting = True
                continue
        # after a player interacts, they need to confirm the interaction
        elif action == "0 INTERACT" and not confirmed:
            confirmed = True
            continue

        # when picking leeks, go to the approximate x, y coordinate of the leek shelf
        # then let go of the cart, then pick up the leek from the shelf
        if picking_leeks:
            if dropped_cart:
                if output["observation"]["players"][0]["holding_food"] is None:
                    action = "0 NORTH"
                    continue
            if output["observation"]["players"][0]["position"][1] > 19:
                if output["observation"]["players"][0]["position"][0] > 8:
                    if output["observation"]["players"][0]["holding_food"] == "leek":
                        if action == "0 EAST":
                            action = "0 INTERACT"
                            continue
                        action = "0 EAST"
                        continue
                    action = "0 TOGGLE_CART"
                    dropped_cart = True
                    continue
                action = "0 EAST"
                continue

        # when picking carrots, go to the approximate x, y coordinate of the carrot shelf
        # then let go of the cart, then pick up the carrot from the shelf
        if picking_carrots:
            if dropped_cart:
                if output["observation"]["players"][0]["holding_food"] is None:
                    action = "0 NORTH"
                    continue
            if output["observation"]["players"][0]["position"][1] > 19:
                if output["observation"]["players"][0]["position"][0] > 12:
                    if output["observation"]["players"][0]["holding_food"] == "carrot":
                        if action == "0 EAST":
                            action = "0 INTERACT"
                            continue
                        action = "0 EAST"
                        continue
                    action = "0 TOGGLE_CART"
                    dropped_cart = True
                    continue
                action = "0 EAST"
                continue

        # after the items have been grabbed, go right to get around the shelves
        # once passed the threshold of the shelves, go up to the level of the exit
        # and then left all the way out
        if exiting:
            if output["observation"]["players"][0]["position"][0] > 16:
                if output["observation"]["players"][0]["position"][1] < 15.5:
                    action = "0 WEST"
                    continue
                action = "0 NORTH"
                continue
            if action != "0 WEST":
                action = "0 EAST"
                continue




