
import numpy as np
from utils import recv_socket_data
import json
from queue import PriorityQueue

import time
import socket
import argparse

from utils import recv_socket_data

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


def update_position_to_center(obj_pose):
    """
    Update the position of objects to their re_centered_position if their current position matches obj_pose.

    Parameters:
        objects (list of dicts): List of objects with details including position and re_centered_position.
        obj_pose (list): The position to match for updating to re_centered_position.
    
    Returns:
        None: Objects are modified in place.
    """
    global objs
    for obj in objs:
        # Compare current position with obj_pose
        if obj['position'] == obj_pose:
            # If they match, update position to re_centered_position
            obj_pose = obj['re_centered_position']
            break
    return obj_pose


class Agent:
    def __init__(self, socket_game, env, player_idx):
        self.player_idx = int(player_idx)
        self.shopping_list = env['observation']['players'][self.player_idx]['shopping_list']
        self.shopping_quant = env['observation']['players'][self.player_idx]['list_quant']
        self.game = socket_game
        self.map_width = 20
        self.map_height = 25
        self.obs = env['observation']
        self.cart = None
        self.basket = None
        self.player = self.obs['players'][self.player_idx]
        self.last_action = "NOP"
        self.current_direction = self.player['direction']
        self.size = [0.6, 0.4]

        

    def step(self, action):
        #print("Sending action: ", action)
        action = str(self.player_idx) + " " + action
        self.game.send(str.encode(action))  # send action to env
        output = recv_socket_data(self.game)  # get observation from env
        if output:
            output = json.loads(output)
            self.obs = output['observation']
            self.last_action = action
            self.player = self.obs['players'][self.player_idx]
        return output
    
    def heuristic(self, a, b):
        """Calculate the Manhattan distance from point a to point b."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def overlap(self, x1, y1, width_1, height_1, x2, y2, width_2, height_2):
        return  (x1 > x2 + width_2 or x2 > x1 + width_1 or y1 > y2 + height_2 or y2 > y1 + height_1)

    def objects_overlap(self, x1, y1, width_1, height_1, x2, y2, width_2, height_2):
        return self.overlap(x1, y1, width_1, height_1, x2, y2, width_2, height_2)
    def collision(self, x, y, width, height, obj):
        """
        Check if a rectangle defined by (x, y, width, height) does NOT intersect with an object
        and ensure the rectangle stays within the map boundaries.

        Parameters:
            x (float): The x-coordinate of the rectangle's top-left corner.
            y (float): The y-coordinate of the rectangle's top-left corner.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            obj (dict): An object with 'position', 'width', and 'height'.

        Returns:
            bool: Returns True if there is NO collision (i.e., no overlap) and the rectangle is within map boundaries,
                False if there is a collision or the rectangle goes outside the map boundaries.
        """
        # Define map boundaries
        min_x = 0.5
        max_x = 24
        min_y = 2.5
        max_y = 19.5

        # Calculate the boundaries of the rectangle
        rectangle = {
            'northmost': y,
            'southmost': y + height,
            'westmost': x,
            'eastmost': x + width
        }
        
        # Ensure the rectangle is within the map boundaries
        if not (min_x <= rectangle['westmost'] and rectangle['eastmost'] <= max_x and
                min_y <= rectangle['northmost'] and rectangle['southmost'] <= max_y):
            return False  # The rectangle is out of the map boundaries

        # Calculate the boundaries of the object
        obj_box = {
            'northmost': obj['position'][1],
            'southmost': obj['position'][1] + obj['height'],
            'westmost': obj['position'][0],
            'eastmost': obj['position'][0] + obj['width']
        }

        # Check if there is no overlap using the specified cardinal bounds
        no_overlap = not (
            (obj_box['northmost'] <= rectangle['northmost'] <= obj_box['southmost'] or
            obj_box['northmost'] <= rectangle['southmost'] <= obj_box['southmost']) and (
                (obj_box['westmost'] <= rectangle['westmost'] <= obj_box['eastmost'] or
                obj_box['westmost'] <= rectangle['eastmost'] <= obj_box['eastmost'])
            )
        )
        
        return no_overlap

    # The function will return False if the rectangle is outside the map boundaries or intersects with the object.
    

    def hits_wall(self, x, y):
        wall_width = 0.4
        return not (y <= 2 or y + self.size[1] >= self.map_height - wall_width or \
                x + self.size[0] >= self.map_width - wall_width) 
        # return y <= 2 or y + unit.height >= len(self.map) - wall_width or \
        #        x + unit.width >= len(self.map[0]) - wall_width or (x <= wall_width and
        #                                                            not self.at_door(unit, x, y))


    def neighbors(self, point, map_width, map_height, objs):
        """Generate walkable neighboring points avoiding collisions with objects."""
        step = 0.150
        directions = [(0, step), (step, 0), (0, -step), (-step, 0)]  # Adjacent squares: N, E, S, W
        x, y = point
        
        results = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
           # if 0 <= nx < map_width and 0 <= ny < map_height and all(self.collision(nx, ny, self.size[0], self.size[1], obj[]) for obj in objs):
            if 0 <= nx < map_width and 0 <= ny < map_height and all(self.objects_overlap(nx, ny, self.size[0], self.size[1], obj['position'][0],
                                                                                           obj['position'][1], obj['width'], obj['height']) for obj in objs) and  self.hits_wall( nx, ny):
                results.append((nx, ny))
        #print(results)
        return results
    def is_close_enough(self, current, goal, tolerance=0.15, is_item = True):
        """Check if the current position is within tolerance of the goal position."""
        if is_item is not None:
            tolerance = 0.6
            return (abs(current[0] - goal[0]) < tolerance - 0.15  and abs(current[1] - goal[1]) < tolerance +0.05 )

        else:
            return (abs(current[0] - goal[0]) < tolerance and abs(current[1] - goal[1]) < tolerance)
    def distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def astar(self, start, goal, objs, map_width, map_height, is_item = True):
        """Perform the A* algorithm to find the shortest path from start to goal."""
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        distance = 1000
   
        while not frontier.empty():

            current = frontier.get()
            #print(current, goal)
            # if distance > self.distance(current, goal):
            #     distance = self.distance(current, goal)
            #     print("getting closer: ", distance)
            if self.is_close_enough(current, goal, is_item=is_item):
                break

            for next in self.neighbors(current, map_width, map_height, objs):
                new_cost = cost_so_far[current] + 0.15  # Assume cost between neighbors is 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, goal)
                    frontier.put(next, priority)
                    came_from[next] = current

        # Reconstruct path
        if self.is_close_enough(current, goal, is_item=is_item):
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        return None  # No path found
    # PlayerAction.NORTH: (Direction.NORTH, (0, -1), 0),
    # PlayerAction.SOUTH: (Direction.SOUTH, (0, 1), 1),
    # PlayerAction.EAST: (Direction.EAST, (1, 0), 2),
    # PlayerAction.WEST: (Direction.WEST, (-1, 0), 3)
    def from_path_to_actions(self, path):
        """Convert a path to a list of actions."""
        # if the current direction is not the same as the direction of the first step in the path, add a TURN action
        # directions = [(0, step), (step, 0), (0, -step), (-step, 0)]  # Adjacent squares: N, E, S, W
        actions = []
        cur_dir = self.current_direction

        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            if x2 > x1:
                if cur_dir != '2':
                    actions.append('EAST')
                    cur_dir = '2'
                    actions.append('EAST' )
                else:
                    actions.append('EAST')
            elif x2 < x1:
                if cur_dir != '3':
                    actions.append('WEST')
                    cur_dir = '3'
                    actions.append('WEST')
                else:
                    actions.append('WEST')
            elif y2 < y1:
                if cur_dir != '0':
                    actions.append('NORTH')
                    cur_dir = '0'
                    actions.append('NORTH')
                else:
                    actions.append('NORTH')
            elif y2 > y1:
                if cur_dir != '1':
                    actions.append('SOUTH')
                    cur_dir = '1'
                    actions.append('SOUTH')
                else:
                    actions.append('SOUTH')
        return actions

    def face_item(self, goal_x, goal_y):
        x, y = self.player['position']
        cur_dir = self.current_direction
        #print("info: ", cur_dir, y, goal_y)
        if goal_y < y:
            if cur_dir != '0':
                self.step('NORTH')
                dis = abs(goal_y - y)
                while dis> 0.75:
                    self.step('NORTH')
                    if abs(dis - abs(goal_y - self.player['position'][1])) < 0.1:
                        break
                    else:
                        dis = abs(goal_y - self.player['position'][1])
                return 'NORTH'
        elif goal_y > y:
            if cur_dir != '1':
                self.step('SOUTH')
                dis = abs(goal_y - y)
                while dis > 0.75:
                    self.step('SOUTH')
                    if abs(dis - abs(goal_y - self.player['position'][1])) < 0.1:
                        break
                    else:
                        dis = abs(goal_y - self.player['position'][1])
                return 'SOUTH'

    def perform_actions(self, actions):
        """Perform a list of actions."""
        for action in actions:
            self.step(action)
        return self.obs
    
    def change_direction(self, direction):
        cur_dir = self.current_direction
        if direction == 'NORTH':
            if cur_dir != '0':
                self.step('NORTH')
                return 'NORTH'
        elif direction == 'SOUTH':
            if cur_dir != '1':
                self.step('SOUTH')
                return 'SOUTH'
        elif direction == 'EAST':
            if cur_dir != '2':
                self.step('EAST')
                return 'EAST'
        elif direction == 'WEST':
            if cur_dir != '3':
                self.step('WEST')
                return 'WEST'
            

def pick_random_shopping_item(data):
    item_options = []
    # Loop through each shelf in the data
    for shelf in data['observation']['shelves']:
        item_options.append(shelf['food_name'])
    for counter in data['observation']['counters']:
        item_options.append(counter['food'])
    
    return np.random.choice(item_options)

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

    parser = argparse.ArgumentParser(description="Run script with a player index.")
    parser.add_argument("--player_idx", type=int, required=True, help="Index of the player")
    args = parser.parse_args()


    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    player_idx = args.player_idx
    sock_game.send(str.encode(str(player_idx) + " NOP"))
    state = recv_socket_data(sock_game)
    game_state = json.loads(state)
    player = Agent(socket_game=sock_game, env=game_state, player_idx=player_idx)


    while True:
        x_offset = 1
        y_offset = 0
        item = pick_random_shopping_item(game_state)
        print("go for item: ", item)
        item_pos = find_item_position(game_state, item)

        if item == 'milk' or item == 'chocolate milk' or item == 'strawberry milk':
            y_offset = 3
        path  = player.astar((player.player['position'][0], player.player['position'][1]),
                                (item_pos[0] + x_offset, item_pos[1] + y_offset), objs, 20, 25)

        player.perform_actions(player.from_path_to_actions(path))
        player.face_item(item_pos[0] + x_offset, item_pos[1])
        time.sleep(10)
    

