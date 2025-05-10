import numpy as np
from utils import recv_socket_data
import json
from queue import PriorityQueue
from itertools import permutations
import socket
import time

# SETUP A* WITHOUT Q-LEARNING

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
        return not (y <= 2 or y + self.size[1] >= self.map_height - wall_width or x + self.size[0] >= self.map_width - wall_width)

    def neighbors(self, point, map_width, map_height, objs):
        """Generate walkable neighboring points avoiding collisions with objects."""
        step = 0.150
        directions = [(0, step), (step, 0), (0, -step), (-step, 0)]  # Adjacent squares: N, E, S, W
        x, y = point
        
        results = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < map_width and 0 <= ny < map_height and all(self.objects_overlap(nx, ny, self.size[0], self.size[1], obj['position'][0], obj['position'][1], obj['width'], obj['height']) for obj in objs) and  self.hits_wall( nx, ny):
                results.append((nx, ny))
        return results
    
    def is_close_enough(self, current, goal, tolerance=0.15, is_item = True):
        """Check if the current position is within tolerance of the goal position."""
        if is_item is not None:
            tolerance = 0.6
            return (abs(current[0] - goal[0]) < tolerance - 0.15  and abs(current[1] - goal[1]) < tolerance + 0.05 )
        else:
            return (abs(current[0] - goal[0]) < tolerance and abs(current[1] - goal[1]) < tolerance)
        
    def distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def astar(self, start, goal, objs, map_width, map_height, is_item=True):
        """Perform the A* algorithm to find the shortest path from start to goal."""
        print(f"Starting A* from {start} to {goal}")
        
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():

            current = frontier.get()

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

        print("No path found")
        return None
    
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
                # dis = abs(goal_y - y)
                # while dis> 0.75:
                #     self.step('NORTH')
                #     if abs(dis - abs(goal_y - self.player['position'][1])) < 0.1:
                #         break
                #     else:
                #         dis = abs(goal_y - self.player['position'][1])
                # return 'NORTH'
        elif goal_y > y:
            if cur_dir != '1':
                self.step('SOUTH')
                # dis = abs(goal_y - y)
                # while dis > 0.75:
                #     self.step('SOUTH')
                #     if abs(dis - abs(goal_y - self.player['position'][1])) < 0.1:
                #         break
                #     else:
                #         dis = abs(goal_y - self.player['position'][1])
                # return 'SOUTH'

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

    def compute_optimal_shelf_order(self, start, shelves, objs, map_width, map_height):
        """
        Compute the best sequence of shelves to visit using A* for distance calculations.

        :param start: (x, y) starting position.
        :param shelves: List of (x, y) positions of shelves to visit.
        :param objs: List of objects in the environment.
        :param map_width: Width of the map.
        :param map_height: Height of the map.
        :return: List of (x, y) shelves in the optimal order.
        """
        print("shelves: ", shelves)
        distance_matrix = {}

        close_enough = []
        # Compute shortest paths between all pairs of points
        for i in range(len(shelves)):
            path = self.astar(tuple(start), tuple(shelves[i]), objs, map_width, map_height)
            distance = len(path)
            close_enough.append(path[-1])
            distance_matrix[(tuple(start), tuple(shelves[i]))] = distance

        for j in range(len(shelves)):
            for k in range(j+1, len(shelves)):
                path = self.astar(tuple(close_enough[j]), tuple(shelves[k]), objs, map_width, map_height)

                distance = len(path)
                print("path distance from", tuple(close_enough[j]), "to", tuple(shelves[k]), ":", distance)
                distance_matrix[(tuple(shelves[j]), tuple(shelves[k]))] = distance
                distance_matrix[(tuple(shelves[k]), tuple(shelves[j]))] = distance

        # Try all possible shelf orders and pick the shortest one
        best_order = None
        best_distance = float('inf')

        for perm in permutations(shelves):
            perm = [tuple(p) for p in perm]
            total_distance = distance_matrix.get((tuple(start), perm[0]))
            for k in range(len(perm) - 1):
                total_distance += distance_matrix.get((perm[k], perm[k + 1]))

            if total_distance < best_distance:
                best_distance = total_distance
                best_order = perm

        print("best_distance:", best_distance)
        return [list(p) for p in best_order]
    

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

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    sock_game.send(str.encode("0 RESET"))  # reset the game
    state = recv_socket_data(sock_game)
    game_state = json.loads(state)

    player_num = int(input("Enter the player number: "))
    shopping_list = game_state['observation']['players'][player_num]['shopping_list']
    shopping_quant = game_state['observation']['players'][player_num]['list_quant']
    player = Agent(socket_game=sock_game, env=game_state, player_num=player_num)
    cartReturns = [2, 18.5]
    basketReturns = [3.5, 18.5]
    registerReturns_1 = [2, 4.5]
    registerReturns_2 = [2, 9.5]
    offset = 1
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
    print("shopping list: ", shopping_list)
    print("shopping quant: ", shopping_quant)

    path_to_basket = player.astar((player.player['position'][0], player.player['position'][1]), (basketReturns[0], basketReturns[1]), objs, player.map_width, player.map_height)
    player.perform_actions(player.from_path_to_actions(path_to_basket))
    player.face_item(basketReturns[0], basketReturns[1])
    player.step('INTERACT')
    player.step('INTERACT')

    shelf_positions = []
    for item in shopping_list:
        y_offset = -0.9
        item_pos = find_item_position(game_state, item)

        if item == 'milk' or item == 'chocolate milk' or item == 'strawberry milk':
            y_offset = 1.9  # adjust y position to visit the milk shelves from the bottom

        shelf_positions.append((item_pos[0] + offset, item_pos[1] + y_offset))

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
        y_offset = -0.9
        print("go for item: ", item)
        item_pos = find_item_position(game_state, item)

        if item == 'milk' or item == 'chocolate milk' or item == 'strawberry milk':
            y_offset = 1.9  # adjust y position to visit the milk shelves from the bottom
        
        # Compute path to next item in optimal item order list
        path = player.astar((player.player['position'][0], player.player['position'][1]),
                            (item_pos[0] + offset, item_pos[1] + y_offset), objs, player.map_width, player.map_height)
        print("first position in path: ", path[0])
        print("length of path: ", len(path))
        print("final position in path: ", path[-1])
        player.perform_actions(player.from_path_to_actions(path))
        player.face_item(item_pos[0] + offset, item_pos[1])
        
        print("Stop A* planner and use Q-learning agent to pick up the item")
        time.sleep(5)
        # for i in range(shopping_quant[shopping_list.index(item)]):
        #     player.step('INTERACT')
        #     if item == 'prepared foods' and item == 'fresh fish':
        #         player.step('INTERACT')

        basket_contents = player.obs['baskets'][0]['contents']
        basket_quant = player.obs['baskets'][0]['contents_quant']
        # print("basket_contents: ", basket_contents)
        # print("basket_quant: ", basket_quant)


