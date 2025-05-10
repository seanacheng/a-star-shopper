import numpy as np
import pandas as pd
from utils import recv_socket_data
import json
from queue import PriorityQueue
from itertools import permutations



def euclidean_distance(pos1, pos2):
    # Calculate Euclidean distance between two points
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, socket_game, env, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table
        self.qtable.index = pd.MultiIndex(levels=[[], [], [], []], codes=[[], [], [], []])  # allows multi-level indexing of states which are tuples 

        self.state = env
        self.shopping_list = env['observation']['players'][0]['shopping_list']
        self.shopping_quant = env['observation']['players'][0]['list_quant']
        self.game = socket_game
        self.map_width = 20
        self.map_height = 25
        self.obs = self.state['observation']
        self.cart = None
        self.basket = None
        self.player = self.obs['players'][0]
        self.last_action = "NOP"
        self.current_direction = self.player['direction']
        self.size = [0.6, 0.4]

    
    def trans(self, state):
        x_position = round(state['observation']['players'][0]['position'][0], 1)
        y_position = round(state['observation']['players'][0]['position'][1], 1)
        basket_contents = []
        basket_contents_quant = []
        if state['observation']['baskets']:
            basket_contents = state['observation']['baskets'][0]['contents']
            basket_contents_quant = state['observation']['baskets'][0]['contents_quant']
        holding_basket_flag = True
        if state['observation']['players'][0]['curr_basket'] == -1:
            holding_basket_flag = False

        personal_space_flag = False
        if len(state['observation']['players']) > 1:
            for i in range(1, len(state['observation']['players'])):
                other_x_position = round(state['observation']['players'][i]['position'][0], 1)
                other_y_position = round(state['observation']['players'][i]['position'][1], 1)
                if abs(x_position - other_x_position) <= 0.9 and abs(y_position - other_y_position) <= 0.7:
                    personal_space_flag = True

        next_item = state['observation']['players'][0]['next_item']
        next_item_x_position = None
        next_item_y_position = None
        for shelf in state['observation']['shelves']:
            if shelf['food'] == next_item:
                next_item_x_position = round(shelf['position'][0], 1)
                next_item_y_position = round(shelf['position'][1], 1)
                break
        for counter in state['observation']['counters']:
            if counter['food'] == next_item:
                next_item_x_position = round(counter['position'][0], 1)
                next_item_y_position = round(counter['position'][1], 1)

        # next_item_distance = euclidean_distance((x_position, y_position), (next_item_x_position, next_item_y_position))
        # if y_position > next_item_y_position:
        #     next_item_distance = -next_item_distance
        next_item_distance = next_item_y_position - y_position  # distance only measured in y coordinates
        next_item_idx = self.shopping_list.index(next_item)
        next_item_quant = self.shopping_quant[next_item_idx]
        remaining_next_item_quant = next_item_quant
        if next_item in basket_contents:
            basket_contents_item_quant = basket_contents_quant[basket_contents.index(next_item)]
            remaining_next_item_quant = int(next_item_quant) - int(basket_contents_item_quant)
            
        return (round(next_item_distance, 1), holding_basket_flag, remaining_next_item_quant, personal_space_flag)
        
    def learning(self, action, rwd, state, next_state):
        state = self.trans(state)
        next_state = self.trans(next_state)

        # check if next_state is in Q-table, otherwise initialize to 0
        if next_state not in self.qtable.index:
            self.qtable.loc[next_state] = [0] * self.action_space

        # Q(S, A) = Q(S, A) + alpha * [R + gamma * max(a)[Q(S', a)] - Q(S, A)]
        max_next_q_value = np.max(self.qtable.loc[next_state])
        q_value = self.qtable.loc[state, action]
        new_q_value = q_value + self.alpha * (rwd + self.gamma * max_next_q_value - q_value)
        
        # Update Q-table
        self.qtable.loc[state, action] = new_q_value

        # Decay epsilon
        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay

    def choose_action(self, state, testing_flag=False):
        state = self.trans(state)
        if testing_flag:
            # reading state from json requires string state
            state = str(state)
        print("state:", state)

        # check if state is in Q-table, otherwise initialize to 0
        if state not in self.qtable.index:
            self.qtable.loc[state] = [0] * self.action_space
        
        # When testing, do not explore
        if np.random.rand() < self.epsilon and not testing_flag:
            # Explore
            return np.random.choice(self.action_space)
        else:
            # Exploit
            max_value = self.qtable.loc[state].max()  # Get the max Q-value
            best_actions = self.qtable.loc[state][self.qtable.loc[state] == max_value].index  # Get all actions with max value
            return np.random.choice(best_actions)  # Randomly select one

    def calculate_reward(self, previous_state, current_state):
        previous_state = self.trans(previous_state)
        current_state = self.trans(current_state)

        previous_next_item_distance = previous_state[0]
        previous_holding_basket_flag = previous_state[1]
        previous_remaining_next_item_quant = previous_state[2]
        previous_personal_space_flag = previous_state[3]

        current_next_item_distance = current_state[0]
        current_holding_basket_flag = current_state[1]
        current_remaining_next_item_quant = current_state[2]
        current_personal_space_flag = current_state[3]  # don't care about this until next action

        # Large penalty for dropping the basket in any state
        if previous_holding_basket_flag and not current_holding_basket_flag:
            return -3
        # In states where player is not close to another shopper, pick and place the desired item in the basket
        elif not previous_personal_space_flag:
            # No penalty for moving towards the shelf with the basket
            if abs(current_next_item_distance) < abs(previous_next_item_distance) and current_holding_basket_flag:
                return 0
            # Positive reward for reducing next item quantity with the basket
            elif current_remaining_next_item_quant < previous_remaining_next_item_quant and current_holding_basket_flag:
                return 10
            # Penalize everything else when not near other players
            else:
                return -1
        # In states where player is close to another shopper, avoid moving to prevent triggering the personal space norm
        elif previous_next_item_distance == current_next_item_distance:
            # No penalty for waiting when in the personal space of another player
            return 1
        else:
            # Penalize movement if in the personal space of another player
            return -2


    def step(self, action):
        action = "0 " + action
        self.game.send(str.encode(action))  # send action to env
        output = recv_socket_data(self.game)  # get observation from env
        if output:
            output = json.loads(output)
            self.state = output
            self.obs = output['observation']
            self.player = self.obs['players'][0]
            self.current_direction = self.player['direction']
            self.last_action = action
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
        if goal_y < y:
            if cur_dir != '0':
                self.step('NORTH')
        elif goal_y > y:
            if cur_dir != '1':
                self.step('SOUTH')

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
            print("permutation:", perm)
            total_distance = distance_matrix.get((tuple(start), perm[0]))
            for k in range(len(perm) - 1):
                total_distance += distance_matrix.get((perm[k], perm[k + 1]))
            print("total distance:", total_distance)

            if total_distance < best_distance:
                best_distance = total_distance
                best_order = perm

        print("best_distance:", best_distance)
        return [list(p) for p in best_order]



