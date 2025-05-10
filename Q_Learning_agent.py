import numpy as np
import pandas as pd
import torch
import random
import torch.nn.functional as F
import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table
        self.qtable.index = pd.MultiIndex(levels=[[], [], [], []], codes=[[], [], [], []])  # allows multi-level indexing of states which are tuples 

    
    def trans(self, state, granularity=0.5):
        # TODO: You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn

        x_position = np.round(state['observation']['players'][0]['position'][0] / granularity)
        y_position = np.round(state['observation']['players'][0]['position'][1] / granularity)
        direction = state['observation']['players'][0]['direction']
        holding_food = state['observation']['players'][0]['holding_food']
        food_idx = None
        if holding_food is None:
            food_idx = 0
        elif holding_food == "carrot":
            food_idx = 1
        else:
            food_idx = 2
        return (float(x_position), float(y_position), direction, food_idx)
        
    def learning(self, action, rwd, state, next_state):
        # TODO: implement the Q-learning function
        
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

    def choose_action(self, state):
        # TODO: implement the action selection for the fully trained agent
        state = self.trans(state)
        print("state:", state)

        # check if state is in Q-table, otherwise initialize to 0
        if state not in self.qtable.index:
            self.qtable.loc[state] = [0] * self.action_space

        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.choice(self.action_space)
        else:
            # Exploit
            return self.qtable.loc[state].idxmax()
