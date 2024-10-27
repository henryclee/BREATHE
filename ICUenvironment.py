import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from collections import defaultdict
from config import *


class GridEnvironment(gym.Env):
    metadata = {'render.modes':[]}

    def __init__(self, feature_dim, action_dim):       
        self.observation_space = spaces.Box(low = np.array([float('-inf') * feature_dim]), high = np.array([float('inf')* feature_dim]))        
        self.action_space = spaces.Discrete(action_dim)
        self.max_timesteps = 200

    def reset(self, **kwargs):

        # grid is the grid with: 0 = floor, 1 = wall, 2 = item, 3 = goal        
        self.grid = np.array([
            [0,0,1,0,0,0],
            [0,1,0,0,1,0],
            [0,0,0,1,0,0],
            [0,0,2,0,0,0],
            [1,0,1,0,1,0],
            [0,0,0,0,0,3]
        ], dtype = np.int8)

        self.timestep = 0

        self.state = {}

        self.state['agentloc'] = 0
        self.state['full'] = False
        self.state['item0'] = True

        info = {}

        # This avoids using one-hot encoding for agent location, but can't be unflattened easily
        # flattened_grid = self.state['grid'].flatten()
        # flattened_observation = np.concatenate([[self.state['agentloc']], flattened_grid, [self.state['full']]])

        flattened_observation = spaces.flatten(self.observation_space, self.state)

        return flattened_observation, info
    
   
    def step(self, action):

        reward = -1 # Every step starts with a reward of -1
        info = {}
        # dr and dc are zero for actions drop and pickup, so agent_r and agent_c will be correct
        dr, dc = dirs[action]

        ''' 
        If the environmenttype == "stochastic", the outcomes of the agent's actions are not fixed. E.g., if the agent 
        takes the action to go right, the agent ends up in the grid block on the right with a probability of 0.9 (90%),
        but stays in the same grid block with a probability of 0.1 (10%).
        
        This will only affect movement actions
        '''

        if self.environment_type == "stochastic":
            if action in [UP, DN, LT, RT]:
                if np.random.random() < 0.1:
                    dr, dc = 0,0

        agent_r = self.state['agentloc'] // COLS
        agent_c = self.state['agentloc'] % COLS

        agent_r += dr
        agent_c += dc

        # If we crash into a shelf or a boundary, update reward, and DON'T update self.state['agentloc']
        if agent_r < 0 or agent_r >= ROWS or agent_c < 0 or agent_c >= COLS or self.grid[agent_r][agent_c] == WALL:
            reward += RHITWALL
        else:
            self.state['agentloc'] = agent_r * COLS + agent_c

        truncated = False
        terminated = False

        if action == DROP:
            # We are allowing all actions now. If we try to drop when we're not holding, or try to drop the item not
            # in the exit location, we will make the reward negative

            if self.state['full']:
                self.state['full'] = False
                if self.grid[agent_r][agent_c] == GOAL:
                    reward += RSUCCESS
                    terminated = True
                else:
                    self.grid[agent_r][agent_c] = ITEM
                    reward += RWRONGDROP
            else:
                # print ("huh? - can't drop")
                reward += RCANTDROP

        if action == PICKUP:
            # Pickup is only valid if we're empty, item0 is stil on the grid and we're on a tile with an item.
            # We change our state to full, we update the grid to FLOOR, and we change the state of item0 to false
            # since the item has been pickedup
            if not self.state['full'] and self.state['item0'] and self.grid[agent_r][agent_c] == ITEM:
                self.state['full'] = True
                self.state['item0'] = False
                self.grid[agent_r][agent_c] = FLOOR
                reward += RPICKUP
                # print("pickup")
            # If we try to pickup when we are already holding, or when there is no item on the ground, 
            # we will make the reward negative
            else:
                # print ("huh? can't pickup")
                reward += RCANTPICKUP

        self.timestep += 1
        if not terminated and self.timestep >= self.max_timesteps:
            truncated = True

        flattened_observation = spaces.flatten(self.observation_space, self.state)

        return flattened_observation, reward, terminated, truncated, info

    def render(self):

        return

