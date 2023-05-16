# Author: Siddhant Tandon

import numpy as np
import random
import copy
import math
import argparse

class DataFrame():
    '''
    Class to save:
    1. flattened image
    2. Each agent's action as an array
    3. Each agent's rewards as an array
    4. Each agent's true state coordinate
    '''
    def __init__(self, image_array, agent_actions, agent_rewards, agent_states):
        self.image = image_array #(255-image_array)/255
        self.action = agent_actions
        self.reward = agent_rewards
        self.states = agent_states

class Two_Agent_Exchange_Location_Scenario(DataFrame):
    '''
    Scenario Environment Class
    Description: Random action simulator for an environment in which two agents
    are present. The goal location of each agent is to reach other agent's
    initial location. The purpose of defining a scenario is to help with
    generating rewards which will be used by loss functions in the gradient
    descent program.

    Attributes:
    1. num_agents = 2 (constant)
    2. agent_goals - array with first set describing goal location for Agent-1
    and second set describing goal location of Agent-2
    '''
    def __init__(self, agent_goals) -> None:
        self.num_agents = 2
        self.goals = agent_goals

    def generate_rewards(self,x,y, goal_x, goal_y):
        '''
        Description: Compute rewards based on distance from goal. Uses log()
        of distance and gives high reward when reaching the goal state.

        inputs:
        1. x - location of agent
        2. y - location of agent
        3. goal_x - goal location of agent
        4. goal_y - goal location of agent

        outputs:
        1. reward - computed reward value
        '''
        pg = [goal_x, goal_y]
        pa = [x,y]
        dist = math.dist(pg,pa)
        if dist == 0:
            reward = 3
        else:
            reward = np.log(dist)
        return reward

    def generate_pairs(self, x_prev, y_prev):
        '''
        Description: next random state generated for an agent with its action
        inputs:
        1. x_prev - previous state location of agent
        2. y_prev - previous state location of agent
        outputs:
        1. x - location of agent after taking action -> step
        2. y - location of agent after taking action -> step
        3. step - action taken
        '''
        step = random.randint(1,4) # actions

        if step == 1: # move up
            y = y_prev + 1
            x = x_prev
        elif step == 2: # move down
            y = y_prev - 1
            x = x_prev
        elif step == 3: # move left
            x = x_prev - 1
            y = y_prev
        elif step == 4: # move right
            x = x_prev + 1
            y = y_prev
        return x,y,step

    def travel_function(self, grid, x_prev, y_prev, goal_x, goal_y):
        '''
        Description: Calls generate pairs to get next state of agent, while
        checking whether the agent is traveling within the grid space. Also
        computes the reward and updates the grid with agent's new location
        inputs:
        1. grid - environment state
        2. x_prev - previous state location of agent
        3. y_prev - previous state location of agent
        4. goal_x - goal location of agent
        5. goal_y - goal location of agent
        outputs:
        1. grid - environment state
        2. x - new location of agent
        3. y - new location of agent
        4. step - action taken to reach new location
        5. reward - computed reward value for the new location
        '''
        s = grid.shape
        dim = s[0]
        x,y,step = self.generate_pairs(x_prev, y_prev)
        while not (x >= 0 and x < dim and y >= 0 and y < dim):
            x,y,step = self.generate_pairs(x_prev, y_prev)

        grid[x][y] = 1
        grid[x_prev][y_prev] = 255
        reward = self.generate_rewards(x,y, goal_x, goal_y)
        return grid, x, y, step, reward

    def simulate_function(self, grid_size, movements):
        '''
        Description: Runs the two agent scenario in a grid space for given
        number of movements.
        inputs:
        1. grid_size - dimension for a square environment
        2. movements - number of images generated
        outputs:
        1. dataset - list of DataFrame objects
        2. mat_full - matrix with flattened image vectors
        '''
        big_dim = grid_size ** 2
        mat_full = np.zeros([big_dim, movements])
        arr = np.ones(big_dim, dtype=int) * 255
        grid = arr.reshape(grid_size, grid_size)
        r_arr = np.ones([2,2], dtype=int)
        i = 0
        while i < 2:
            t = i + 1
            r_arr[t%2][0] = int(self.goals[i][0])
            r_arr[t%2][1] = int(self.goals[i][1])
            i +=1
        for r in range(0, 2):
            x = r_arr[r][0]
            y = r_arr[r][1]
            grid[x][y] = 1


        dataset = []
        for i in range(0,movements):
            actions = []
            rewards = []
            agent_coords = np.ones([self.num_agents, 2])
            idx = 0
            for rt in r_arr:
                grid,x,y, step, reward = self.travel_function(copy.deepcopy(grid), rt[0], rt[1], self.goals[idx%2][0], self.goals[idx%2][1])
                rt[0] = x
                rt[1] = y
                actions.append(step)
                rewards.append(reward)
                agent_coords[idx][0] = rt[0]
                agent_coords[idx][1] = rt[1]
                idx += 1
            grid.reshape(big_dim, 1)
            mat_full[:,i] = np.divide(np.subtract(255,grid.reshape(big_dim)),255)
            dt = DataFrame(mat_full[:,i], tuple(actions), tuple(rewards), agent_coords)
            dataset.append(dt)

        return dataset, mat_full


def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--movements', dest="movements", required=True, type=float, help="Please give the number of images for the dataset")

    args = parser.parse_args()
    goals = [[4,4],[0,0]]
    ma = Two_Agent_Exchange_Location_Scenario(goals)
    dataset, X = ma.simulate_function(5, int(args.movements))

    print("Test Worked!")
    print("Agent states after 1 movements: ")
    print(dataset[0].states)
    print("Agent states after 2 movements: ")
    print(dataset[1].states)




if __name__ == "__main__":
    main()
