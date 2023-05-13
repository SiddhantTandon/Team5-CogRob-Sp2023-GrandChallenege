# -*- coding: utf-8 -*-
"""
Author: Allegra Farrar
Date: May 2023

This script encodes the transition from current state to next state based on provided
actions for a multi-agent case
The corresponding reward function that computes the reward value (float) from
taking the actions is implemented and applied here
"""
import numpy as np

def step(current_state, goal_state, actions, reward_fcn, size):
    """
    Determines the next state and reward from taking an action from a current state for each agent
    Implement following line to call fcn: next_state, reward = step(current_state, goal_state, actions, reward_fcn)
    
    Arguments:
    ----------
    current_state: numpy nd array of (x,y) positions for current state of each agent
    goal_state: numpy nd array of the target (x,y) positions for each agent
    actions: list of single actions for each agent
    reward_fcn: function computing the reward value for taking an action
    size: tuple (x,y) identifying the dimensions of the state space
    
    Returns:
    --------
    next_state: numpy nd array of (x,y) positions for next state of each agent after taking action
    reward: numpy 1d array containing float value reward from taking action for each agent
    """
    
    #encodes state changes for each available action
    #encodes violations to the state space
    
    #initialize array for updated state
    next_state = np.zeros(current_state.shape)
    
    for agent, agent_pos in enumerate(current_state):
        action = actions[agent]
        
        # "UP" Action:
        if action == "up":
            if agent_pos[0] == size[1]-1: #check is movement violates state space bound
                break
            else:
                next_state[agent] = agent_pos+(1,0) #determine next state
        
        # "DOWN" Action:
        if action == "down":
            if agent_pos[0] == 0:
                break
            else:
                next_state[agent] = agent_pos-(1,0)
        
        # "LEFT" Action:
        if action == "left":
            if agent_pos[1] == 0:
                break
            else:
                next_state[agent] = agent_pos-(0,1)
        
        # "RIGHT" Action:
        if action == "right":
            if agent_pos[1] == size[0]-1:
                break
            else:
                next_state[agent] = agent_pos+(0,1)
    
    if if_equal_positions(next_state) == True:
        return None, None
    
    else:          
        #Compute the reward from taking action:
        reward = reward_fcn(next_state, goal_state)    

        return next_state, reward


def reward_fcn(next_state, goal_state):
    """ 
    Computes reward value for updated state after taking action 
    reward value = log(1/euclidean distance from state to goal)
    
    Arguments:
    ----------
    next_state: numpy nd array of (x,y) positions for state of each agent
    goal_state: numpy nd array of the target (x,y) positions for each agent
    
    Returns:
    --------
    reward: numpy 1d array containing float value reward from taking action for each agent
    """
    #initialize:
    reward = np.zeros(len(next_state))
    
    #compute:
    for idx, state in enumerate(next_state):
        reward[idx] = np.log(1/np.linalg.norm(state - goal_state[idx]))
        
    return reward


def if_equal_positions(state):
    """
    Determines if action causes any of the agents to move to the same location on the grid
    
    Arguments:
    ----------
    state: numpy nd array of (x,y) positions for state of each agent
    
    Returns:
    --------
    boolean: true if any agent positions are the same, false if not
    """
    #check for single agent case
    if len(state) == 1:
        return False

    for idx1, agent_pos in enumerate(state):
        
        for idx2, next_agent_pos in enumerate(state):    

            if idx1 == idx2: #making sure you're not checking against positions for the same agent
                continue
            
            elif np.array_equal(agent_pos, next_agent_pos):
                print("Actions force agent", idx1, "to same grid position as agent", idx2, "-- try again!")
                return True
            
            else:
                #print("All Clear!")
                return False
            

