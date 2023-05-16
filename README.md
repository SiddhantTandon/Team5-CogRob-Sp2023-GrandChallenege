# Project: Learning State Representations for Multi-Agent Systems from Multi-Agent Priors
####Team 5 - Learning by Exploiting Structure and Fundamental Knowledge
####CogRob-Spring2023-MIT

## Introduction
The repository contains the guide and implementation of the code for the grand challenge project. 

For the project, we are learning the state representation for a multi-robot system in a grid world. The data flow and process is outlined in the figure below. 


![Fig. Process outlining implementation](system_architecture_main_report.png)


## Simulation Environments

* Open grid — 2 agents, goals on opposing corners (they must swap locations)
* Object in middle — 3 agents, must navigate to opposite side of object from start location
* Partial maze — 2 agents, must navigate through maze


## Code Helpers
python3 gradient_descent.py --iterations 1000 --learning_rate 0.01 --weight_name save_weights_test.txt

python3 q_learning.py
