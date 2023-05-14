# Author: Siddhant Tandon

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np
import random

from multi_agent_sim_data import Two_Agent_Exchange_Location_Scenario


class Generator():
    '''
    Class to try different sampling techniques instead of using multi_agent_sim_data
    data set images.
    The commented out code in main() can be used to test and generate coordinates.
    '''
    def __init__(self, N, dim):
        self.number = N
        self.dim = dim

    def travel_function(self, grid, x_prev, y_prev, step):

        if step == 1:
            y = y_prev + 1
            x = x_prev
        elif step == 2:
            y = y_prev - 1
            x = x_prev
        elif step == 3:
            x = x_prev - 1
            y = y_prev
        elif step == 4:
            x = x_prev + 1
            y = y_prev
        try:
            grid[x][y] = 1
        except:
            return 100,100,grid
        return x,y,grid

    def movements(self, initial_states):
        '''
        Using numpy random choices to generate the full sequence in one shot.
        Can be changed to some other sampling techniques
        '''
        c = [1,2,3,4]
        mat_full = np.zeros([self.dim * self.dim, self.number])
        a1 = random.choices(c, k=self.number)
        a2 = random.choices(c, k=self.number)
        x1 = initial_states[0][0]
        y1 = initial_states[0][1]
        x2 = initial_states[1][0]
        y2 = initial_states[1][1]
        arr = np.zeros(self.dim * self.dim)
        grid = arr.reshape(self.dim, self.dim)
        grid[x1][y1] = 1
        grid[x2][y2] = 1
        mat_full[:,0] = grid.reshape(self.dim * self.dim)

        # states
        ax1 = []
        ay1 = []
        ax2 = []
        ay2 = []
        ax1.append(x1)
        ay1.append(y1)
        ax2.append(x2)
        ay2.append(y2)

        for i in range(1, self.number):
            arr = np.zeros(self.dim * self.dim)
            grid = arr.reshape(self.dim, self.dim)
            for d in range(0,2):
                if d % 2 == 0:
                    x,y,grid = self.travel_function(grid, x1, y1, a1[i-1])
                    while (x < 0 or x > 4 or y < 0 or y > 4):
                        a_re = random.choices(c, k=1)
                        x,y,grid = self.travel_function(grid, x1, y1, a_re[0])
                    x1 = x
                    y1 = y
                    ax1.append(x)
                    ay1.append(y)
                else:
                    x,y,grid = self.travel_function(grid, x2, y2, a2[i-1])
                    while (x < 0 or x > 4 or y < 0 or y > 4):
                        a_re = random.choices(c, k=1)
                        x,y,grid = self.travel_function(grid, x2, y2, a_re[0])
                    x2 = x
                    y2 = y
                    ax2.append(x)
                    ay2.append(y)
            mat_full[:,i] = grid.reshape(self.dim * self.dim)

        return mat_full, ax1, ay1, ax2, ay2




def main():

    # generate images from the simulator
    goals = [[3,3], [2,2]]
    ma = Two_Agent_Exchange_Location_Scenario(goals)
    batch, X = ma.simulate_function(5, 500)

    # # load the trained weights
    # W = np.loadtxt('siddhant_5grid_1000img_0015lr_300epochs.txt')
    W = np.loadtxt('siddhant_weights_300epochs_0015la_1500imgs_5x5grid.txt')

    agent1_x = []
    agent1_y = []
    agent2_x = []
    agent2_y = []
    w_agent1_x = []
    w_agent1_y = []
    w_agent2_x = []
    w_agent2_y = []
    for b in batch:
        weight_coords = W.dot(b.image)
        w_agent1_x.append(weight_coords[0])
        w_agent1_y.append(weight_coords[1])
        w_agent2_x.append(weight_coords[2])
        w_agent2_y.append(weight_coords[3])
        agent1_x.append(b.states[0][0])
        agent1_y.append(b.states[0][1])
        agent2_x.append(b.states[1][0])
        agent2_y.append(b.states[1][1])


    # gen = Generator(100001, 5)
    # X2,ax1,ay1,ax2,ay2 = gen.movements(goals)
    # w_agent1_x1 = []
    # w_agent1_y1 = []
    # w_agent2_x2 = []
    # w_agent2_y2 = []
    #
    # for idx, img in enumerate(X2):
    #     wcoords2 = W.dot(X2[:,idx])
    #     w_agent1_x1.append(wcoords2[0])
    #     w_agent1_y1.append(wcoords2[1])
    #     w_agent2_x2.append(wcoords2[2])
    #     w_agent2_y2.append(wcoords2[3])

    # plotting happes here - DRAFT
    # TODO: ADD titles and more figues if necessary
    fig1 = plt.figure(1)
    hsv_modified = cm.get_cmap('hsv', 256)# create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
    newcmp = ListedColormap(hsv_modified(np.linspace(0.2, 0.8, 256)))# show figure
    # my_cmap = plt.get_cmap('hsv')
    p = plt.scatter(w_agent2_x, w_agent2_y,c=(agent2_x),cmap=newcmp)
    fig1.colorbar(p)
    fig1.show()

    fig2 = plt.figure(2)
    # plt.scatter(w_agent2_x, w_agent2_y, c=(agent2_y), cmap=my_cmap)
    plt.scatter(agent2_x, agent2_y)
    fig2.show()

    input()

if  __name__ == "__main__":
    main()
