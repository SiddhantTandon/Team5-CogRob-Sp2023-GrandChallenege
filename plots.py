# Author: Siddhant Tandon

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np
import random


class SampleDataFrame():
    '''
    Class dataframe to save
    1. image
    2. Agent 1 x coord
    3. Agent 1 y coord
    4. Agent 2 x coord
    5. Agent 2 y coord
    '''
    def __init__(self, image, ax1, ay1, ax2, ay2):
        self.image = image
        self.ax1 = ax1
        self.ay1 = ay1
        self.ax2 = ax2
        self.ay2 = ay2


class Sampler():
    '''
    Class with methods for sampling the envrionment state space

    Attributes:
    1. agents = 2 (constant)
    2. dim = size of grid
    3. count = number of samples

    Methods:
    1. simulation samples
    '''

    def __init__(self, grid_size, N):
        self.agents = 2
        self.dim = grid_size
        self.count = N

    def simulate_samples(self):
        '''
        input: self
        output:
        1. dataset - list of SampleDataFrame objects,
        2. mat_full - Matrix of flattened images
        '''
        c = [0,1,2,3,4] # used to create choices in the grid space
        mat_full = np.zeros([self.dim * self.dim, self.count])

        init_loc = [[1,1], [4,4]]
        arr = np.zeros(self.dim * self.dim)
        grid = arr.reshape(self.dim, self.dim)
        grid[init_loc[0][0]][init_loc[0][1]] = 1
        grid[init_loc[1][0]][init_loc[1][1]] = 1
        mat_full[:,0] = grid.reshape(self.dim * self.dim)

        dataset = []
        dt = SampleDataFrame(mat_full[:,0],init_loc[0][0], init_loc[0][1], init_loc[1][0], init_loc[1][1] )
        dataset.append(dt)

        for i in range(1, self.count):


            x1 = random.choices(c, k=1)
            y1 = random.choices(c, k=1)

            x2 = random.choices(c, k=1)
            y2 = random.choices(c, k=1)

            p = [x1[0],y1[0]]
            p2 = [x2[0], y2[0]]

            while (p == p2):
                x2 = random.choices(c, k=1)
                y2 = random.choices(c, k=1)
                p2 = [x2[0], y2[0]]

            arr = np.zeros(self.dim * self.dim)
            grid = arr.reshape(self.dim, self.dim)
            grid[x1[0]][y1[0]] = 1
            grid[x2[0]][y2[0]] = 1
            mat_full[:,i] = grid.reshape(self.dim * self.dim)
            dt = SampleDataFrame(mat_full[:,i],x1[0], y1[0], x2[0], y2[0] )
            dataset.append(dt)

        return dataset, mat_full


def main():

    ################################################
    sa = Sampler( 5, 50000)
    batch, X = sa.simulate_samples()

    # # load the trained weights
    W1 = np.loadtxt('siddhant_weights_400img_1000epochs_0025lr_5x5grid.txt')
    W3 = np.loadtxt('siddhant_weights_400img_3000epochs_0025lr_5x5grid.txt')
    W2 = np.loadtxt('siddhant_weights_400img_2000epochs_0025lr_5x5grid.txt')


    agent1_x = []
    agent1_y = []
    agent2_x = []
    agent2_y = []
    w1_agent1_x = []
    w1_agent1_y = []
    w1_agent2_x = []
    w1_agent2_y = []
    w2_agent1_x = []
    w2_agent1_y = []
    w2_agent2_x = []
    w2_agent2_y = []
    w3_agent1_x = []
    w3_agent1_y = []
    w3_agent2_x = []
    w3_agent2_y = []

    for b in batch:
        weight_coords1 = W1.dot(b.image)
        weight_coords2 = W2.dot(b.image)
        weight_coords3 = W3.dot(b.image)
        w1_agent1_x.append(weight_coords1[0])
        w1_agent1_y.append(weight_coords1[1])
        w1_agent2_x.append(weight_coords1[2])
        w1_agent2_y.append(weight_coords1[3])
        w2_agent1_x.append(weight_coords2[0])
        w2_agent1_y.append(weight_coords2[1])
        w2_agent2_x.append(weight_coords2[2])
        w2_agent2_y.append(weight_coords2[3])
        w3_agent1_x.append(weight_coords3[0])
        w3_agent1_y.append(weight_coords3[1])
        w3_agent2_x.append(weight_coords3[2])
        w3_agent2_y.append(weight_coords3[3])
        agent1_x.append(b.ax1)
        agent1_y.append(b.ay1)
        agent2_x.append(b.ax2)
        agent2_y.append(b.ay2)



    fig1 = plt.figure(1)
    hsv_modified = cm.get_cmap('hsv', 256)# create new hsv colormaps
    newcmp = ListedColormap(hsv_modified(np.linspace(0.2, 0.8, 256)))
    p1p = plt.scatter(w1_agent1_x, w1_agent1_y,c=(agent1_x),cmap=newcmp)
    plt.title("Mapped Coordinates - 400I, 0.0025LR, 1000E")
    plt.xlabel("Agent 2 Component 1")
    plt.ylabel("Agent 2 Component 2")
    fig1.colorbar(p1p,label="True X Coord Agent 2")
    fig1.show()



    fig2 = plt.figure(2)
    p2p = plt.scatter(w2_agent1_x, w2_agent1_y, c=(agent1_x), cmap=newcmp)
    plt.title("Mapped Coordinates - 400I, 0.0025LR, 2000E")
    plt.xlabel("Agent 2 Component 1")
    plt.ylabel("Agent 2 Component 2")
    fig2.colorbar(p2p,label="True X Coord Agent 2")
    fig2.show()



    fig3 = plt.figure(3)
    p3p = plt.scatter(w3_agent1_x, w3_agent1_y, c=(agent1_x), cmap=newcmp)
    plt.title("Mapped Coordinates - 400I, 0.0025LR, 3000E")
    plt.xlabel("Agent 2 Component 1")
    plt.ylabel("Agent 2 Component 2")
    fig3.colorbar(p3p,label="True X Coord Agent 2")
    fig3.show()


    fig4 = plt.figure(4)
    p4p = plt.scatter(w1_agent1_x, w1_agent1_y,c=(agent1_y),cmap=newcmp)
    plt.title("Mapped Coordinates - 400I, 0.0025LR, 1000E")
    plt.xlabel("Agent 2 Component 1")
    plt.ylabel("Agent 2 Component 2")
    fig4.colorbar(p4p,label="True X Coord Agent 2")
    fig4.show()



    fig5 = plt.figure(5)
    p5p = plt.scatter(w2_agent1_x, w2_agent1_y, c=(agent1_y), cmap=newcmp)
    plt.title("Mapped Coordinates - 400I, 0.0025LR, 2000E")
    plt.xlabel("Agent 2 Component 1")
    plt.ylabel("Agent 2 Component 2")
    fig5.colorbar(p5p,label="True X Coord Agent 2")
    fig5.show()



    fig6 = plt.figure(6)
    p6p = plt.scatter(w3_agent1_x, w3_agent1_y, c=(agent1_y), cmap=newcmp)
    plt.title("Mapped Coordinates - 400I, 0.0025LR, 3000E")
    plt.xlabel("Agent 2 Component 1")
    plt.ylabel("Agent 2 Component 2")
    fig6.colorbar(p6p,label="True X Coord Agent 2")
    fig6.show()


    # Plotting the losses
    fig7 = plt.figure(7)
    losses = np.loadtxt('losses.txt')
    epochs = np.linspace(0,500,500)

    plt.plot(epochs, losses)
    plt.title("Gradient Descent Loss for 500 Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig7.show()


    input()

if  __name__ == "__main__":
    main()
