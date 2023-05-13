# Author: Siddhant Tandon
# import
import numpy as np
import random
import copy
import math

class DataFrame():
    def __init__(self, image_array, agent_actions, agent_rewards, agent_states):
        self.image = image_array #(255-image_array)/255
        self.action = agent_actions
        self.reward = agent_rewards
        self.states = agent_states

class Two_Agent_Exchange_Location_Scenario(DataFrame):
    def __init__(self, agent_goals) -> None:
        self.num_agents = 2
        self.goals = agent_goals

    def generate_rewards(self,x,y, goal_x, goal_y):
        # goal_x = 9
        # goal_y = 9
        pg = [goal_x, goal_y]
        pa = [x,y]
        dist = math.dist(pg,pa)
        if dist == 0:
            reward = 3
        else:
            reward = np.log(dist)
        return reward

    def generate_pairs(self, x_prev, y_prev):
        step = random.randint(1,4)
        # prev = "previous "
        # print(prev, x_prev, y_prev)
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
        # print(x,y)
        return x,y,step

    def travel_function(self, grid, x_prev, y_prev, goal_x, goal_y):
        size = grid.shape
        dim = size[0]
        # print(dim)
        x,y,step = self.generate_pairs(x_prev, y_prev)
        while not (x >= 0 and x < dim and y >= 0 and y < dim):
            x,y,step = self.generate_pairs(x_prev, y_prev)

        # print(grid[x])
        grid[x][y] = 1
        grid[x_prev][y_prev] = 255
        reward = self.generate_rewards(x,y, goal_x, goal_y)
        return grid, x, y, step, reward

    def simulate_function(self, grid_size, movements):
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
        # coords_rand = np.random.randint(0,9,[self.num_agents, 2])
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

#############################################################################################################
class Single_Agent_Scenario(DataFrame):
    def __init__(self, agent_goal) -> None:
        self.num_agents = 1
        self.goals = agent_goal

    def generate_rewards(self,x,y, goal_x, goal_y):
        # goal_x = 9
        # goal_y = 9
        pg = [goal_x, goal_y]
        pa = [x,y]
        dist = math.dist(pg,pa)
        d = float(1/dist)
        reward = np.log(d)
        return reward

    def generate_pairs(self, x_prev, y_prev):
        step = random.randint(1,4)
        # prev = "previous "
        # print(prev, x_prev, y_prev)
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
        # print(x,y)
        return x,y,step

    def travel_function(self, grid, x_prev, y_prev, goal_x, goal_y):
        size = grid.shape
        dim = size[0]
        print(dim)
        x,y,step = self.generate_pairs(x_prev, y_prev)
        while not (x >= 0 and x < 10 and y >= 0 and y < 10):
            x,y,step = self.generate_pairs(x_prev, y_prev)

        # print(grid[x])
        grid[x][y] = 1
        grid[x_prev][y_prev] = 255
        reward = self.generate_rewards(x,y, goal_x, goal_y)
        return grid, x, y, step, reward

    def simulate_function(self, grid_size, movements):
        big_dim = grid_size ** 2
        mat_full = np.zeros([big_dim, movements])
        arr = np.ones(big_dim, dtype=int) * 255
        grid = arr.reshape(grid_size, grid_size)
        init_loc = [2,2]
        grid[init_loc[0]][init_loc[1]] = 1


        dataset = []
        for i in range(0,movements):
            actions = []
            rewards = []
            agent_coords = np.ones([self.num_agents, 2])

            grid,x,y, step, reward = self.travel_function(copy.deepcopy(grid), init_loc[0], init_loc[1], self.goals[0], self.goals[1])
            init_loc[0] = x
            init_loc[1] = y
            actions.append(step)
            rewards.append(reward)
            agent_coords[0][0] = init_loc[0]
            agent_coords[0][1] = init_loc[1]
            grid.reshape(big_dim, 1)
            mat_full[:,i] = (255 - grid.reshape(100)) / 255
            dt = DataFrame(mat_full[:,i], tuple(actions), tuple(rewards), agent_coords)
            dataset.append(dt)

        return dataset, mat_full # not adjacent timestamps right now
######################################################################################################################

def main ():
    # gt = Ground_truth()
    # i,c,a,r = gt.run_program()
    # print(r)
    # x1 = 2
    # y1 = 2
    # x_data,y_data = travel_function(10,grid, x1, y1)
    # print(x_data, y_data)

    # mat = np.zeros([100,25])
    # mat[:,1] = np.random.uniform(0,1,[100,1]).reshape(100)
    # print(mat)
    # ma = Multi_Agent(3)


    goals = np.array([[4,4],[0,0]])
    ma = Two_Agent_Exchange_Location_Scenario(goals)
    dataset, X = ma.simulate_function(5, 20)
    print(X)


    # print(grid_out)
    # np.savetxt('full_array_test.txt', dataset, fmt='%i')
    # goal = [9,9]
    # sa = Single_Agent_Scenario(goal)
    # dataset, _ = sa.simulate_function(10, 20)

    print(dataset[0].states, dataset[2].states)
    print(dataset[0].reward, dataset[2].reward)




if __name__ == "__main__":
    main()
