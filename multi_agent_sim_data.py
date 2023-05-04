# import
import numpy as np
import random
import copy



class multi_agent():
    def __init__(self) -> None:
        pass

    def generate_pairs(self, x_prev, y_prev):
        step = random.randint(1,4)
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
        return x,y

    def travel_function(self, grid, x_prev, y_prev):

        x,y = self.generate_pairs(x_prev, y_prev)
        while not (x >= 0 and x < 10 and y >= 0 and y < 10):
            x,y = self.generate_pairs(x_prev, y_prev)

        print(x_prev, y_prev)
        print(x,y)
        grid[x][y] = 1
        grid[x_prev][y_prev] = 255
        return grid
        
    def simulate_function(self, num_agents, grid_size, movements):
        arr = np.ones(grid_size ** 2, dtype=int) * 255
        grid = arr.reshape(10,10)
        r_arr = []
        coords_rand = np.random.randint(0,9,[2, num_agents])
        for r in range(0, num_agents):
            x = coords_rand[r][0]
            y = coords_rand[r][1]
            grid[x][y] = 1
            r_arr.append((x,y))
        
        print(grid)
        for i in range(0,movements):
            for rt in r_arr:
                grid = self.travel_function(copy.deepcopy(grid), rt[0], rt[1])
        return grid
            
            
            

def main ():
    # gt = Ground_truth()
    # i,c,a,r = gt.run_program()
    # print(r)
    # x1 = 2
    # y1 = 2
    # x_data,y_data = travel_function(10,grid, x1, y1)
    # print(x_data, y_data)
    ma = multi_agent()
    grid_out = ma.simulate_function(2, 10, 1)
    print(grid_out)
    
    
    
    
if __name__ == "__main__":
    main()
