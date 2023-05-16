# Author: Siddhant Tandon

import numpy as np # array manipulation
import argparse
import random
import time

#### loss function calls ####
from loss_functions import temporal_cohesion_sol
from loss_functions import proportionality_prior_sol
from loss_functions import causality_prior_sol
from loss_functions import repeatability_prior_sol
from loss_functions import temporal_loss
from loss_functions import proportional_loss
from loss_functions import causal_loss
from loss_functions import repeatability_loss
from loss_functions import multi_agent_loss
from loss_functions import multi_agent_loss_same_act
from loss_functions import multi_prior_sol
from loss_functions import multi_same_prior_sol

#### training set function call ####
from multi_agent_sim_data import Two_Agent_Exchange_Location_Scenario



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', dest="iterations", required=True, type=int, help="Please give the max iterations to train for")
    parser.add_argument('--learning_rate', dest="alpha", required=True, type=float, help="Please give the learning rate for gradient descent")
    parser.add_argument('--weight_name', dest="fname", required=True, help="Please give the name of a .txt file to save weights")

    args = parser.parse_args()
############################# 2 AGENTS ###############################
    # train data setup
    goals = [[4,4],[0,0]]
    ma = Two_Agent_Exchange_Location_Scenario(goals)
    # get all images X dim [25 {flattened image vector of 5x5 grid}, 400 {num images}]
    batch, X = ma.simulate_function(5, 400)

    # Weight matrix setup
    agent_dim = 2 * 2 # each agent has 2D-SPACE: x-coordinate, y-coordinate
    W = np.random.uniform(0,1,[agent_dim,25])


    start = time.time() # timing the loop
    # gradient descent loop
    prior_loss = 0
    losses_saved = np.zeros(args.iterations)
    for i in range(0, args.iterations):
        ep_time = time.time()
        total_loss = 0
        # get temporal loss
        temp_loss_grad = temporal_cohesion_sol(batch, W)
        total_loss += temp_loss_grad

        # get proportionality
        prop_loss_grad = proportionality_prior_sol(batch,W)
        total_loss += prop_loss_grad

        # get causality loss
        causal_loss_grad = causality_prior_sol(batch,W)
        total_loss += causal_loss_grad

        # get repeatability loss
        repeat_loss_grad = repeatability_prior_sol(batch,W)
        total_loss += repeat_loss_grad

        # get multi prior loss
        #multi_loss = multi_prior_sol(batch,W)
        #total_loss += multi_loss

        multi_loss = multi_same_prior_sol(batch, W)
        total_loss += multi_loss


        # update weights
        W = W - args.alpha * total_loss

        t_loss = temporal_loss(batch,W)
        p_loss = proportional_loss(batch,W)
        c_loss = causal_loss(batch,W)
        r_loss = repeatability_loss(batch,W)
        m_loss = multi_agent_loss_same_act(batch,W)
        loss_total = t_loss + p_loss + c_loss + r_loss + m_loss

        losses_saved[i] = loss_total
        if abs(prior_loss - loss_total) < 0.0012:
            print("Breaking loop! delta loss is lower than 0.0012")
            break

        prior_loss = loss_total

        # are we reducing loss?
        print("{}: {}_t + {}_p + {}_c + {}_r + {}_m = {}".format(i, t_loss, p_loss, c_loss, r_loss, m_loss, loss_total))
        # time keepers
        print("Epoch run time: {}".format(time.time() - ep_time))
        print("Time since start of program: {}".format(time.time() - start))


    # save weight
    print("Saving weights ... ")
    np.savetxt(args.fname, W, fmt='%f')
    print("Saved!")

    print("Saving losses ... ")
    np.savetxt('losses.txt', losses_saved, fmt='%f')
    print("Saved!")

    print("Time taken by the program: {}".format(time.time() - start))

if __name__ == "__main__":
    main()
