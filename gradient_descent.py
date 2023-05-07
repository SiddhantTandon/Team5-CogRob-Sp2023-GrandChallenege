# Author: Siddhant Tandon
# CODE #
########
import numpy as np # array manipulation
import argparse  # cli for user
import random

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
from loss_functions import multi_prior_sol

#### training set function call ####
from multi_agent_sim_data import Two_Agent_Exchange_Location_Scenario



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', dest="iterations", required=True, type=int, help="Please give the max iterations to train for")
    parser.add_argument('--learning_rate', dest="alpha", required=True, type=float, help="Please give the learning rate for gradient descent")
    parser.add_argument('--weight_name', dest="fname", required=True, help="Please give the name of a .txt file to save weights")

    args = parser.parse_args()

    # train data setup
    goals = [[9,9],[0,0]]
    ma = Two_Agent_Exchange_Location_Scenario(goals)
    batch, X = ma.simulate_function(10, 20) # give all images X dim [100 {flattened image vector of 10x10 grid}, 25 {num images}]

    # Weight matrix setup
    agent_dim = 2 * 2 # each agent has 2D-SPACE: x-coordinate, y-coordinate
    W = np.random.uniform(0,1,[agent_dim,100])

    # gradient descent loop
    for i in range(0, args.iterations):

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
        multi_loss = multi_prior_sol(batch,W)
        total_loss += multi_loss

        # update weights
        W = W - args.alpha * total_loss

        t_loss = temporal_loss(batch,W)
        p_loss = proportional_loss(batch,W)
        c_loss = causal_loss(batch,W)
        r_loss = repeatability_loss(batch,W)
        m_loss = multi_agent_loss(batch,W)
        loss_total = t_loss + p_loss + c_loss + r_loss + m_loss

        # are we reducing loss?
        print("{}: {}_t + {}_p + {}_c + {}_r + {}_m = {}".format(i, t_loss, p_loss, c_loss, r_loss, m_loss, loss_total))
        print("temp: {}\nprop: {}\nCausal: {}\nRepeat: {}".format(np.isnan(temp_loss_grad).any(),
                                                                  np.isnan(prop_loss_grad).any(),
                                                                  np.isnan(causal_loss_grad).any(),
                                                                  np.isnan(repeat_loss_grad).any()))

    # coordinates learnt
    y = W.dot(X)
    print(y[0][:])

    # save weight
    print("Saving weights ... ")
    np.savetxt(args.fname, W, fmt='%f')
    print("Saved!")


if __name__ == "__main__":
    main()
