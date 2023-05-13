import numpy as np
from implementations_nfq.nfq.agents import NFQAgent
from implementations_nfq.nfq.networks import NFQNetwork
import torch
import torch.optim as optim
from step import step, reward_fcn
from multi_agent_sim_data import Two_Agent_Exchange_Location_Scenario

class QLearner_Sim:

    def __init__(self, goal_states, init_states, step_fun, reward_fun, mapping, grid_size=(10,10), max_steps = 20):
        self.goal_states = goal_states
        self.init_states = init_states
        self.current_states = init_states.copy()
        self.dims = grid_size
        self.step_fun = step_fun
        self.reward_fun = reward_fun
        self.mapping = mapping
        self.act_arr = ["up", "down", "left", "right"]
        self.t = 0
        self.max_steps = max_steps


    def action_to_str(self, action):
        action = int(action.item())
        return (self.act_arr[int(action/4)], self.act_arr[action % 4])

    def step(self, action):
        self.t += 1
        as_str = self.action_to_str(action)
        new_state, reward = self.step_fun(self.current_states, self.goal_states, as_str, self.reward_fun)
        if new_state is None:
            new_state = self.current_states
            reward = (0,0)
        new_state = new_state.astype(int)

        out_img = np.zeros(self.dims)
        out_img[new_state[0][0], new_state[0][1]] = 1
        out_img[new_state[1][0], new_state[1][1]] = 1
        done = True
        for i in range(0, len(new_state)):
            if (new_state[i] != self.goal_states[i]).any():
                done = False
        #print("step with action: {} from {} gave reward {}".format(as_str, self.current_states, reward))
        self.current_states = new_state
        #TODO: set out to be the coordintate points of the agents
        out = self.mapping @ out_img.flatten()
        return out, reward[0]+reward[1], done, {"time_limit": self.t >= self.max_steps}

    def reset(self):
        self.current_states = self.init_states.copy()
        out_img = np.zeros(self.dims)
        out_img[self.current_states[0][0], self.current_states[0][1]] = 1
        out_img[self.current_states[1][0], self.current_states[1][1]] = 1
        out = self.mapping @ out_img.flatten()
        self.t = 0
        return out

    def is_success(self):
        done = True
        for i in range(0, len(self.current_states)):
            if (self.current_states[i] != self.goal_states[i]).any():
                done = False
        return done

    def generate_rollout(self, steps, best_action):
        done = False
        i = 0
        rollout = []
        while i < steps and not done:
            out_img = np.zeros(self.dims)
            out_img[self.current_states[0][0], self.current_states[0][1]] = 1
            out_img[self.current_states[1][0], self.current_states[1][1]] = 1
            obs = self.mapping @ out_img.flatten()
            action = best_action(obs)
            next_obs, reward, done, info = self.step(action)
            #TODO: make sure obs and next_obs are the coordinate points of the agents
            rollout.append((obs, action, reward, next_obs, done))
            i += 1
        return rollout

def format_rollout(batch, mapping):
    rollouts = []
    for i in range(0,len(batch)-1):
        action_int = batch[i].action[0]*4 + batch[i].action[1]
        #TODO: make sure first and fourth entry of tuple are the coordinate points of the agents
        rollouts.append((mapping @ batch[i].image, action_int, batch[i].reward[0] + batch[i].reward[1], mapping @ batch[i+1].image, False))
    return rollouts

if __name__ == "__main__":
    nfq_net = NFQNetwork()
    optimizer = optim.Rprop(nfq_net.parameters())
    nfq_agent = NFQAgent(nfq_net, optimizer)
    mapping = np.loadtxt("weights.txt")
    init_states = np.array([[1,1],[8,8]])
    goal_states = np.array([[8,8],[1,1]])
    ma = Two_Agent_Exchange_Location_Scenario(goal_states)
    batch, X = ma.simulate_function(10,
                                    200)  # give all images X dim [100 {flattened image vector of 10x10 grid}, 25 {num images}]
    rollouts = format_rollout(batch, mapping)
    sim = QLearner_Sim(goal_states, init_states, step, reward_fcn, mapping)
    for i in range(0, 100):

        state_action_b, target_q_values = nfq_agent.generate_pattern_set(rollouts)
        loss = nfq_agent.train((state_action_b, target_q_values))



        eval_episode_length, eval_success, eval_episode_cost = nfq_agent.evaluate(
            sim, False
        )

        if i % 10 == 0:
            print("epoch: {}, cost: {}, loss:{}".format(i, eval_episode_cost, loss))
        rollouts.extend(sim.generate_rollout(sim.max_steps, nfq_agent.get_best_action))



class QLearner:

    def __init__(self, simulation, mapping, actions):
        self.sim = simulation
        self.map = mapping
        self.actions = actions
        self.qtable = dict()



    def train(self, epochs, steps, epsilon, learning_rate):

        for _ in epochs:
            state = self.simulation.initialize_state()
            self.update_qtable(state)
            for s in steps:

                act = self.select_action(state, epsilon)
                new_state, reward, done = self.simulation.step(act)
                self.update_qtable(new_state)
                if done:
                    break


                self.qtable[state][act] = reward + learning_rate * max(self.qtable[new_state])
                    

                state = new_state

    def update_qtable(self, state):
        if state not in self.qtable.keys():
            self.qtable[state] = {}
            for a in self.actions:
                self.qtable[state][a] = np.random.random()

    def select_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            return self.actions[np.random.randint(0, len(self.actions))]
        else:
            max_val = 0
            max_act = None
            for a in self.actions:
                if max_act:
                    if self.qtable[state][a] > max_val:
                        max_act= a
                        max_val = self.qtable[state][a]
                else:
                    max_act = a
                    max_val = self.qtable[state][a]
            return a

