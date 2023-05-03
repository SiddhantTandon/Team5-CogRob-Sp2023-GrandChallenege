import numpy as np

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

