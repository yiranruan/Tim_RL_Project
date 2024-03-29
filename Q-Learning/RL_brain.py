import numpy as np
import pandas as pd
import time
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = epsilon
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)

    def choose_action(self, s, train):
        self.isStateExist(s)
        if train:
            if np.random.uniform() < self.e:
                state_action = self.q_table.loc[s,:]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                action = np.random.choice(self.actions)
            return action
        else:
            state_action = self.q_table.loc[s,:]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            return action
    
    def learn(self, s, a, r, s_, done):
        self.isStateExist(s_)
        q_predict = self.q_table.loc[s, a]
        if done != True:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def isStateExist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )