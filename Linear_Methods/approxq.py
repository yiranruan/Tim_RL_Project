import approxq_env
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import PolynomialFeatures


class FeatureExtractors():
    def __init__(self, n_features=2, n_polyfeatures=2):
        self.n_features = n_features
        self.features = np.zeros((1, self.n_features))

        # degree
        self.poly = PolynomialFeatures(n_polyfeatures)

        # this will include bias feature
        # array([bias, feature_1, feature_2,...,feature_n, polynomial features])
        self.features = self.poly.fit_transform(self.features)[0]

    def getFeatures(self, state, action):
        # state of goals
        goals = state[0:-1]
        fake_goalx, fake_goaly = state[0]
        real_goalx, real_goaly = state[-2]
        rectx, recty = state[-1]

        # the number of goal faced
        face = 0

        # get the position of next state
        if action == 'u':  # up，并且检查是否超出边界
            if recty > 0:
                for goal in goals:
                    if goal[1] < recty:
                        face += 1
                recty -= 1
        elif action == 'd':  # down
            if recty < approxq_env.MAZE_H - 1:
                for goal in goals:
                    if goal[1] > recty:
                        face += 1
                recty += 1
        elif action == 'r':  # right
            if rectx < approxq_env.MAZE_W - 1:
                for goal in goals:
                    if goal[0] > recty:
                        face += 1
                rectx += 1
        elif action == 'l':  # left
            if rectx > 0:
                for goal in goals:
                    if goal[0] < recty:
                        face += 1
                rectx -= 1

        # make number less than 1
        c = approxq_env.MAZE_H + approxq_env.MAZE_W

        # the distance to the real goal
        dist_to_real = abs(rectx - real_goalx) + abs(recty - real_goaly)
        feature1 = dist_to_real / c
        temp = np.array(feature1)

        # the distance to fake goal
        dist_to_fake = abs(rectx - fake_goalx) + abs(recty - fake_goaly)
        feature2 = dist_to_fake / c
        temp = np.append(temp, feature2)

        # the distance between the fake goal and real goal
        dist_between_goal = abs(fake_goalx - real_goalx) + abs(fake_goaly - real_goaly)
        feature3 = dist_between_goal / c
        temp = np.append(temp, feature3)

        # facing how many goals
        feature4 = face / len(goals)
        # temp = np.append(temp, feature4)

        # stay in the middle of fake goal and real goal (to be deceptive)
        # this is like manually picking a feature that I think is deceptive, not so interesting
        if dist_to_fake + dist_to_real - dist_between_goal == 0 or dist_to_fake - dist_to_real == 0:
            feature5 = 0
        else:
            feature5 = abs(dist_to_fake - dist_to_real) / c
        # temp = np.append(temp, feature5)

        # get the polynomial features
        self.features = self.poly.fit_transform(temp.reshape(1, self.n_features))[0]



        return self.features


"""
the following parameters need to be changed before every run:
n_features -> the number of features
n_polyfeatures -> the number of polyfeatures, this will include bias
self.r -> render or not
self.u -> update or not
need to change the comment to create new weights or load from file
need to change the save file at the end of the script
need to modify the features in FeatureExtractors class as needed
"""


class TrainingAgent():
    def __init__(self):
        self.train_episode = 10000
        self.max_steps = 100
        self.learning_rate = 0.1
        self.gamma = 0.8
        self.epsilon = 0.00
        self.featureExtract = FeatureExtractors(n_features=3, n_polyfeatures=1)
        self.r = True  # render or not
        self.u = False   # update or not
        '''
        self.weights
        array([bias, feature_1, feature_2,...,feature_n, polynomial features])
        '''
        # create new
        # self.weights = np.full(shape=self.featureExtract.features.shape,
        #                        fill_value=-0.1, dtype='float32')

        # load from file (change)
        self.weights = np.load("f123d1v1.npy")

    def chooseAction(self, state):
        if random.uniform(0, 1) > self.epsilon:
            action = self.getQValues(state).idxmax()
        else:
            action = random.choice(self.getLegalActions(state))
        return action

    # switch between mode (change)
    def reset(self):
        self.env = approxq_env.Maze(mode=2)

    # state is the coordinates of rect and goals
    # return a tuple (goals coords, rect coords)
    def getState(self):
        rect = self.env.getRect()
        rect_c = self.getCoords(rect)

        # use for mode 2 (change)
        # goals = (fake, real)
        goals = self.env.getGoal2()
        goals_c = [self.getCoords(goal) for goal in goals]

        state = []
        state += goals_c
        state.append(rect_c)

        # (fake, real, rect)
        return tuple(state)

    def getCoords(self, state):
        # input state is a list of coordinate from the environment
        # hard code the pixels 5 and 40
        """
        state value of a maze
        ------------
        | 00 10 20 |
        | 01 11 21 |
        | 02 12 22 |
        ------------
        """
        return int((state[0] - 5) / 40), int((state[1] - 5) / 40)

    def update(self, state, action, reward, new_state, done):
        qvalues = self.getQValues(state)

        if done:
            corrections = reward - qvalues[action]
        else:
            new_qvalues = self.getQValues(new_state)
            corrections = reward + self.gamma * new_qvalues.max() - qvalues[action]
        self.weights += self.learning_rate * corrections * \
            self.featureExtract.getFeatures(state, action)

    def getQValues(self, state):
        actions = self.getLegalActions(state)
        qvalues = [self.featureExtract.getFeatures(
            state, act).dot(self.weights) for act in actions]

        #   return qvalues of each action
        return pd.Series(qvalues, index=actions, dtype='float32')

    def getLegalActions(self, state):
        rectx, recty = state[-1]
        actions = self.env.action_space.copy()

        # cannot move out the maze
        if recty == 0:
            actions.remove('u')
        elif recty == approxq_env.MAZE_H:
            actions.remove('d')
        if rectx == approxq_env.MAZE_W:
            actions.remove('r')
        elif rectx == 0:
            actions.remove('l')
        return actions

    def train(self):
        for episode in range(self.train_episode):
            print("episode:", episode)
            self.reset()
            step = 0
            done = False
            for step in range(self.max_steps):
                state = self.getState()
                action = self.chooseAction(state)
                reward, done = self.env.step2_1(action)
                new_state = self.getState()
                # print(state, action, reward, new_state)

                if self.u:
                    self.update(state, action, reward, new_state, done)

                # If done : finish episode
                if done:
                    print(step)
                    break

                if episode % 100 == 0:
                    print(episode, self.weights)

                # refresh env
                if self.r:
                    self.env.render()

            self.env.destroy()


if __name__ == '__main__':
    noob = TrainingAgent()
    noob.train()
    # np.save("f12d1.npy", noob.weights)
    print("weights", noob.weights)

