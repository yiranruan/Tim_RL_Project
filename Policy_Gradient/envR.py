#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Maze import generateMaze
from Maze import copy_grid
import tensorflow as tf
import random
import sys
import time
import numpy as np
# from cnn import cnn


seed = None


class envR:
    def __init__(self,
                 rows=10,
                 cols=10,
                 n_features=10
                 ):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = n_features
        self.rows = rows
        self.cols = cols
        self.seed = None
        if len(sys.argv) > 1:
            seed = int(sys.argv[1])
        self.maze = generateMaze(self.rows, self.cols, self.seed)
        self.save_grid = copy_grid(self.maze.grid)
        # self.total_cost = 0

    def reset(self):
        # print('reset',self.grid)

        # fixed grid
        # self.maze.grid = copy_grid(self.save_grid)

        # random grid
        self.maze = generateMaze(self.rows, self.cols, self.seed)

        self.agent = self.maze.get_start()
        # self.maze.random_reset()
        # self.maze.reset()

        # self.total_cost = 0
        # return self._cnn(self.get_maps())  # !!!

    def update_map(self, s, a, s_, train):
        self.maze.pass_by(s, a)
        self.maze.next_step(s_)
        # if not train:
        #     print("action:", a)
        #     print(str(self.maze))
        # time.sleep(1)

    # def _move(self, action):
    #     row, col = self.agent
    #     if action == 'u':
    #         if self.maze.isWall(row-1, col):s_ = (row-1, col)
    #     elif action == 'd':
    #         if self.maze.isWall(row+1, col):s_ = (row+1, col)
    #     elif action == 'l':
    #         if self.maze.isWall(row, col-1):s_ = (row, col-1)
    #     elif action == 'r':
    #         if self.maze.isWall(row, col+1):s_ = (row, col+1)

    def step(self, action, train=False):
        action = self.action_space[action]
        row, col = self.agent
        # done = False
        s = s_ = self.agent

        if action == 'u':
            if not self.maze.isWall(row-1, col):
                s_ = (row-1, col)
        elif action == 'd':
            if not self.maze.isWall(row+1, col):
                s_ = (row+1, col)
        elif action == 'l':
            if not self.maze.isWall(row, col-1):
                s_ = (row, col-1)
        elif action == 'r':
            if not self.maze.isWall(row, col+1):
                s_ = (row, col+1)

        # if self.maze.isTerminal(s_):
        #     done = True
        done = self.maze.isTerminal(s_)
        reward = self.maze.get_reward(s, s_)
        # print("tut",action, self.agent, s_)
        # self.total_cost += reward
        self.update_map(self.agent, action, s_, train)
        self.agent = s_
        self.maze.set_position(s_)
        # return self._cnn(self.get_maps()), reward, done
        return reward, done, action

    def get_maps(self):
        game_map = np.array(list(self.maze.getWalls())).astype(int)
        # food = np.array(list(self.maze.getFoods())).astype(int) * 5
        food1 = np.array(list(self.maze.getRealFood())).astype(int) * 2
        food2 = np.array(list(self.maze.getFakeFood())).astype(int) * 3
        game_map += food1 + food2
        game_map += np.array(list(self.maze.getPos())).astype(int) * 9
        return game_map

    # def _cnn(self, image):
    #     # return cnn(image, self.n_features)
    #     x = tf.placeholder(tf.float32, [16, 32])
    #     img = tf.reshape(x, [-1, 16, 32, 1])
    #     y = tf.placeholder(tf.float32, [None, self.n_features])
    #
    #     conv1 = tf.layers.conv2d(
    #         img,
    #         filters=32,
    #         kernel_size=[4, 4],
    #         padding='SAME',
    #         activation=tf.nn.relu)
    #
    #     pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
    #
    #     conv2 = tf.layers.conv2d(
    #         pool1,
    #         filters=64,
    #         kernel_size=[4, 4],
    #         padding='SAME',
    #         activation=tf.nn.relu)
    #
    #     pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
    #
    #     pool2_flat = tf.reshape(pool2, [-1, 4*8*64])
    #     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    #     drop = tf.layers.dropout(inputs=dense, rate=0.4)
    #
    #     logits = tf.layers.dense(inputs=drop, units=self.n_features)
    #
    #     # loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    #     # train = tf.train.AdamOptimizer(0.001),minimize(loss)
    #
    #     sess = tf.Session()
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(init_op)
    #     train = sess.run(logits, {x: image})
    #     return train


if __name__ == '__main__':
    env = envR()
    env.reset()
    print(env.get_maps())
    print(str(env.maze))
    a = input("pause")
    for _ in range(1000):
        # print(env.get_maps())
        # print(len(env.get_maps().flatten()))
        action = random.choice(env.action_space)
        print(str(env.maze))
        env.step(action)
        time.sleep(0.1)
    # print(env.reset())
