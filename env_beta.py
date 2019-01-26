"""
Reinforcement learning environment_Beta
可视化说明
Red rectangle:          agent.
Black rectangles:       blocks.
Yellow bin circle:      false goals.
Blue bin circle:        real goal.
All other states:       ground.

Output of step method: state: [中心点左上角x，y，右下角x，右下角y]
                        state: [横坐标，纵坐标]
                        reward: integer
                        done: boolean

self.block1:            block的像素坐标
self.block01:           block在GUI上的坐标参数；也是在GUI上控制block的句柄
self.Block:             block的像素坐标的集合

self.real_goal:         real goal的像素坐标
self.real_goal_can:     real goal在GUI上的坐标参数
self.Goal:              goal的像素坐标集合
self.Goal2:             false goals的GUI句柄集合

self.agent:             agent的GUI坐标参数

mode2：
我有一个real goal
"""
import time
from queue import PriorityQueue
import numpy as np
import tkinter as tk

UNIT = 1  # pixels ？？？
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width


class Maze:
    def __init__(self, mode=1,
                 x_real=None,
                 y_real=None,
                 height=6,
                 width=6):
        self.origin = np.array([0, 0])

        global MAZE_H, MAZE_W
        MAZE_H = height
        MAZE_W = width

        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)

        # 建立地图
        if mode == 1:
            self._build_maze_01()
        elif mode == 2:
            self._build_maze_02(x_real, y_real)
        elif mode == 3:
            pass
            # self._build_maze_03()

    def _build_maze_01(self):

        # ---------blocks----------------------------------------------- #
        # block1
        self.block1 = self.origin + np.array([UNIT * 2, UNIT])
        # block2
        self.block2 = self.origin + np.array([UNIT, UNIT * 2])
        # block list
        self.Block = [tuple(self.block1), tuple(self.block2)]

        # ----------goals------------------------------------------ #
        # real goal
        self.real_goal = self.origin + np.array([UNIT * 4, UNIT * 2])
        # goal2
        self.goal_02 = self.origin + np.array([UNIT * 2, UNIT * 4])
        # goal3
        self.goal_03 = self.origin + np.array([UNIT * 1, UNIT * 5])
        # goal list
        self.Goal = [tuple(self.real_goal), tuple(self.goal_02), tuple(self.goal_03)]

        # fake goals list
        self.Goal2 = []
        for goal in self.Goal[1:]:
            self.Goal2.append(goal)

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Block + self.Goal
        # truthful area
        self.truthful_nodes = []

        # create red agent（yourself）；
        self.agent = self.origin

        # Initialize state: target_node, real_goal
        self.state = [False, False]

        # Total cost
        self.total_cost = 0

    def _build_maze_02(self, x_real, y_real):

        # ----------goals------------------------------------------ #
        self.real_goal = self.origin + np.array([UNIT * x_real, UNIT * y_real])
        self.Goal = [tuple(self.real_goal)]

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Goal
        # truthful area
        self.truthful_nodes = []

        # create red agent；
        self.agent = self.origin

        # Initialize state: target_node, real_goal
        self.state = [False, False]

        # Total cost
        self.total_cost = 0

    def reset(self):

        self.agent = self.origin

        self.total_cost = 0

        state = np.hstack((self.real_goal, self.agent))

        return state

    # ---------------------Test part---------------------------------- #

    def move_play_01(self, action):

        s = self.agent
        base_action = np.array([0, 0])

        if action == 'w':  # up，并且检查是否超出边界
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 's':  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'd':  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 'a':  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 检查是否撞墙
        if (s[0] + base_action[0], s[1] + base_action[1]) in self.Block:
            base_action = np.array([0, 0])

        # self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent
        self.agent = self.agent + base_action
        s_ = self.agent

        # 注意all的用法
        if (s_ == self.real_goal).all():
            if self.state[0]:
                reward = 100
                self.state[1] = True
            else:
                reward = -100
                self.state[1] = True
        elif (s_ == self.Goal2[0]).all():
            reward = -1
            self.state[0] = True
        # 撞墙，原地踏步
        elif (s_ == s).all():
            reward = 0
        else:
            if [s_[0] + 15, s_[1] + 15] in self.truthful_nodes:
                reward = -2  # 不鼓励走到真实区，付出更大代价
            else:
                reward = -1  # 每走一步，付出1个reward的代价

        self.total_cost += reward
        # 改造成显示给人和神经网络看的坐标
        # s_ = ((s_[0] - 5) / UNIT, (s_[1] - 5) / UNIT)
        print(s_, reward, self.state, self.total_cost)

    def step_02(self, action):
        # s = np.hstack((self.real_goal, self.agent))
        s = self.agent
        base_action = np.array([0, 0])

        if action == 0:  # up，并且检查是否超出边界
            if s[1] > 0:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > 0:
                base_action[0] -= UNIT

        self.agent = self.agent + base_action
        s_ = self.agent

        # 到达real goal
        if (s_ == self.real_goal).all():
            reward = 50
            done = True
        else:
            reward = -1
            done = False

        self.total_cost += reward

        s_ = np.hstack((self.real_goal, self.agent))

        return s_, reward, done


# if __name__ == '__main__':
#     env = Maze(mode=2, x_real=3, y_real=3)
#     print(env.step_02(0))
#     # print(env.step_02(1))
#     # print(env.step_02(2))
#     # print(env.step_02(3))
#     # print(env.reset())
