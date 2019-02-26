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

UNIT = 40  # pixels 
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width


class Maze_view(tk.Tk, object):  # 继承类Tk()
    def __init__(self, mode=1,
                 x_real=None,
                 y_real=None,
                 x_fake=None,
                 y_fake=None,
                 height=6,
                 width=6):
        # super(Maze, self) 首先找到 Maze 的父类（就是类 Tk）
        # 然后把类B的对象 Maze 转换为类 Tk 的对象
        # 该方法的目的多用于处理多继承

        # create origin（原点坐标；因为第一个格子的大小是40x40）
        # 生命子类独有的attribute时要放super（）前面
        self.origin = np.array([20, 20])

        global MAZE_H, MAZE_W
        MAZE_H = height
        MAZE_W = width

        super(Maze_view, self).__init__()  # 继承Tk（）。
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        # 窗口名
        self.title('maze')
        # 定义窗口（地图）的高和宽
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))

        # 建立地图
        if mode == 1:
            self._build_maze_01()
        elif mode == 2:
            self._build_maze_02(x_real, y_real)
        elif mode == 3:
            self._build_simple_strategy_one(x_real, y_real, x_fake, y_fake)

    def _build_maze_01(self):
        # 装载在Maze上的canvas
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,  # 这里的数值单位是pixel
                                width=MAZE_W * UNIT)  # 这样算是把白色canvas覆盖满整个地图

        # create grids；（画线）
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # ---------blocks----------------------------------------------- #
        # block1
        self.block1 = self.origin + np.array([UNIT * 2, UNIT])
        # block2
        self.block2 = self.origin + np.array([UNIT, UNIT * 2])
        # block list
        self.Block = [tuple(self.block1), tuple(self.block2)]

        # block1
        # 一个特殊格子大小大概是30（流出白边好识别）。这两个坐标是对角线上的两个点（左上，右下）
        self.block01 = self.canvas.create_rectangle(
            self.block1[0] - 15, self.block1[1] - 15,
            self.block1[0] + 15, self.block1[1] + 15,
            fill='black')
        # block2
        self.block02 = self.canvas.create_rectangle(
            self.block2[0] - 15, self.block2[1] - 15,
            self.block2[0] + 15, self.block2[1] + 15,
            fill='black')

        # ----------goals------------------------------------------ #
        # real goal
        self.real_goal = self.origin + np.array([UNIT * 4, UNIT * 2])
        # goal2
        self.goal_02 = self.origin + np.array([UNIT * 2, UNIT * 4])
        # goal3
        self.goal_03 = self.origin + np.array([UNIT * 1, UNIT * 5])
        # goal list
        self.Goal = [tuple(self.real_goal), tuple(self.goal_02), tuple(self.goal_03)]

        # real goal
        self.real_goal_can = self.canvas.create_oval(
            self.Goal[0][0] - 15, self.Goal[0][1] - 15,
            self.Goal[0][0] + 15, self.Goal[0][1] + 15,
            fill='blue')

        # display fake goals；真假目标点不可能移动，就这样
        self.Goal2 = []
        for index in self.Goal[1:]:
            self.Goal2.append(self.canvas.create_oval(
                index[0] - 15, index[1] - 15,
                index[0] + 15, index[1] + 15,
                fill='yellow'))

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Block + self.Goal
        # truthful area
        self.truthful_nodes = []

        # create red agent；agent只有对角线坐标，没有中心坐标
        self.agent = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        # Initialize state: target_node, real_goal
        self.state = [False, False]

        # Total cost
        self.total_cost = 0

        # pack all
        self.canvas.pack()

    def _build_maze_02(self, x_real, y_real):
        # 装载在Maze上的canvas
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,  # 这里的数值单位是pixel
                                width=MAZE_W * UNIT)  # 这样算是把白色canvas覆盖满整个地图

        # create grids；（画线）
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # ----------goals------------------------------------------ #
        self.real_goal = self.origin + np.array([UNIT * x_real, UNIT * y_real])
        self.Goal = [tuple(self.real_goal)]
        # for index in range(1):
        #     temp = np.random.randint(0, MAZE_H, size=2)
        #     while temp[0] + temp[1] == 0:
        #         temp = np.random.randint(0, MAZE_H, size=2)
        #     self.Goal.append(self.origin + np.array([unit * temp[0], unit * temp[1]]))

        # real goal
        self.real_goal_can = self.canvas.create_oval(self.Goal[0][0] - 15, self.Goal[0][1] - 15,
                                                     self.Goal[0][0] + 15, self.Goal[0][1] + 15,
                                                     fill='blue')
        # # fake goal
        # self.fake_goal = self.canvas.create_oval(self.Goal[1][0] - 15, self.Goal[1][1] - 15,
        #                                          self.Goal[1][0] + 15, self.Goal[1][1] + 15,
        #                                          fill='yellow')

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Goal
        # truthful area
        self.truthful_nodes = []

        # create red agent；agent只有对角线坐标，没有中心坐标
        self.agent = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        # Initialize state: target_node, real_goal
        self.state = [False, False]

        # Total cost
        self.total_cost = 0

        # pack all
        self.canvas.pack()

    # --------------------Deceptive part--------------------------------

    def _build_simple_strategy_one(self, x_real, y_real, x_fake, y_fake):
        # ----------------set background----------------------------------
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # ----------------set goal----------------------------------------

        self.goal = self.origin + np.array([UNIT * x_real, UNIT * y_real])

        self.real_goal = self.canvas.create_oval(self.goal[0] - 15, self.goal[1] - 15,
                                                 self.goal[0] + 15, self.goal[1] + 15,
                                                 fill='blue')

        self.fake = self.origin + np.array([UNIT * x_fake, UNIT * y_fake])

        self.canvas.create_oval(self.fake[0] - 15, self.fake[1] - 15,
                                self.fake[0] + 15, self.fake[1] + 15,
                                fill='yellow')

        # ------------------------agent---------------------------------
        self.agent = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        # ------------------------CT-----------------------------------
        self.canvas.pack()

    def step(self, action):
        # 返回矩形的参数;其实是返回坐标
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        if action == 0:  # up，并且检查是否超出边界
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 检查是否撞墙
        if (s[0] + 15 + base_action[0], s[1] + 15 + base_action[1]) in self.Block:
            base_action = np.array([0, 0])

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.agent)  # next state

        # reward function
        # 到达real goal
        if s_ == self.canvas.coords(self.real_goal):
            reward = 100
            done = True
            # s_ = True
        # 撞墙，原地踏步
        elif s_ == s:
            reward = 0
            done = False
        else:
            if [s_[0] + 15, s_[1] + 15] in self.truthful_nodes:
                reward = -2  # 不鼓励走到真实区，付出更大代价
                done = False
            else:
                reward = -1  # 每走一步，付出1个reward的代价
                done = False

        return s_, reward, done

    def reset(self):

        # 删除agent
        self.canvas.delete(self.agent)
        # 让agent回到原点
        self.agent = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        self.canvas.coords(self.real_goal_can)

        # 修改并回馈坐标；一定要这样
        state = (np.hstack((self.canvas.coords(self.real_goal_can)[:2],
                           self.canvas.coords(self.agent)[:2]))-5) / UNIT

        self.update()

        # cost计数归0
        self.total_cost = 0

        return state

    # ---------------------Test part---------------------------------- #

    def move_play_01(self, event):
        action = event.char

        s = self.canvas.coords(self.agent)
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
        if (s[0] + 15 + base_action[0], s[1] + 15 + base_action[1]) in self.Block:
            base_action = np.array([0, 0])

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.agent)

        if s_ == self.canvas.coords(self.real_goal_can):
            if self.state[0]:
                reward = 100
                self.state[1] = True
            else:
                reward = -100
                self.state[1] = True
        elif s_ == self.canvas.coords(self.Goal2[0]):
            reward = -1
            self.state[0] = True
        # 撞墙，原地踏步
        elif s_ == s:
            reward = 0
        else:
            if [s_[0] + 15, s_[1] + 15] in self.truthful_nodes:
                reward = -2  # 不鼓励走到真实区，付出更大代价
            else:
                reward = -1  # 每走一步，付出1个reward的代价

        self.total_cost += reward
        # 改造成显示给人和神经网络看的坐标
        s_ = ((s_[0] - 5) / UNIT, (s_[1] - 5) / UNIT)
        print(s_, reward, self.state, self.total_cost)

    def step_02(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        if action == 0:  # up，并且检查是否超出边界
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.agent)

        # 到达real goal
        if s_ == self.canvas.coords(self.real_goal_can):
            reward = 50
            done = True
        else:
            reward = -1
            done = False

        self.total_cost += reward

        # s_ = (np.array(self.canvas.coords(self.agent)[:2]) - 5) / unit
        s_ = (np.hstack((self.canvas.coords(self.real_goal_can)[:2],
                        self.canvas.coords(self.agent)[:2])) - 5) / UNIT

        return s_, reward, done

    def step_03(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        if action == 0:  # up，并且检查是否超出边界
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.agent)

        # 到达real goal
        if s_ == self.canvas.coords(self.real_goal):
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        s_ = (np.array(self.canvas.coords(self.agent)[:2]) - 5) / UNIT

        return s_, reward, done

    # ---------------------Deceptive step-------------------------------

    def step_04(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        if action == 0:  # up，并且检查是否超出边界
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        for index in self.colorful_area:
            if self.canvas.coords(self.agent) == self.canvas.coords(index):
                self.canvas.delete(index)

    def display(self, truthful_steps_area=None, R_M_P=None):

        self.colorful_area = []

        for index in truthful_steps_area:
            if index in R_M_P:
                self.colorful_area.append(self.canvas.create_rectangle(
                    5+index[0]*UNIT, 5+index[1]*UNIT,
                    35+index[0]*UNIT, 35+index[1]*UNIT,
                    fill='gray'))
            else:
                self.colorful_area.append(self.canvas.create_rectangle(
                    5+index[0]*UNIT, 5+index[1]*UNIT,
                    35+index[0]*UNIT, 35+index[1]*UNIT,
                    fill='green'))


if __name__ == '__main__':
    env = Maze_view(mode=3, x_real=0, y_real=6, x_fake=6, y_fake=3, height=8, width=8)
    env.pre_work()
    # a = env.bind('<Key>', env.move_play_01)

    # env = Maze(mode=2, x_real=5, y_real=5, height=10, width=10)
    # print(env.reset())
    # step = [1, 1, 1, 1, 2]
    # env = Maze(mode=3)
    # print(env.step_03(3))

    # for i in step:
    #     env.step_02(i)
    #     time.sleep(2)
    #     env.update()

    env.mainloop()  # 实际就是维持并刷新窗口
