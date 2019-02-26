"""
Reinforcement learning environment_Beta

Red rectangle:          agent.
Black rectangles:       blocks.
Yellow bin circle:      goals.
All other states:       ground.

Output of step method: state: [中心点左上角x，y，右下角x，右下角y]
                        state: [横坐标，纵坐标]
                        reward: integer
                        done: boolean

Play agent like playing CS: w,a,s,d
"""

from queue import PriorityQueue
import numpy as np
import tkinter as tk
import time

UNIT = 40  # pixels ？？？
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width


class Maze(tk.Tk, object):  # 继承类Tk()
    def __init__(self, mode=1):
        # super(Maze, self) 首先找到 Maze 的父类（就是类 Tk）
        # 然后把类B的对象 Maze 转换为类 Tk 的对象
        # 该方法的目的多用于处理多继承

        # create origin（原点坐标；因为第一个格子的大小是40x40）
        # 生命子类独有的attribute时要放super（）前面
        self.origin = np.array([20, 20])

        super(Maze, self).__init__()  # 继承Tk（）。
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        # self.n_features = 2  # 特征
        # 窗口名
        self.title('maze')
        # 定义窗口（地图）的高和宽
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        # 建立地图
        if mode == 1:
            self._build_maze_01()
        elif mode == 2:
            self._build_maze_02()

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
        # create goal1
        self.goal_01 = self.origin + np.array([UNIT * 4, UNIT * 2])
        # goal2
        self.goal_02 = self.origin + np.array([UNIT * 2, UNIT * 4])
        # goal3
        self.goal_03 = self.origin + np.array([UNIT * 1, UNIT * 5])
        # goal list
        self.Goal = [tuple(self.goal_01), tuple(self.goal_02), tuple(self.goal_03)]

        # display fake goals；真假目标点不可能移动，就这样
        self.temp_goal = []
        for index in self.Goal[1:]:
            self.temp_goal.append(self.canvas.create_oval(
                index[0] - 15, index[1] - 15,
                index[0] + 15, index[1] + 15,
                fill='yellow'))

        # real goal
        self.real_goal = self.canvas.create_oval(
            self.Goal[0][0] - 15, self.Goal[0][1] - 15,
            self.Goal[0][0] + 15, self.Goal[0][1] + 15,
            fill='yellow')

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Block + self.Goal
        # truthful area
        self.truthful_nodes = []

        # create red agent；agent只有对角线坐标，没有中心坐标
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        # Initialize state: target_node, real_goal
        self.state = [False, False]

        # Total cost
        self.total_cost = 0

        # pack all
        self.canvas.pack()

    def _build_maze_02(self):
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
        self.Goal = []
        for index in range(2):
            temp = np.random.randint(0, MAZE_H, size=2)
            while temp[0] + temp[1] == 0:
                temp = np.random.randint(0, MAZE_H, size=2)
            self.Goal.append(self.origin + np.array([UNIT * temp[0], UNIT * temp[1]]))

        # real goal
        self.real_goal = self.canvas.create_oval(self.Goal[0][0] - 15, self.Goal[0][1] - 15,
                                                 self.Goal[0][0] + 15, self.Goal[0][1] + 15,
                                                 fill='blue')
        # fake goal
        self.fake_goal = self.canvas.create_oval(self.Goal[1][0] - 15, self.Goal[1][1] - 15,
                                                 self.Goal[1][0] + 15, self.Goal[1][1] + 15,
                                                 fill='yellow')

        # # customized
        # self.Goal = []
        # # create goal1
        # self.goal_01 = self.origin + np.array([UNIT * 0, UNIT * 5])
        # # goal2
        # self.goal_02 = self.origin + np.array([UNIT * 4, UNIT * 4])
        #
        # # goal list
        # self.Goal = [tuple(self.goal_01), tuple(self.goal_02)]
        #
        # # real goal
        # self.real_goal = self.canvas.create_oval(self.Goal[0][0] - 15, self.Goal[0][1] - 15,
        #                                          self.Goal[0][0] + 15, self.Goal[0][1] + 15,
        #                                          fill='blue')
        # # fake goal
        # self.fake_goal = self.canvas.create_oval(self.Goal[1][0] - 15, self.Goal[1][1] - 15,
        #                                          self.Goal[1][0] + 15, self.Goal[1][1] + 15,
        #                                          fill='yellow')

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Goal
        # truthful area
        self.truthful_nodes = []

        # create red agent；agent只有对角线坐标，没有中心坐标
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        # Initialize state: target_node, real_goal
        self.state = [False, False]

        # Total cost
        self.total_cost = 0

        # pack all
        self.canvas.pack()

    def step2_1(self, event):
        # action = event.char
        action = event
        # 返回矩形的参数;其实是返回坐标
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if action == 'u':  # up，并且检查是否超出边界
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 'd':  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'r':  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 'l':  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 检查是否撞墙 commented for mode 2 because no Block (change)
        # if (s[0] + 15 + base_action[0], s[1] + 15 + base_action[1]) in self.Block:
        #     base_action = np.array([0, 0])

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state

        # print(s_, reward, done, self.state)
        return self.getReward(s, s_)

    def getReward(self, s, s_):
        # function approximation
        rectx, recty = int((s[0] - 5) / 40), int((s[1] - 5) / 40)
        rectx_, recty_ = int((s_[0] - 5) / 40), int((s_[1] - 5) / 40)
        real_goalx, real_goaly = int((self.canvas.coords(self.real_goal)[
                                     0] - 5) / 40), int((self.canvas.coords(self.real_goal)[1] - 5) / 40)

        fake_goalx, fake_goaly = int((self.canvas.coords(self.fake_goal)[
                                     0] - 5) / 40), int((self.canvas.coords(self.fake_goal)[1] - 5) / 40)
        # manhattan distance
        # previous state s
        pre_dist_to_real = abs(rectx - real_goalx) + abs(recty - real_goaly)
        pre_dist_to_fake = abs(rectx - fake_goalx) + abs(recty - fake_goaly)

        # next state s_
        dist_to_real = abs(rectx_ - real_goalx) + abs(recty_ - real_goaly)
        dist_to_fake = abs(rectx_ - fake_goalx) + abs(recty_ - fake_goaly)
        dist_between_goal = abs(fake_goalx - real_goalx) + abs(fake_goaly - real_goaly)

        done = False
        reward = 0

        # moving towards goal
        if pre_dist_to_real > dist_to_real:
            reward += 0.1
        else:
            reward -= 0.1

        # check whether the new state is truthful 這裡默認了出生點為(0,0)
        truthful = dist_to_real - (real_goalx + real_goaly) < dist_to_fake - \
            (fake_goalx + fake_goaly)
        if truthful:
            reward -= 0.1
        else:
            reward += 0.1

        if s_ == self.canvas.coords(self.real_goal):
            done = True

            """
            change between big reward or small reward to real goal
            """
            # big reward
            # reward += (MAZE_H + MAZE_W) / 2 * 0.1

            # small reward
            reward += 0.1
        # print(reward, done)
        return reward, done

    # get the coordinate of rectangle (agent)
    def getRect(self):
        return self.canvas.coords(self.rect)

    # get goal for mode 1
    def getGoal1(self):
        goals = []
        goals += self.temp_goal
        goals.append(self.real_goal)
        return [self.canvas.coords(x) for x in goals]

    # get goal for mode 2
    def getGoal2(self):
        goals = []
        goals.append(self.fake_goal)
        goals.append(self.real_goal)
        return [self.canvas.coords(x) for x in goals]

    def render(self):
        time.sleep(0.1)
        self.update()


"""
两种模式 mode1 和 mode2，更改模式的时候记得把底下的a一起注解掉
mode2 目前无法开probability
"""

if __name__ == '__main__':
    env = Maze(mode=2)
    a = env.bind('<Key>', env.step2_1)
    env.mainloop()
