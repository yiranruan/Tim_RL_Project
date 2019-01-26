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

UNIT = 40  # pixels ？？？
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width


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

    # def reset(self):

    def step(self, action):
        # 返回矩形的参数;其实是返回坐标
        s = self.canvas.coords(self.rect)
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

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

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

    # def render(self):

    # ---------------------Test part---------------------------------- #

    def move_play_01(self, event):
        action = event.char

        s = self.canvas.coords(self.rect)
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

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.real_goal):
            if self.state[0]:
                reward = 100
                self.state[1] = True
            else:
                reward = -100
                self.state[1] = True
        elif s_ == self.canvas.coords(self.temp_goal[0]):
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
        s_ = ((s_[0] - 5) / UNIT, (s_[1] - 5) / UNIT)
        print(s_, reward, self.state, self.total_cost)

    def move_play_02(self, event):
        action = event.char

        s = self.canvas.coords(self.rect)
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

        # # 检查是否撞墙
        # if (s[0] + 15 + base_action[0], s[1] + 15 + base_action[1]) in self.Block:
        #     base_action = np.array([0, 0])

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)
        if s_ == self.canvas.coords(self.real_goal):
            print('You touch the real goal.')
        elif s_ == self.canvas.coords(self.fake_goal):
            print('You touch the fake goal.')

        # if s_ == self.canvas.coords(self.real_goal):
        #     if self.state[0]:
        #         reward = 100
        #         self.state[1] = True
        #     else:
        #         reward = -100
        #         self.state[1] = True
        # elif s_ == self.canvas.coords(self.temp_goal[0]):
        #     reward = -1
        #     self.state[0] = True
        # # 撞墙，原地踏步
        # elif s_ == s:
        #     reward = 0
        # else:
        #     if [s_[0] + 15, s_[1] + 15] in self.truthful_nodes:
        #         reward = -2  # 不鼓励走到真实区，付出更大代价
        #     else:
        #         reward = -1  # 每走一步，付出1个reward的代价
        #
        # self.total_cost += reward
        s_ = ((s_[0] - 5) / UNIT, (s_[1] - 5) / UNIT)
        # print(s_, reward, self.state, self.total_cost)
        print(s_)

    def neighbors(self, node):
        neighbor_nodes = []

        if node[1] > 20:
            neighbor_nodes.append((node[0], node[1] - UNIT))
        if node[0] < MAZE_W * UNIT - 20:
            neighbor_nodes.append((node[0] + UNIT, node[1]))
        if node[1] < MAZE_H * UNIT - 20:
            neighbor_nodes.append((node[0], node[1] + UNIT))
        if node[0] > 20:
            neighbor_nodes.append((node[0] - UNIT, node[1]))

        # 检查周围是否有墙
        for block in self.Block:
            if block in neighbor_nodes:
                neighbor_nodes.remove(block)

        neighbor_nodes_02 = []

        for valid_node in neighbor_nodes:
            neighbor_nodes_02.append(list(valid_node))

        return neighbor_nodes_02

    # 求得两点之间最短距离
    def optimal_distance(self, start, goal):
        frontier = PriorityQueue()
        # 0代表距离；也是priority的依据
        frontier.put(start, 0)

        came_from = {}
        cost_so_far = {}

        # 将start转成tuple，作为字典的键
        start_ = tuple(start)
        goal_ = tuple(goal)
        came_from[start_] = None
        cost_so_far[start_] = 0

        while not frontier.empty():
            current = frontier.get()
            current_ = tuple(current)
            # 找到目标点的时候
            if current_ == goal_:
                break
            # 这里是做search
            for nextOne in self.neighbors(current):
                nextOne_ = tuple(nextOne)
                new_cost = cost_so_far[current_] + 1  # 每步的cost都是1
                if nextOne_ not in cost_so_far or new_cost < cost_so_far[nextOne_]:
                    cost_so_far[nextOne_] = new_cost
                    priority = new_cost
                    frontier.put(nextOne, priority)
                    came_from[nextOne_] = current

        return cost_so_far[goal_]

    def pre_work(self):
        opt_s_g = []
        costdif_g = {}

        for goal in self.Goal:
            opt_s_g.append(self.optimal_distance(self.origin, goal))

        for node_x in range(20, MAZE_W * UNIT, UNIT):
            for node_y in range(20, MAZE_H * UNIT, UNIT):
                node = [node_x, node_y]
                if tuple(node) not in self.meaningful_area:

                    opt_n_g = []
                    for goal in self.Goal:
                        opt_n_g.append(self.optimal_distance(node, goal))

                    for goal in range(len(self.Goal)):
                        costdif_g[goal + 1] = opt_n_g[goal] - opt_s_g[goal]
                    # 按元素对不同的goal进行排序
                    rank = sorted(costdif_g.items(), key=lambda item: item[1])

                    if rank[0][0] == 1 and rank[0][1] != rank[1][1]:
                        self.truthful_nodes.append(node)

        for index in self.truthful_nodes:
            self.canvas.create_rectangle(
                index[0] - 15, index[1] - 15,
                index[0] + 15, index[1] + 15,
                fill='green')


"""
两种模式 mode1 和 mode2，更改模式的时候记得把底下的a一起注解掉
mode2 目前无法开probability
"""

if __name__ == '__main__':
    # env = Maze(mode=1)
    # a = env.bind('<Key>', env.move_play)

    env = Maze(mode=2)
    a = env.bind('<Key>', env.move_play_02)

    # 划掉这个注解开启probability模式
    # env.pre_work()

    env.mainloop()
