
# from env_beta_view import Maze_view
from queue import PriorityQueue
import numpy as np

UNIT = 1  # pixels
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width


def manhattan_distance(start, goal):
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])


class Maze:
    def __init__(self, mode=1,
                 x_real=None,
                 y_real=None,
                 x_fake=None,
                 y_fake=None,
                 height=6,
                 width=6):
        self.origin = np.array([0, 0])

        global MAZE_H, MAZE_W
        MAZE_H = height
        MAZE_W = width

        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.state = 0

        # 建立地图
        if mode == 1:
            self._build_maze_01()
        elif mode == 2:
            self._build_maze_02(x_real, y_real)
        elif mode == 3:
            self._build_maze_03(x_real=x_real, y_real=y_real,
                                x_fake=x_fake, y_fake=y_fake)

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

        # create red agent（yourself）；
        self.agent = self.origin

        # Total cost
        self.total_cost = 0

    def _build_maze_02(self, x_real, y_real):

        # ----------goals------------------------------------------ #
        self.real_goal = self.origin + np.array([UNIT * x_real, UNIT * y_real])
        self.Goal = [tuple(self.real_goal)]

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Goal

        # create red agent；
        self.agent = np.array([0, 0])

        # Total cost
        self.total_cost = 0

    def _build_maze_03(self, x_real, y_real, x_fake, y_fake):

        # ----------goals------------------------------------------ #
        self.real_goal = self.origin + np.array([UNIT * x_real, UNIT * y_real])
        self.fake_goal = self.origin + np.array([UNIT * x_fake, UNIT * y_fake])
        self.Goal = [tuple(self.real_goal), tuple(self.fake_goal)]

        # -----------------整合-------------------------------------------#
        self.meaningful_area = self.Goal
        # truthful area
        self.truthful_steps_area = []

        # create red agent；
        self.agent = np.array([0, 0])

        # Initialize state: target_node, real_goal
        # self.state = [0, 0]

        # Total cost (single episode)
        self.total_cost = 0

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
            reward = -1  # 每走一步，付出1个reward的代价

        self.total_cost += reward
        # 改造成显示给人和神经网络看的坐标
        # s_ = ((s_[0] - 5) / unit, (s_[1] - 5) / unit)
        print(s_, reward, self.state, self.total_cost)

    def step_02(self, action):
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
            reward = -self.optimal_distance(self.agent, self.real_goal) + 500
            done = True
        else:
            reward = -self.optimal_distance(self.agent, self.real_goal)
            done = False

        self.total_cost += reward

        s_ = np.hstack((self.real_goal, self.agent))

        return s_, reward, done

    def step_03(self, action):
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

        self.agent += base_action

        s_ = self.agent

        # state judgement
        if self.state == 0:
            reward = -manhattan_distance(s_, self.fake_goal)
            if (s_ == self.fake_goal).all():
                self.state = 1
        else:
            reward = -manhattan_distance(s_, self.real_goal)
            if (s_ == self.real_goal).all():
                self.state = 2

        self.total_cost += reward

        s_ = np.hstack((self.state/2, self.fake_goal/(MAZE_H-1), self.real_goal/(MAZE_H-1), self.agent/(MAZE_H-1)))

        return s_, reward, self.state

    # ---------------------Reset part-------------------------------------

    def reset_02(self):

        self.agent = np.array([0, 0])

        self.total_cost = 0

        state = np.hstack((self.real_goal, self.agent))

        return state

    def reset_03(self):
        self.agent = np.array([0, 0])
        self.state = 0
        self.total_cost = 0

        state = np.hstack((self.state/2, self.fake_goal/(MAZE_H-1), self.real_goal/(MAZE_H-1), self.agent/(MAZE_H-1)))

        return state

    # ---------------------Tools part--------------------------------------

    def neighbors(self, node):
        neighbor_nodes = []

        if node[1] > 0:  # up
            neighbor_nodes.append((node[0], node[1] - UNIT))
        if node[0] < MAZE_W * UNIT - UNIT:  # right
            neighbor_nodes.append((node[0] + UNIT, node[1]))
        if node[1] < MAZE_H * UNIT - UNIT:  # down
            neighbor_nodes.append((node[0], node[1] + UNIT))
        if node[0] > 0:  # left
            neighbor_nodes.append((node[0] - UNIT, node[1]))

        # # 检查周围是否有墙
        # for block in self.Block:
        #     if block in neighbor_nodes:
        #         neighbor_nodes.remove(block)

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

        for node_x in range(MAZE_W):
            for node_y in range(MAZE_H):
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
                        self.truthful_steps_area.append(node)

        a, b, c = opt_s_g[0], opt_s_g[1], self.optimal_distance(self.Goal[0], self.Goal[1])
        self.beta = (c+a-b)/2
        self.R_M_P = []

        for node in self.truthful_steps_area:
            if self.optimal_distance(node, self.Goal[0]) <= self.beta:
                self.R_M_P.append(node)


if __name__ == '__main__':
    env01 = Maze(mode=3, x_real=3, y_real=15, x_fake=17, y_fake=13, height=19, width=19)
    # env = Maze_view(mode=3, x_real=0, y_real=6, x_fake=6, y_fake=3, height=8, width=8)
    # env.pre_work()
    # a = env.bind('<Key>', env.move_play_01)

    # env = Maze(mode=2, x_real=5, y_real=5, height=10, width=10)
    # print(env.reset())
    step = [1, 1, 1, 1, 2]
    # env = Maze(mode=3)
    # print(env.step_03(3))

    for i in step:
        print(env01.step_03(i))
    print(env01.reset_03())

    # for i in step:
    #     env.step_02(i)
    #     time.sleep(2)
    #     env.update()

    # env.mainloop()  # 实际就是维持并刷新窗口

