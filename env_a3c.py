from queue import PriorityQueue
import numpy as np
import tkinter as tk
import time

UNIT = 40  # pixels ？？？
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width


class Maze(tk.Tk, object):
    def __init__(self, mode=1):

        self.origin = np.array([20, 20])
        super(Maze, self).__init__()
        self.action_space = ['0', '1', '2', '3']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))

        if mode == 1:
            self._build_maze_01()
        elif mode == 2:
            self._build_maze_02()
        # elif mode == 3:
        #     self._build_maze_03()

    def _build_maze_01(self):
        def _build_maze_01(self):
            self.canvas = tk.Canvas(self, bg='white',
                                    height=MAZE_H * UNIT,
                                    width=MAZE_W * UNIT)

            # create grids；
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

            # display fake goals；
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

            self.meaningful_area = self.Block + self.Goal
            # truthful area
            self.truthful_nodes = []

            # create red agent；
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

        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids；
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        # ----------goals------------------------------------------ #
        # create goal1


        # temp = np.random.randint(1, MAZE_H, size=6)
        # while temp.any() == 0:
        #     temp = np.random.randint(1, MAZE_H, size=6)
        #
        # self.goal_01 = self.origin + np.array([UNIT * temp[0], UNIT * temp[1]])
        # self.goal_02 = self.origin + np.array([UNIT * temp[2], UNIT * temp[3]])
        #
        # self.Goal = [tuple(self.goal_01), tuple(self.goal_02)]
        # # display fake goals;
        # # self.temp_goal = []
        # # for index in self.Goal[1:]:
        # #     self.temp_goal.append(self.canvas.create_oval(
        # #         index[0] - 15, index[1] - 15,
        # #         index[0] + 15, index[1] + 15,
        # #         fill='green'))
        # #
        # # real goal
        # self.real_goal = self.canvas.create_oval(
        #     self.Goal[0][0] - 15, self.Goal[0][1] - 15,
        #     self.Goal[0][0] + 15, self.Goal[0][1] + 15,
        #     fill='blue')
        # #
        # # fake goal
        # self.fake_goal = self.canvas.create_oval(self.Goal[1][0] - 15, self.Goal[1][1] - 15,
        #                                          self.Goal[1][0] + 15, self.Goal[1][1] + 15,
        #                                          fill='yellow')

        # customized
        self.Goal = []
        # create goal1
        self.goal_01 = self.origin + np.array([UNIT * 1, UNIT * 7])
        # goal2
        self.goal_02 = self.origin + np.array([UNIT * 2, UNIT * 6])

        # goal list
        self.Goal = [tuple(self.goal_01), tuple(self.goal_02)]

        # real goal
        self.real_goal = self.canvas.create_oval(self.Goal[0][0] - 15, self.Goal[0][1] - 15,
                                                 self.Goal[0][0] + 15, self.Goal[0][1] + 15,
                                                 fill='blue')
        # fake goal
        self.fake_goal = self.canvas.create_oval(self.Goal[1][0] - 15, self.Goal[1][1] - 15,
                                                 self.Goal[1][0] + 15, self.Goal[1][1] + 15,
                                                 fill='yellow')


#        self.meaningful_area = self.Block + self.Goal
        # truthful area
        self.truthful_nodes = []

        # create red agent；
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='red')

        # # Initialize state: target_node, real_goal
        # self.state = [False, False]
        #
        # # Total cost
        # self.total_cost = 0

        # pack all
        self.canvas.pack()
        #print(self.canvas.coords(self.rect))

    def getState(self):
        s = self.canvas.coords(self.rect)
        if s == self.real_goal:
            s.append(0)
            s.append(1)

        if s == self.fake_goal:
            s.append(1)
            s.append(0)

        if s != self.real_goal and s != self.fake_goal:
            s.append(0)
            s.append(0)
        return s

    def step(self, action):
        #print(action)
        #action = action.char
        #action = event
        s = self.canvas.coords(self.rect)
        # s.append(0)
        # s.append(0)
        #s = self.getState()

        base_action = np.array([0, 0])

        if action == 0:
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

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        return self.getReward(s, s_)

    def getReward(self, s, s_):
        # function approximation
        rectx, recty = int((s[0] - 5) / 40), int((s[1] - 5) / 40)
        rectx_, recty_ = int((s_[0] - 5) / 40), int((s_[1] - 5) / 40)
        real_goalx, real_goaly = int((self.canvas.coords(self.real_goal)[
                                     0] - 5) / 40), int((self.canvas.coords(self.real_goal)[1] - 5) / 40)

        # fake_goal_coor = []
        # index = 0
        # while index < len(self.fake_goal):
        #     fake_goalx, fake_goaly = int((self.canvas.coords(self.fake_goal[index])[
        #                                       0] - 5) / 40), int((self.canvas.coords(self.fake_goal[index])[1] - 5) / 40)
        #     index += 1
        #     fake_goal_coor.append([fake_goalx, fake_goaly])

        fake_goalx, fake_goaly = int((self.canvas.coords(self.fake_goal)[
                                      0] - 5) / 40), int((self.canvas.coords(self.fake_goal)[1] - 5) / 40)
        # manhattan distance
        # previous state s
        pre_dist_to_real = abs(rectx - real_goalx) + abs(recty - real_goaly)
        #pre_dist_to_fake = abs(rectx - fake_goalx) + abs(recty - fake_goaly)

        # next state s_
        dist_to_real = abs(rectx_ - real_goalx) + abs(recty_ - real_goaly)
        # dist_to_fake_list = []
        # for goal in fake_goal_coor:
        #     dist = abs(rectx_ - goal[0]) + abs(recty_ - goal[1])
        #     dist_to_fake_list.append(dist)
        #
        # dist_to_fake = min(dist_to_fake_list)
        # index = dist_to_fake_list.index(dist_to_fake)
        # fake_goal_nearest = fake_goal_coor[index]
        # #print(fake_goal_nearest)
        pre_dist_to_fake = abs(rectx - fake_goalx) + abs(recty - fake_goaly)
        dist_to_fake = abs(rectx_ - fake_goalx) + abs(recty_ - fake_goaly)
        #dist_between_goal = abs(fake_goalx - real_goalx) + abs(fake_goaly - real_goaly)

        done = False
        reward = 0

        # moving towards goal
        if pre_dist_to_real > dist_to_real:
            reward += 3
        else:
            reward -= 5

        # if pre_dist_to_fake > dist_to_fake:
        #     reward += 0.1
        # else:
        #     reward += 0

        #truthful = dist_to_real - (real_goalx + real_goaly) < dist_to_fake - (fake_goal_nearest[0] + fake_goal_nearest[1])
        truthful = dist_to_real - (real_goalx + real_goaly) < dist_to_fake - \
                   (fake_goalx + fake_goaly)
        if truthful:
            reward -= 1
        else:
            reward += 1

        if s_ == self.canvas.coords(self.real_goal):
            done = True

            """
            change between big reward or small reward to real goal
            """
            #big reward
            #reward += (MAZE_H + MAZE_W) / 2 * 0.1
            reward += 500
            #reward += 0.1
            # small reward
            #reward += 0.1
        # print(reward, done)
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

if __name__ == '__main__':
    #env = Maze(mode=1)
    #a = env.bind('<Key>', env.move_play_01)

    #env = Maze(mode=2)
    #a = env.bind('<Key>', env.move_play_02)

    # env.pre_work()

    env = Maze(mode=2)
    a = env.bind('<Key>', env.step)
    env.mainloop()

