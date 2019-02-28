from Maze import generateMaze
from Maze import copy_grid
import tensorflow as tf
import random,sys,time
import numpy as np
# from cnn import cnn


class envR:
    def __init__(self, 
                show=True,
                rows=10,
                cols=10,
                n_features=10,
                ):
        self.show = show
        self.action_space  = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.rows = rows
        self.cols = cols
        self.n_features = n_features
        # self.save_grid = copy_grid(self.maze.grid)
        self.total_cost = 0

    def reset(self, random=True):
        # print('reset',self.grid)
        # m = self.generateMaze(rows,cols,None)
        # self.save_grid = copy_grid(m.grid)

        self.maze = generateMaze(self.rows, self.cols, random)
        self.agent = self.maze.get_start()
        self.total_cost = 0
        # self.maze.reset()
        self.visited = []
        return self.get_maps() #!!!

    def update_map(self, s, a, s_):
        self.maze.pass_by(s, a)
        self.maze.next_step(s_)
        if self.show:
            print(str(self.maze))
    
    def step(self, action):
        isVisited = True
        row, col = self.agent
        done = False
        s_ = self.agent
        reward = 0
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
        # m = (s_, action)
        # if m in self.visited:
        #     reward -= 2
        # else:
        #     self.visited.append(m)
        #     isVisited = False
        # if isVisited:
        #     return True, [], reward, done
        reward = self.maze.get_reward(self.agent,s_)
        done = self.maze.isTerminal(s_)
        # reward = self.maze.get_reward(s_)
        # print("tut",action, self.agent, s_)
        self.total_cost += reward
        self.update_map(self.agent, action, s_)
        self.agent = s_
        self.maze.set_position(s_)

        return reward, done, action

    def get_maps(self):
        game_map = np.array(list(self.maze.getWalls())).astype(int)
        food = np.array(list(self.maze.getFoods())).astype(int) * 4
        game_map += food
        game_map += np.array(list(self.maze.getPos())).astype(int) * 2
        return game_map

    def action_translate(self, action):
        if action == 0: return 'u'
        elif action == 1: return 'd'
        elif action == 2: return 'l'
        elif action == 3: return 'r'
