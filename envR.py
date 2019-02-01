from Maze import generateMaze
from Maze import copy_grid
import random,sys,time
import numpy as np

seed = None
class envR:
    def __init__(self, mode=1,
                x_real=None,
                y_real=None,
                ):
        self.action_space  = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2

        seed = None
        if len(sys.argv) > 1:
            seed = int(sys.argv[1])
        self.maze = generateMaze(seed)
        self.save_grid = copy_grid(self.maze.grid)
        self.total_cost = 0
        self.real = self.maze.get_real_goal()
        self.fake = self.maze.get_fake_goal()

    def reset(self):
        # print('reset',self.grid)
        self.maze.grid = copy_grid(self.save_grid)
        self.agent = self.maze.get_start()
        self.total_cost = 0
        self.maze.reset()
        return np.array(self.agent) #!!!

    def update_map(self, s, s_, train):
        self.maze.pass_by(s)
        self.maze.next_step(s_)
        if not train:
            print(str(self.maze))
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
    
    def step(self, action, train):
        row, col = self.agent
        done = False
        s_ = self.agent
        
        if action == 'u':
            if self.maze.isWall(row-1, col):s_ = (row-1, col)
        elif action == 'd':
            if self.maze.isWall(row+1, col):s_ = (row+1, col)
        elif action == 'l':
            if self.maze.isWall(row, col-1):s_ = (row, col-1)
        elif action == 'r':
            if self.maze.isWall(row, col+1):s_ = (row, col+1)

        if self.maze.isTerminal(s_): done = True
        reward = self.maze.get_reward(s_)
        # print("tut",action, self.agent, s_)
        self.total_cost += reward
        self.update_map(self.agent, s_, train)
        self.agent = s_
        self.maze.set_position(s_)
        return np.array(s_), reward, done

# if __name__ == '__main__':
#     env = envR()
#     for _ in range(1000):
#         action = random.choice(env.action_space)
#         print(action)
#         env.step(action)
#         time.sleep(0.1)
