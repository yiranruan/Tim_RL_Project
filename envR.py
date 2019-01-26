from Maze import generateMaze
from Maze import copy_grid
import random,sys,time

seed = None
class envR:
    def __init__(self, mode=1,
                x_real=None,
                y_real=None,
                ):
        self.action_space  = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)

        seed = None
        if len(sys.argv) > 1:
            seed = int(sys.argv[1])
        self.maze = generateMaze(seed)
        self.backup_maze = generateMaze(seed)
        self.grid = copy_grid(self.maze.grid)
        self.agent = self.maze.get_start()
        self.total_cost = 0
        self.real = self.maze.get_real_goal()
        self.fake = self.maze.get_fake_goal()

    def reset(self):
        self.agent = self.maze.get_start()
        self.total_cost = 0
        self.maze = generateMaze(seed)
        self.maze.grid = self.grid
        return self.agent #!!!

    def update_map(self, s, s_):
        self.maze.pass_by(s)
        self.maze.next_step(s_)
        print(str(self.maze))
        # time.sleep(0.3)

    def step(self, action):
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
        self.total_cost += reward
        self.update_map(self.agent, s_)
        self.agent = s_
        self.maze.set_position(s_)
        return s_, reward, done

# if __name__ == '__main__':
#     env = envR()
#     for _ in range(1000):
#         action = random.choice(env.action_space)
#         print(action)
#         env.step(action)
#         time.sleep(0.1)
