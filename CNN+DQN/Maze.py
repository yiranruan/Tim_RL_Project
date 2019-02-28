import random, sys, math
from Food import food

W = '%'
F = '.'
C = 'o'
E = ' '

class Maze:

    def __init__(self, rows, cols, random=True):
        """
        generate an empty maze
        anchor is the top left corner of this grid's position in its parent grid
        """
        self.r = rows
        self.c = cols
        self.random = random
        # self.r_s, self.c_s = rows, cols
        self.grid = [[E for col in range(cols)] for row in range(rows)]
        self.rooms = []
        # ========================parameter========================
        self.food_list = []
        self.food_pos = []
        self.position = ()
        self.start = ()
    

    def reset(self):
        self.position = (self.r-2, 1)
        self.start = self.position
        for f in self.food_list:
            f.reset()
            row, col = f.get_pos()
            self.grid[row][col] = C


    def to_map(self):
        """
        add a flipped symmetric copy on the right
        add a border
        """

        ## add a flipped symmetric copy
        # for row in range(self.r):
        #     for col in range(self.c-1, -1, -1):
        #         self.grid[self.r-row-1].append(self.grid[row][col])
        # self.c *= 2

        ## add a border
        for row in range(self.r):
            self.grid[row] = [W] + self.grid[row] + [W]
        self.c += 2
        self.grid.insert(0, [W for c in range(self.c)])
        self.grid.append([W for c in range(self.c)])
        self.r += 2

    def __str__(self):
        s = ''
        for row in range(self.r):
            for col in range(self.c):
                s += self.grid[row][col]
            s += '\n'
        return s[:-1]

    
    # def add_pacman_stuff(self, max_food=2, toskip=0):
        # """
        # add pacmen starting position
        # add food at dead ends plus some extra
        # """

        # ## parameters
        # max_depth = 2
        # # ========================1.25========================
        # depth = 0
        # total_food = 0
        # self.grid[self.r-2][1] = 'P'
        # self.position = (self.r-2, 1)
        # self.start = self.position
        # # ========================1.25========================
        # # self.grid[self.r-3][1] = '1'
        # # self.grid[1][self.c-2] = C #----->fixed food
        # # self.grid[2][self.c-2] = '2'
        # # ========================1.25========================
        # # extra random food
        # while total_food < max_food:
        #     # row = random.randint(1, self.r-1)
        #     # col = random.randint(1+toskip, (self.c/2)-1)
        #     row = 1
        #     col = 2
        #     if (row > self.r-6) and (col < 6): continue
        #     if(abs(col - self.c/2) < 3): continue
        #     if self.grid[row][col] == E:
        #         self.grid[row][col] = C
        #         self.grid[self.r-row-1][self.c-(col)-1] = C #中心对称，如果使用 total_food += 2
        #         #food list ----> food object
        #         self.food_list.append(food((row,col)))
        #         self.food_pos.append((row,col))
        #         self.food_list.append(food((self.r-row-1,self.c-(col)-1)))
        #         self.food_pos.append((self.r-row-1,self.c-(col)-1))
        #         # print('foodlist1',row,col)
        #         # print('foodlist2',self.r-row-1,self.c-(col)-1)
        #         total_food += 2
        #         # print('total',total_food)
        # self.food_list[0].set_status(True)
        # ========================1.25========================
        #food list
    def add_pacman_stuff(self, max_food=2, toskip=0):
        """
        add pacmen starting position
        add food at dead ends plus some extra
        """
        # parameters
        max_depth = 2
        depth = 0
        total_food = 0
        self.grid[self.r-2][1] = 'P'
        self.position = (self.r-2, 1)
        self.start = self.position
        self.addFood(max_food)


    def addFood(self, max_food):
        total_food = 0
        self.food_list = []
        self.food_pos = []

        # fixed
        if not self.random:
            self.food_list.append(food((7, 7)))
            self.food_pos.append((7, 7))
            self.grid[7][7] = 'R'
            self.food_list.append(food((5, 7)))
            self.food_pos.append((5, 7))
            self.grid[5][7] = C
        else:
        # random
            while total_food < max_food:
                row = random.randint(1, self.r-1)
                col = random.randint(1, self.c-1)
                if self.grid[row][col] == E:
                    if len(self.food_list) == 0:
                        self.grid[row][col] = 'R'
                    else:
                        self.grid[row][col] = C
                    # food list ----> food object
                    self.food_list.append(food((row, col)))
                    self.food_pos.append((row, col))
                    # print('foodlist1',row,col)
                    # print('foodlist2',self.r-row-1,self.c-(col)-1)
                    total_food += 1
                    # print('total',total_food)
        self.food_list[0].set_status(True)

    def getFoodList(self):
        return self.food_list

    def getFoods(self):
        foodGrid = copy_grid(self.grid)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                    if self.isFood(i,j) and not self.isFoodVisited(i,j):
                            foodGrid[i][j] = '1'
                    else:
                        foodGrid[i][j] = '0'
        row, col = self.food_list[0].get_pos()
        if not self.isFoodVisited(row,col):
            foodGrid[row][col] = '2'
        return foodGrid
        
    def getWalls(self):
        wallGrid = copy_grid(self.grid)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.isWall(i,j):
                    wallGrid[i][j] = '1'
                else:
                    wallGrid[i][j] = '0'
        return wallGrid

    def getPos(self):
        row, col = self.position
        agentGrid = copy_grid(self.grid)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if i == row and j == col:
                    agentGrid[i][j] = '1'
                else:
                    agentGrid[i][j] = '0'
        return agentGrid
    
    def action_to_arrow(self, action):
        if action == 'u':
            return '↑'
        elif action == 'd':
            return '↓'
        elif action == 'l':
            return '←'
        elif action == 'r':
            return '→'
        elif action == 'ul':
            return '↖'
        elif action == 'dl':
            return '↙'
        elif action == 'ur':
            return '↗'
        elif action == 'dr':
            return '↘'
        
    def pass_by(self, pos, action):
        row, col = pos
        for f in self.food_list:
            # print("pos:",pos)
            # print("f.get_pos",f.get_pos())
            if f.get_pos() == pos: 
                self.grid[row][col] = 'R'
                return True
        arrow = self.action_to_arrow(action)
        self.grid[row][col] = arrow
        return False

    def next_step(self, pos):
        row, col = pos
        self.grid[row][col] = 'P'
        self.position = (row, col)

    def get_start(self):
        return self.start
    
    def set_position(self, pos):
        self.position = pos

    def get_real_goal(self):
        return self.food_list[0]
     
    def get_fake_goal(self):
        return self.food_list[1]
    
    def isWall(self, row, col):
        if self.grid[row][col] == '%': return True
        else: return False

    def isFood(self, row, col):
        if (row,col) in self.food_pos: return True
        else: False
    
    def isFoodVisited(self, row, col):
        for f in self.food_list:
            if f.get_pos() == (row, col):
                return f.is_visited
        return True

    def isTerminal(self, pos):
        # for i in range(len(self.food_list)):
        #     if (pos == self.food_list[i].get_pos()) and self.food_list[i].get_status(): return True
        #     elif (pos == self.food_list[i].get_pos()) and (not self.food_list[i].get_status()): 
        #         self.food_list[i].visited()
        #         self.food_list[0].set_reward(10)
        #         return False
        # return False
        if pos == self.food_list[0].get_pos():
            return True
        else: return False
    
    def get_reward(self, pre_pos, pos):
        rectx, recty = pre_pos
        rectx_, recty_ = pos
        real_goalx, real_goaly = self.food_list[0].get_pos()
        fake_goalx, fake_goaly = self.food_list[1].get_pos()
        # print(rectx, recty)
        # print(rectx_, recty_)
        # print(real_goalx, real_goaly)
        # print(fake_goalx, fake_goaly)
        # print(self.r, self.c)

        # manhattan distance
        # previous state s
        pre_dist_to_real = abs(rectx - real_goalx) + abs(recty - real_goaly)
        pre_dist_to_fake = abs(rectx - fake_goalx) + abs(recty - fake_goaly)

        # next state s_
        dist_to_real = abs(rectx_ - real_goalx) + abs(recty_ - real_goaly)
        dist_to_fake = abs(rectx_ - fake_goalx) + abs(recty_ - fake_goaly)
        dist_between_goal = abs(fake_goalx - real_goalx) + abs(fake_goaly - real_goaly)

        reward = 0

        # moving towards goal
        if pre_dist_to_real > dist_to_real:
            reward += 3
        else:
            reward -= 5    # 5

        # check whether the new state is truthful 這裡默認了出生點為(self.r - 2, 1)
        truthful = dist_to_real - (abs(real_goalx - (self.r - 2)) + abs(real_goaly - 1)) < dist_to_fake - \
            (abs(fake_goalx - (self.r - 2)) + abs(fake_goaly - 1))
        # print(truthful)
        if not truthful or (dist_to_real + dist_to_fake == dist_between_goal):
            reward += 1
        else:
            reward -= 1

        # if pos == pre_pos:
        #     reward -= 5
        if pos == self.food_list[0].get_pos():
            reward += 500
        return reward


    def get_distance(self, ppos):
        r, c = ppos
        min_d = 9999999
        for i in self.food_list:
            r_f, c_f = i.get_pos()
            d = math.sqrt((r - r_f) ** 2 + (c - c_f) ** 2)
            if d < min_d:
                min_d = d
                min_i = i
        return min_d * min_i.get_reward()
    

def copy_grid(grid):
    new_grid = []
    for row in range(len(grid)):
        new_grid.append([])
        for col in range(len(grid[row])):
            new_grid[row].append(grid[row][col])
    return new_grid



MAX_DIFFERENT_MAZES = 10000

def generateMaze(length, width, random):
    maze = Maze(length, width, random)
    # maze = Maze(14,15)
    maze.to_map()
    maze.add_pacman_stuff(2)
    return maze

if __name__ == '__main__':
    import numpy as np
    m = generateMaze(10, 10)
    print(str(m))
    # print(np.array(list(m.getWalls())).astype(int))