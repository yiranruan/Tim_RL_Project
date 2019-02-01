import random, sys
from Food import food

W = '%'
F = '.'
C = 'o'
E = ' '

class Maze:

    def __init__(self, rows = 16, cols = 16, anchor=(0, 0), root=None):
        """
        generate an empty maze
        anchor is the top left corner of this grid's position in its parent grid
        """
        self.r = rows
        self.c = cols
        # self.r_s, self.c_s = rows, cols
        self.grid = [[E for col in range(cols)] for row in range(rows)]
        self.anchor = anchor
        self.rooms = []
        self.root = root
        # ========================parameter========================
        self.food_list = []
        self.position = ()
        self.start = ()
        # ========================parameter========================
        if not self.root: self.root = self
    

    def reset(self):
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
        for row in range(self.r):
            for col in range(self.c-1, -1, -1):
                self.grid[self.r-row-1].append(self.grid[row][col])
        self.c *= 2

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

    
    def add_pacman_stuff(self, max_food=2, toskip=0):
        """
        add pacmen starting position
        add food at dead ends plus some extra
        """

        ## parameters
        max_depth = 2
        # ========================1.25========================
        depth = 0
        total_food = 0
        self.grid[self.r-2][1] = 'P'
        self.position = (self.r-2, 1)
        self.start = self.position
        # ========================1.25========================
        # self.grid[self.r-3][1] = '1'
        # self.grid[1][self.c-2] = C #----->fixed food
        # self.grid[2][self.c-2] = '2'
        # ========================1.25========================
        # extra random food
        while total_food < max_food:
            row = random.randint(1, self.r-1)
            col = random.randint(1+toskip, (self.c/2)-1)
            if (row > self.r-6) and (col < 6): continue
            if(abs(col - self.c/2) < 3): continue
            if self.grid[row][col] == E:
                self.grid[row][col] = C
                self.grid[self.r-row-1][self.c-(col)-1] = C #中心对称，如果使用 total_food += 2
                #food list ----> food object
                self.food_list.append(food((row,col)))
                self.food_list.append(food((self.r-row-1,self.c-(col)-1)))
                # print('foodlist1',row,col)
                # print('foodlist2',self.r-row-1,self.c-(col)-1)
                total_food += 2
                # print('total',total_food)
        self.food_list[0].set_status(True)
        # ========================1.25========================
        #food list

    def getFoodList(self):
        return self.food_list
    
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
        if self.grid[row][col] == '%': return False
        else: return True
    
    def isTerminal(self, pos):
        for i in range(len(self.food_list)):
            if (pos == self.food_list[i].get_pos()) and self.food_list[i].get_status(): return True
            elif (pos == self.food_list[i].get_pos()) and (not self.food_list[i].get_status()): 
                self.food_list[i].visited()
                self.food_list[0].set_reward(10)
                return False
        return False
    
    def get_reward(self, pos):
        for i in self.food_list:
            if pos == i.get_pos(): return i.get_reward()
        return -1
    

def copy_grid(grid):
    new_grid = []
    for row in range(len(grid)):
        new_grid.append([])
        for col in range(len(grid[row])):
            new_grid[row].append(grid[row][col])
    return new_grid



MAX_DIFFERENT_MAZES = 10000

def generateMaze(seed = None):
    if not seed:
        seed = random.randint(1,MAX_DIFFERENT_MAZES)
    random.seed(seed)
    maze = Maze(19,19)
    maze.to_map()
    maze.add_pacman_stuff(2)
    return maze

if __name__ == '__main__':
    seed = None
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    print(generateMaze(seed))