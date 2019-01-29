import random, sys
from Food import food

W = '%'
F = '.'
C = 'o'
E = ' '

class Maze:

    def __init__(self, rows, cols, anchor=(0, 0), root=None):
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
        # self.r, self.c = self.r_s, self.c_s

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

    def add_wall(self, i, gaps=1, vert=True):
        """
        add a wall with gaps
        """
        pass
        # add_r, add_c = self.anchor
        # if vert:
        #     gaps = min(self.r, gaps)
        #     slots = [add_r+x for x in range(self.r)]
        #     if not 0 in slots:
        #         if self.root.grid[min(slots)-1][add_c+i] == E: slots.remove(min(slots))
        #         if len(slots) <= gaps: return 0
        #     if not self.root.c-1 in slots:
        #         if self.root.grid[max(slots)+1][add_c+i] == E: slots.remove(max(slots))
        #     if len(slots) <= gaps: return 0
        #     random.shuffle(slots)
        #     for row in slots[int(round(gaps)):]:
        #         self.root.grid[row][add_c+i] = W
        #     self.rooms.append(Maze(self.r, i, (add_r,add_c), self.root))
        #     self.rooms.append(Maze(self.r, self.c-i-1, (add_r,add_c+i+1), self.root))
        # else:
        #     gaps = min(self.c, gaps)
        #     slots = [add_c+x for x in range(self.c)]
        #     if not 0 in slots:
        #         if self.root.grid[add_r+i][min(slots)-1] == E: slots.remove(min(slots))
        #         if len(slots) <= gaps: return 0
        #     if not self.root.r-1 in slots:
        #         if self.root.grid[add_r+i][max(slots)+1] == E: slots.remove(max(slots))
        #     if len(slots) <= gaps: return 0
        #     random.shuffle(slots)
        #     for col in slots[int(round(gaps)):]:
        #         self.root.grid[add_r+i][col] = W
        #     self.rooms.append(Maze(i, self.c, (add_r,add_c), self.root))
        #     self.rooms.append(Maze(self.r-i-1, self.c, (add_r+i+1,add_c), self.root))

        # return 1
    
    def add_pacman_stuff(self, max_food=2, max_capsules=4, toskip=0):
        """
        add pacmen starting position
        add food at dead ends plus some extra
        """

        ## parameters
        max_depth = 2

        # ========================1.25========================
        # ## add food at dead ends
        depth = 0
        total_food = 0
        # while True:
        #     new_grid = copy_grid(self.grid)
        #     depth += 1
        #     num_added = 0
        #     for row in range(1, self.r-1):
        #         for col in range(1+toskip, int((self.c/2)-1)):
        #             if (row > self.r-6) and (col < 6): continue
        #             if self.grid[row][col] != E: continue
        #             neighbors = (self.grid[row-1][col]==E) + (self.grid[row][col-1]==E) + (self.grid[row+1][col]==E) + (self.grid[row][col+1]==E)
        #             if neighbors == 1:
        #                 new_grid[row][col] = F
        #                 new_grid[self.r-row-1][self.c-(col)-1] = F
        #                 num_added += 2
        #                 total_food += 2
        #     self.grid = new_grid
        #     if num_added == 0: break
        #     if depth >= max_depth: break
        # ========================1.25========================
        ## starting pacmen positions
        self.grid[self.r-2][1] = 'P'
        self.position = (self.r-2, 1)
        self.start = self.position
        # self.grid[self.r-2][1] = 'P'
        # self.position = (self.r-2, 1)
        # ========================1.25========================
        # self.grid[self.r-3][1] = '1'
        # self.grid[1][self.c-2] = C #----->fixed food
        # self.grid[2][self.c-2] = '2'
        # ========================1.25========================
        ## add capsules
        # total_capsules = 0
        # while total_capsules < max_capsules:
        #     row = random.randint(1, self.r-1)
        #     col = random.randint(1+toskip, (self.c/2)-2)
        #     if (row > self.r-6) and (col < 6): continue
        #     if(abs(col - self.c/2) < 3): continue
        #     if self.grid[row][col] == E:
        #         self.grid[row][col] = C
        #         self.grid[self.r-row-1][self.c-(col)-1] = C
        #         total_capsules += 2
        # ========================1.25========================
        # extra random food
        while total_food < max_food:
            row = random.randint(1, self.r-1)
            col = random.randint(1+toskip, (self.c/2)-1)
            if (row > self.r-6) and (col < 6): continue
            if(abs(col - self.c/2) < 3): continue
            if self.grid[row][col] == E:
                self.grid[row][col] = 'H'
                self.grid[self.r-row-1][self.c-(col)-1] = C #中心对称，如果使用 total_food += 2
                #food list ----> food object
                self.food_list.append(food((row,col)))
                self.food_list.append(food((self.r-row-1,self.c-(col)-1)))
                print('foodlist1',row,col)
                print('foodlist2',self.r-row-1,self.c-(col)-1)
                total_food += 2
                print('total',total_food)
        self.food_list[0].set_status(True)
        # ========================1.25========================
        #food list

    def getFoodList(self):
        return self.food_list

    def pass_by(self, pos):
        row, col = pos
        for f in self.food_list:
            print("pos:",pos)
            print("f.get_pos",f.get_pos())
            if f.get_pos() == pos: 
                self.grid[row][col] = 'A'
                return True
        self.grid[row][col] = E
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
            if pos == self.food_list[i].get_pos() and self.food_list[i].get_status(): return True
            elif pos == self.food_list[i].get_pos() and (not self.food_list[i].get_status()): 
                self.food_list[i].visited()
                self.food_list[0].set_reward(10)
                return False
        return False
    
    def get_reward(self, pos):
        for i in self.food_list:
            if pos == i.get_pos(): return i.get_reward()
        return -1
    

def make_with_prison(room, depth, gaps=1, vert=True, min_width=1, gapfactor=0.5):
    """
    Build a maze with 0,1,2 layers of prison (randomly)
    """
    p = random.randint(0,2)
    proll = random.random()
    if proll < 0.5:
        p = 1
    elif proll < 0.7:
        p = 0
    elif proll < 0.9:
        p = 2
    else:
        p = 3

    add_r, add_c = room.anchor
    print(p)
    for j in range(p):
        cur_col = 2*(j+1)-1
        for row in range(room.r):
            room.root.grid[row][cur_col] = W
        if j % 2 == 0:
            room.root.grid[0][cur_col] = E

        else:
            room.root.grid[room.r-1][cur_col] = E


    room.rooms.append(Maze(room.r, room.c-(2*p), (add_r, add_c+(2*p)), room.root))
    for sub_room in room.rooms:

        make(sub_room, depth+1, gaps, vert, min_width, gapfactor)
    return 2*p

def make(room, depth, gaps=1, vert=True, min_width=1, gapfactor=0.5):
    """
    recursively build a maze
    TODO: randomize number of gaps?
    """

    ## extreme base case
    if room.r <= min_width and room.c <= min_width: return

    ## decide between vertical and horizontal wall
    if vert: num = room.c
    else: num = room.r
    if num < min_width + 2:
        vert = not vert
        if vert: num = room.c
        else: num = room.r

    ## add a wall to the current room
    if depth==0: 
        wall_slots = [num-2]    ## fix the first wall
    else: 
        wall_slots = range(1, num-1)

    if len(wall_slots) == 0: return
    choice = random.choice(wall_slots)
    if not room.add_wall(choice, gaps, vert): return

    ## recursively add walls
    # if random.random() < 0.8:
    #         vert = not vert


    for sub_room in room.rooms:
        make(sub_room, depth+1, max(1,gaps*gapfactor), not vert,
                 min_width, gapfactor)

                 
    # for sub_room in room.rooms:
    #         make(sub_room, depth+1, max(1,gaps/2), not vert, min_width)

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
    maze = Maze(16,16)
    gapfactor = min(0.65,random.gauss(0.5,0.1))
    # skip = make_with_prison(maze, depth=0, gaps=3, vert=True, min_width=1, gapfactor=gapfactor)
    maze.to_map()
    maze.add_pacman_stuff(2, 4)
    return maze

if __name__ == '__main__':
    seed = None
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    print(generateMaze(seed))
