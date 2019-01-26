import random, sys

W = '%'
F = '.'
C = 'o'
E = ' '

class Maze:
    def __init__(self, rows, cols, anchor = (0, 0), root = None):
        self.r = rows
        self.c = cols
        self.anchor = anchor
        self.grid = [[E for col in range(cols)] for row in range(rows)]
        self.rooms = []
        self.root = root
        if not self.root: self.root = self

    def to_map(self):
        for row in range(self.r):
            for col in range(self.c-1, -1, -1):
                self.grid[self.r-row-1].append(self.grid[row][col])
        self.c *= 2
        
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
        add_r, add_c = self.anchor
        if vert:
            gaps = min(self.r, gaps)
            slots = [add_r+x for x in range(self.r)]
            if not 0 in slots:
                if self.root.grid[min(slots)-1][add_c+i] == E: slots.remove(min(slots))
                if len(slots) <= gaps: return 0
            if not self.root.c-1 in slots:
                if self.roo

def make_with_prison(room, depth, gaps=1, vert=True, min_width=1, gapfactor=0.5):
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
        cur_col


MAX_DIFFERENT_MAZES = 10000

def generateMaze(seed = None):
    if not seed:
        seed = random.randint(1, MAX_DIFFERENT_MAZES)
    random.seed(seed)
    maze = Maze(16, 16)
    maze.to_map()
    return str(maze)


if __name__ == '__main__':
    seed = None
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    print(generateMaze(seed))
