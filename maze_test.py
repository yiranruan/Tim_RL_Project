from Maze import generateMaze
import Maze,sys,random
from envR import envR


env = envR()
pos = (1,1)
row, col = pos
print('maze_test')
# env.maze.grid[row][col] = 'T'
while True:
    t = input()
    print("step:",t)
    a,b,c = env.step(t)
    for f in env.maze.food_list:
        if f.get_pos() == pos:
            maze.grid[row][col] = 'A'


print(str(maze))