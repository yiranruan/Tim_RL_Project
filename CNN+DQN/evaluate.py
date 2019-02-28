class Evaluate:
    '''
        Need to modify move() method according to your environment!!!!!
    '''

    def __init__(self, rows, cols, start_pos=(0, 0), real_pos=(2, 0), fake_pos=(0, 2)):
        '''
            Initialize the positions that will never change
            For example, if your agent always start at (1,1), initialize start_pos to (1,1)

            inputs:
            start_pos, real_pos, fake_pos are tuples (row, col) -> (int, int)
            rows, cols are int
            1. rows: number of rows
            2. cols: number of cols
            3. start_pos: a coordinate that indicates the starting position of pacman
            4. real_pos: the coordinate of real goal
            5. fake_pos: the coordinate of fake goal
        '''
        self.start_pos = start_pos
        self.real_pos = real_pos
        self.fake_pos = fake_pos
        self.rows = rows
        self.cols = cols

    def evaluate_path(self, path):
        '''
            This function is used to evaluate a path's deceptiveness

            inputs:
            path is a list of string, for example ['u','u','d','l','r']
            1. path: agent's path

            make sure that the input path is correct(reaches the real goal)!

            return:
            1. cost: the cost of the path (each step cost 1)
            2. density: the deceptive density of the path until ldp
            3. find_target_node: whether the path visits the target node, this parameter can show the deceptive extent of your path
        '''
        pos = self.start_pos
        cost = 0
        deceptive = 0
        path_truthful = []  # show whether each step is Truthful or Deceptive
        path_pos = []
        # if (self.get_distance(self.start_pos, self.real_pos) == self.get_rmp()):    # check if start position is target node
        #     find_target_node = True
        # else:
        find_target_node = False

        for action in path:
            pos = self.move(pos, action)
            cost += 1
            # print(pos)
            if self.is_deceptive(pos):
                # if self.get_distance(pos, self.real_pos) == self.get_rmp():
                #     find_target_node = True
                deceptive += 1
                path_truthful.append(False)
            else:
                path_truthful.append(True)
            path_pos.append(pos)

        if False in path_truthful:
            ldp_step = self.get_ldp(path_truthful)
            density = deceptive / ldp_step

            # check whether ldp is target node
            if self.get_distance(path_pos[ldp_step - 1], self.real_pos) == self.get_rmp():
                find_target_node = True
            # print(deceptive, ldp_index)
        else:
            density = 1
            find_target_node = True  # start position is target node
        # print(density)
        # deceptive_percentage = deceptive / len(path)
        return cost, density, find_target_node

    def get_optimal(self):
        '''
            Get the optimal solution for current start_pos and goals_pos
            return optimal cost and number of deceptive node
        '''
        cost = self.get_distance(self.start_pos, self.real_pos)
        # target_pos = (min(abs(self.real_pos[0], self.fake_pos[0]),
        #               min(self.real_pos[1], self.fake_pos[1]))
        deceptive_cost = min(abs(self.start_pos[0] - self.real_pos[0]), abs(self.start_pos[0] - self.fake_pos[0])) + min(
            abs(self.start_pos[1] - self.real_pos[1]), abs(self.start_pos[1] - self.fake_pos[1]))
        # deceptive_percentage = deceptive_cost / cost
        return cost, deceptive_cost

    def get_rmp(self):
        '''
            calculate the length of rmp
        '''
        cost, deceptive_cost = self.get_optimal()
        return cost - deceptive_cost

    def get_ldp(self, path):
        '''
            return the step of ldp
        '''
        for i in reversed(range(len(path))):
            if not path[i]:
                return i + 1
        raise ValueError("False is not in list")

    '''
        Setters: used to set the positions of start and goals
    '''

    def set_start(self, start_pos):
        '''
            Set the coordinate of starting position of pacman
        '''
        self.start_pos = start_pos

    def set_goals(self, real_pos=None, fake_pos=None):
        '''
            Set the coordinates of goals
        '''
        if real_pos is not None:
            self.real_pos = real_pos
        if fake_pos is not None:
            self.fake_pos = fake_pos

    '''
        Following methods need to be modified according to environment!
    '''

    def move(self, pos, action):
        x, y = pos

        if action == 'u':  # up，并且检查是否超出边界
            if x != 1:  # need to be changed according to environment
                x -= 1
        elif action == 'd':  # down
            if x != self.rows:
                x += 1
        elif action == 'r':  # right
            if y != self.cols:
                y += 1
        elif action == 'l':  # left
            if y != 1:
                y -= 1
        return x, y

    '''
        Below are two helper methods, no need to read
    '''

    def is_deceptive(self, pos):
        x, y = pos
        real_goalx, real_goaly = self.real_pos
        fake_goalx, fake_goaly = self.fake_pos
        startx, starty = self.start_pos

        # manhattan distance
        dist_to_real = abs(x - real_goalx) + abs(y - real_goaly)
        dist_to_fake = abs(x - fake_goalx) + abs(y - fake_goaly)

        # check whether the new state is truthful
        truthful = dist_to_real - (abs(real_goalx - startx) + abs(real_goaly - starty)
                                   ) < dist_to_fake - (abs(fake_goalx - startx) + abs(fake_goaly - starty))
        if not truthful:
            return True
        else:
            return False

    def get_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)


if __name__ == '__main__':
    path = []
    for _ in range(11):
        a = input("input: ")
        path.append(a)
    evaluate = Evaluate(rows=10, cols=10, start_pos=(0, 0))
    evaluate.set_start((1, 1))
    evaluate.set_goals(real_pos=(8, 3), fake_pos=(3, 3))  # 8 3 , 3 3
    print(evaluate.evaluate_path(path))
    print(evaluate.get_optimal())
