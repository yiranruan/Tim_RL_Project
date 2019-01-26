class food:
    def __init__(self, pos, status=False):
        self.status = status
        self.pos = pos
        self.reward = 10
        self.is_visited = False

    def get_pos(self):
        return self.pos
    
    def set_status(self, status):
        self.status = status
        if self.status and (not self.is_visited): self.reward = -10

    def visited(self):
        self.is_visited = True
    
    def set_reward(self, r):
        if self.status:
            self.reward = r

    def get_reward(self):
        return self.reward

    def get_status(self):
        return self.status

    def reset(self):
        if self.status:
            self.reward = -10
            self.is_visited = False
        else:
            self.reward = 10
            self.is_visited = False