class food:
    def __init__(self, pos, status=False):
        self.status = status
        self.status_s = status
        self.pos = pos
        self.reward = 1000
        self.reward_s = self.reward
        self.is_visited = False

    def get_pos(self):
        return self.pos
    
    def set_status(self, status):
        self.status = status
        self.status_s = status
        if self.status and (not self.is_visited):
            self.reward = -10
            self.reward_s = -10

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
        self.status = self.status_s
        self.is_visited = False
        self.reward = self.reward_s