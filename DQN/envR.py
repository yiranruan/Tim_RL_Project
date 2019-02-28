from Maze import generateMaze
from Maze import copy_grid
import tensorflow as tf
import random,sys,time
import numpy as np
# from cnn import cnn


class envR:
    def __init__(self, 
                show=True,
                rows=10,
                cols=10,
                n_features=10,
                ):
        self.show = show
        self.action_space  = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.rows = rows
        self.cols = cols
        self.n_features = (self.rows+2)*(self.cols+2)
        # self.save_grid = copy_grid(self.maze.grid)
        self.total_cost = 0

    def reset(self, random=True):
        # print('reset',self.grid)
        # m = self.generateMaze(rows,cols,None)
        # self.save_grid = copy_grid(m.grid)

        self.maze = generateMaze(self.rows, self.cols, random)
        self.agent = self.maze.get_start()
        self.total_cost = 0
        # self.maze.reset()
        self.visited = []
        return self.get_maps() #!!!

    def update_map(self, s, a, s_):
        self.maze.pass_by(s, a)
        self.maze.next_step(s_)
        if self.show:
            print(str(self.maze))
    
    def step(self, action):
        isVisited = True
        row, col = self.agent
        done = False
        s_ = self.agent
        reward = 0
        if action == 'u':
            if not self.maze.isWall(row-1, col):
                s_ = (row-1, col)
        elif action == 'd':
            if not self.maze.isWall(row+1, col):
                s_ = (row+1, col)
        elif action == 'l':
            if not self.maze.isWall(row, col-1):
                s_ = (row, col-1)
        elif action == 'r':
            if not self.maze.isWall(row, col+1):
                s_ = (row, col+1)
        # m = (s_, action)
        # if m in self.visited:
        #     reward -= 2
        # else:
        #     self.visited.append(m)
        #     isVisited = False
        # if isVisited:
        #     return True, [], reward, done
        reward = self.maze.get_reward(self.agent,s_)
        done = self.maze.isTerminal(s_)
        # reward = self.maze.get_reward(s_)
        # print("tut",action, self.agent, s_)
        self.total_cost += reward
        self.update_map(self.agent, action, s_)
        self.agent = s_
        self.maze.set_position(s_)

        return reward, done, action

    def get_maps(self):
        game_map = np.array(list(self.maze.getWalls())).astype(int)
        food = np.array(list(self.maze.getFoods())).astype(int) * 4
        game_map += food
        game_map += np.array(list(self.maze.getPos())).astype(int) * 2
        return game_map.reshape(len(game_map)*len(game_map[0]), order='C')

    def action_translate(self, action):
        if action == 0: return 'u'
        elif action == 1: return 'd'
        elif action == 2: return 'l'
        elif action == 3: return 'r'
    # def _cnn(self):
    #     # return cnn(image, self.n_features)
    #     leg = self.rows + 2
    #     wid = self.cols + 2
    #     self.x = tf.placeholder(tf.float32, [leg, 12])
    #     img = tf.reshape(self.x, [-1, leg, 12, 1])
    #     y = tf.placeholder(tf.float32, [None, self.n_features])

    #     conv1 = tf.layers.conv2d(
    #         img,
    #         filters=32,
    #         kernel_size=[4, 4],
    #         padding='SAME',
    #         activation=tf.nn.relu)

    #     pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)

    #     conv2 = tf.layers.conv2d(
    #         pool1,
    #         filters=64,
    #         kernel_size=[4, 4],
    #         padding='SAME',
    #         activation=tf.nn.relu)

    #     pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)

    #     pool2_flat = tf.reshape(pool2, [-1, int(leg/4)*int(wid/4)*64])
    #     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    #     drop = tf.layers.dropout(inputs=dense, rate=0.4)

    #     self.logits = tf.layers.dense(inputs=drop, units=self.n_features)

    #     # loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    #     # train = tf.train.AdamOptimizer(0.001),minimize(loss)
        
    #     self.sess = tf.Session()
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     self.sess.run(init_op)
    
    # def cnn(self,image):
    #     if not hasattr(self, 'cnn_'):
    #         self.cnn_ = self._cnn()
    #         if self.train: self.saver = tf.train.Saver()
    #         else:                
    #             self.saver = tf.train.import_meta_graph('/Workfiles/python3/Tim_RL_Project/checkpoint_dir/cnn')
                
    #     cnn_train = self.sess.run(self.logits, {self.x:image})

    #     # if self.train:
    #     #     self.saver.save(self.sess, '/Workfiles/python3/Tim_RL_Project/checkpoint_dir/cnn')
    #     # else:self.saver.restore(restore_sess, tf.train.lastest_checkpoint('/Workfiles/python3/Tim_RL_Project/checkpoint_dir/cnn'))

    #     return cnn_train

        #origin saving
        # x = tf.placeholder(tf.float32, [16, 32])
        # img = tf.reshape(x, [-1, 16, 32, 1])
        # y = tf.placeholder(tf.float32, [None, self.n_features])

        # conv1 = tf.layers.conv2d(
        #     img,
        #     filters=32,
        #     kernel_size=[4, 4],
        #     padding='SAME',
        #     activation=tf.nn.relu)

        # pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)

        # conv2 = tf.layers.conv2d(
        #     pool1,
        #     filters=64,
        #     kernel_size=[4, 4],
        #     padding='SAME',
        #     activation=tf.nn.relu)

        # pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)

        # pool2_flat = tf.reshape(pool2, [-1, 4*8*64])
        # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # drop = tf.layers.dropout(inputs=dense, rate=0.4)

        # logits = tf.layers.dense(inputs=drop, units=self.n_features)

        # sess = tf.Session()
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init_op)
        # train = sess.run(logits, {x:image})
        # return train

if __name__ == '__main__':
    env = envR()
    env.reset(random=False)

    d = False
    while True:
        a = input()
        if a == 'q':
            env.reset(random=False)
        else:
            i,c,r = env.step(a)
        # print(env.get_maps())

