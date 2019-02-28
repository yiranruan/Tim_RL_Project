import numpy as np
import pandas as pd
import time
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            rows,
            cols,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max #???

        self.learn_step_counter = 0
        # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory = [0] * self.memory_size

        self._build_net(rows, cols)
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.total_cost = 0
        if output_graph:
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('Tim_RL_Project/logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self, rows, cols):
        length = rows + 2
        width = cols + 2
        self.image = tf.placeholder(tf.float32, [None, length, width], name='Image')
        # self.steps = tf.Variable(0, name='step')
        # self.one = tf.constant(1)
        # self.step_raise = self.steps + [1]
        # ---------------------cnn_net--------------------
        with tf.variable_scope('cnn_net'):
            img_reshape = tf.reshape(self.image, [-1, length, width, 1])
            
            # -----------------------conv1-----------------------
            conv1 = tf.layers.conv2d(
                img_reshape,
                filters=32,
                kernel_size=[4,4],
                padding='SAME',
                activation=tf.nn.relu
            )
            # -----------------------pooling1-----------------------
            pooling1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=[2,2],
                strides=2
            )
            # -----------------------conv2-----------------------
            conv2 = tf.layers.conv2d(
                pooling1,
                filters=64,
                kernel_size=[4,4],
                padding='SAME',
                activation=tf.nn.relu
            )
            # -----------------------pooling2-----------------------
            pooling2 = tf.layers.max_pooling2d(
                conv2,
                pool_size=[2,2],
                strides=2
            )
            # -----------------------pooling_flat-----------------------
            # pooling_flat = tf.reshape(pooling2, [-1, int(length/4)*int(width/4)*64])
            pooling_flat = tf.layers.flatten(pooling2)
            # -----------------------dense1-----------------------
            dense = tf.layers.dense(
                inputs=pooling_flat,
                units=144,
                activation=tf.nn.relu
            )
            drop = tf.layers.dropout(
                inputs=dense,
                rate=0.5
            )
            # -----------------------dense2-----------------------
            self.maps_info = tf.layers.dense(
                inputs=drop,
                units=self.n_features
            )

        # self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 128, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.maps_info, w1) + b1)
                
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
                # tf.summary.histogram('target_net/q_eval', self.q_eval) #111111111111111

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        # ---------------------target_net--------------------
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.maps_info, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
                # tf.summary.histogram('target_net/q_next', self.q_next) #111111111111111

    def store_transition(self, image, a, r, image_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
            # self.memory_warehouse = tuple()
        # observation = self.sess.run(self.maps_info, feed_dict={self.image: image})
        # observation_ = self.sess.run(self.maps_info, feed_dict={self.image: image_})
        # observation = image
        # observation_ = image_
        # print('ob:',observation)
        # print('ob_:',observation_)
        transition = (image, a, r , image_)
        # print('image_',image_)
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition

        self.memory_counter += 1

    def choose_action(self, map_image, train):
        # print('op:',len(tf.get_default_graph().get_operations()),tf.get_default_graph().get_operations())
        # self.sess.run(tf.assign(self.steps, tf.add(self.steps, self.one)))
        # print(self.sess.run(self.steps))
        # observation = self.sess.run(self.maps_info, feed_dict={self.image: map_image})
        observation = map_image
        if train == True:
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval, feed_dict={self.image: observation[np.newaxis, :]})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
            return action
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.image: observation[np.newaxis, :]})
            action = np.argmax(actions_value)
            return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')
        

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = [self.memory[i] for i in sample_index]

        q_next = self.sess.run(
            self.q_next,
            feed_dict={
                self.image: np.array(list(map(lambda x:x[3], batch_memory))),
            })
        
        q_eval = self.sess.run(
            self.q_eval,
            feed_dict={
                self.image: np.array(list(map(lambda x:x[0], batch_memory))),
            })
        # 111111111111
        # rs = self.sess.run(self.merged,feed_dict={
        #         self.s_: batch_memory[:, -self.n_features:],
        #         self.s: batch_memory[:, :self.n_features],
        #     })
        # self.writer.add_summary(rs)
    

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype = np.int32)
        eval_act_index = np.array(list(map(lambda x:x[1], batch_memory))).astype(int)
        reward = np.array(list(map(lambda x:x[2], batch_memory)))
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # time.sleep(3)
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                    feed_dict={self.image: np.array(list(map(lambda x:x[0], batch_memory))),
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.total_cost += self.cost
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
        self.learn_step_counter += 1
    
    def save_network(self, train, name):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(max_to_keep=2000)
        if train:
            # saver = tf.train.Saver()
            # tf.reset_default_graph()
            self.saver.save(self.sess, '/home/yiranruan/cnn_s/data/checkpoint_dir_'+name+'/MyModel_'+name)
        else:
            # saver = tf.train.import_meta_graph('/Workfiles/python3/Tim_RL_Project/checkpoint_dir/MyModel.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint('/home/yiranruan/cnn_s/data/checkpoint_dir_'+name))
    
    # def get_steps(self):
    #     print(self.sess.run(self.steps))
    
    def plot_cost(self, name):
        self.f = open("./cost_2_"+name+".txt",'a') 
        self.f.write(str(self.cost_his)+'\n')
        self.f.close()




    

if __name__ == "__main__":
    from envR import envR
    env = envR(True)
    env.reset()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      rows=env.rows, cols=env.cols,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    for i in range(100):
        action = input('actions: ')
        reward = 0
        pre_maps = env.get_maps()
        env.step(action, True)
        RL.store_transition(pre_maps, 0, reward, env.get_maps())
