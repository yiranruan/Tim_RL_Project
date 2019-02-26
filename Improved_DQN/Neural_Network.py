import tensorflow as tf
import numpy as np
from test import *

tf.set_random_seed(1)
np.random.seed(6)


class DeepQNetwork:
    def __init__(self,
                 state_size,
                 action_size,
                 learning_rate=0.00001,
                 gamma=0.9,
                 epsilon=0.9,
                 epsilon_increment=0,
                 memory_size=100000,
                 batch_size=32,
                 sess=None,
                 mode=1,
                 observer_mode=False):

        # neural network parameter
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.mode = mode

        # epsilon
        self.epsilon = 0 if epsilon_increment != 0 else epsilon  # exploration和exploration的阀门
        self.epsilon_max = epsilon
        self.epsilon_increment = epsilon_increment

        # memory pool
        self.memory_size = memory_size
        # memory_structure：[ [state, action, reward, state_], [...], ...]
        self.memory = np.zeros((self.memory_size, self.state_size * 2 + 2))
        # sample 32 transitions at a time
        self.batch_size = batch_size

        # observer or not
        self.observer_mode = observer_mode

        # build network
        if self.mode == 1:
            self._build_net_natural()
        elif self.mode == 2:
            self._build_net_fixed_q()

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Eval_net')
            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        elif self.mode == 3:
            self._build_net_double_net()

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Eval_net')
            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        elif self.mode == 4:
            self._build_net_prioritized()

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Eval_net')
            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            # --------------------------------memory part-------------------------------
            self.memory = Memory(capacity=memory_size)
        elif self.mode == 5:
            self._build_net_dueling()

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Eval_net')
            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # Others
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        self.loss_collection = []
        self.replace_target_iter = 200  # 4x4: 200; 10x10: 300; 7x7: 300
        self.learn_step_counter = 0  # total learning step
        self.merged = None
        self.writer = None

    def _build_net_natural(self):
        with tf.variable_scope('Inputs'):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.target_q = tf.placeholder(tf.float32, [None, self.action_size], name="target_q")
            self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
            self.action = tf.placeholder(tf.int32, [None, ], name='action')

        # initialize weight and bias
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ----------------------Layer-------------------------------------
        with tf.variable_scope('Eval_net'):
            with tf.variable_scope('layer01'):
                l1 = tf.layers.dense(self.state, self.state_size * 2 + 1, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='layer_8_20')

            with tf.variable_scope('layer02'):
                self.q_eval = tf.layers.dense(l1, self.action_size, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='layer_20_4')

        # ----------------------处理 eval_q-------------------------------------
        with tf.variable_scope('eval_q'):
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
            # index，to find the q value corresponding to actions
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        # ----------------------target_q---------------------------------------
        with tf.variable_scope('target_q'):
            target_q = self.reward + self.gamma * tf.reduce_max(self.target_q, axis=1, name='Qmax_s_')

        # ----------------------Loss and train--------------------------------
        with tf.variable_scope('Loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(target_q, self.q_eval_wrt_a))
            if self.observer_mode:
                tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('Train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def _build_net_fixed_q(self):
        # -----------------------input-------------------------------------------
        with tf.variable_scope('Inputs'):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.state_ = tf.placeholder(tf.float32, [None, self.state_size], name='state_')
            self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
            self.action = tf.placeholder(tf.int32, [None, ], name='action')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ----------------------eval_net---------------------------------------
        with tf.variable_scope('Eval_net'):
            with tf.variable_scope('layer01'):
                e1 = tf.layers.dense(self.state, self.state_size * 2 + 1, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='layer_8_20')

            with tf.variable_scope('layer02'):
                self.q_eval = tf.layers.dense(e1, self.action_size, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='layer_20_4')

        with tf.variable_scope('Eval_q'):
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        # ----------------------target_net--------------------------------------
        with tf.variable_scope('Target_net'):
            with tf.variable_scope('layer01'):
                t1 = tf.layers.dense(self.state_, self.state_size * 2 + 1, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='layer_8_20')

            with tf.variable_scope('layer02'):
                self.q_next = tf.layers.dense(t1, self.action_size, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='layer_20_4')

        with tf.variable_scope('Target_q'):
            q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            # todo 这是锁死target_net参数用的 ^_^
            self.target_q = tf.stop_gradient(q_target)

        # ----------------------Loss and train--------------------------------
        with tf.variable_scope('Loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q_eval_wrt_a))

        with tf.variable_scope('Train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def _build_net_double_net(self):
        # -----------------------input-------------------------------------------
        with tf.variable_scope('Inputs'):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.state_ = tf.placeholder(tf.float32, [None, self.state_size], name='state_')
            self.q_target = tf.placeholder(tf.float32, [None, self.action_size], name='q_target')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ----------------------eval_net---------------------------------------
        with tf.variable_scope('Eval_net'):
            with tf.variable_scope('layer01'):
                e1 = tf.layers.dense(self.state, self.state_size * 2 + 1, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='layer_8_20')
                if self.observer_mode:
                    tf.summary.histogram('e1', e1)

            with tf.variable_scope('layer02'):
                self.q_eval = tf.layers.dense(e1, self.action_size, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='layer_20_4')
                if self.observer_mode:
                    tf.summary.histogram('q_eval', self.q_eval)

        # ----------------------target_net--------------------------------------
        with tf.variable_scope('Target_net'):
            with tf.variable_scope('layer01'):
                t1 = tf.layers.dense(self.state_, self.state_size * 2 + 1, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='layer_8_20', trainable=False)

            with tf.variable_scope('layer02'):
                self.q_next = tf.layers.dense(t1, self.action_size, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='layer_20_4', trainable=False)

        # ----------------------Loss and train--------------------------------
        with tf.variable_scope('Loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval), name='loss')
            if self.observer_mode:
                tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('Train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def _build_net_prioritized(self):
        # -----------------------input--------------------------------------------
        with tf.variable_scope('Inputs'):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.state_ = tf.placeholder(tf.float32, [None, self.state_size], name='state_')
            self.q_target = tf.placeholder(tf.float32, [None, self.action_size], name='q_target')
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ----------------------eval_net---------------------------------------
        with tf.variable_scope('Eval_net'):
            with tf.variable_scope('layer01'):
                e1 = tf.layers.dense(self.state, self.state_size * 2 + 1, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='layer_8_20')

            with tf.variable_scope('layer02'):
                self.q_eval = tf.layers.dense(e1, self.action_size, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='layer_20_4')

        # ----------------------target_net--------------------------------------
        with tf.variable_scope('Target_net'):
            with tf.variable_scope('layer01'):
                t1 = tf.layers.dense(self.state_, self.state_size * 2 + 1, tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='layer_8_20', trainable=False)

            with tf.variable_scope('layer02'):
                self.q_next = tf.layers.dense(t1, self.action_size, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='layer_20_4', trainable=False)

        # ----------------------Loss and train--------------------------------
        with tf.variable_scope('Loss'):
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1, name='abs_errors')
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval), name='loss')

        with tf.variable_scope('Train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def _build_net_dueling(self):
        def build_layers(state, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('layer01'):
                l1 = tf.layers.dense(state, n_l1, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, trainable=trainable, name='layer_8_20')

            with tf.variable_scope('layer02_value'):
                self.V = tf.layers.dense(l1, 1, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, trainable=trainable, name='layer_20_1')

            with tf.variable_scope('layer02_advantage'):
                self.A = tf.layers.dense(l1, self.action_size, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, trainable=trainable, name='layer_20_4')

            with tf.variable_scope('Q'):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))

            return out

        # -----------------------input-------------------------------------------
        with tf.variable_scope('Inputs'):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.state_ = tf.placeholder(tf.float32, [None, self.state_size], name='state_')
            self.q_target = tf.placeholder(tf.float32, [None, self.action_size], name='q_target')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ----------------------eval_net---------------------------------------
        with tf.variable_scope('Eval_net'):
            self.q_eval = build_layers(state=self.state, n_l1=self.state_size * 2 + 1,
                                       w_initializer=w_initializer, b_initializer=b_initializer, trainable=True)

        # ----------------------target_net--------------------------------------
        with tf.variable_scope('Target_net'):
            self.q_next = build_layers(state=self.state_, n_l1=self.state_size * 2 + 1,
                                       w_initializer=w_initializer, b_initializer=b_initializer, trainable=False)

        # ----------------------Loss and train--------------------------------
        with tf.variable_scope('Loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('Train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    # ------------learning part--------------------------------------------

    def store_transition(self, s, a, r, s_):
        if self.mode == 4:
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)
        else:
            # 检查一下是否大脑中是否有关于训练的记忆（即检查该类中是否有memory_counter这个属性)
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            # 记忆形式; 叠过后全变成单一数字存在numpy array中，没有list
            transition = np.hstack((s, [a, r], s_))

            # replace the old memory with new memory
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, state):
        # to have batch dimension when feed into tf placeholder
        # 一定要凑成[None, xxx]; numpy 和 普通list索引方式不同
        state = state[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the state and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: state})
            action = np.argmax(actions_value)
        else:
            # from zero to four pick a number randomly
            action = np.random.randint(0, self.action_size)

        return action

    def learn(self):
        if self.mode != 4:
            # 记忆满了的情况
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            # 没满的情况
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

            batch_memory = self.memory[sample_index, :]

        if self.mode == 1:
            # ------------------Target_Q------------------------------------------
            target_q = self.sess.run(self.q_eval, feed_dict={
                self.state: batch_memory[:, -self.state_size:]})

            # ------------------train------------------------------------------------
            _, loss = self.sess.run([self.train_op, self.merged], feed_dict={
                self.state: batch_memory[:, :self.state_size],
                self.reward: batch_memory[:, self.state_size + 1],
                self.action: batch_memory[:, self.state_size],
                self.target_q: target_q})

            # ------------------observer--------------------------------------
            if self.observer_mode:
                self.writer.add_summary(loss, self.learn_step_counter)

            self.learn_step_counter += 1

        elif self.mode == 2:
            # check to replace target parameters
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.target_replace_op)

            self.sess.run([self.train_op], feed_dict={
                self.state: batch_memory[:, :self.state_size],
                self.reward: batch_memory[:, self.state_size + 1],
                self.action: batch_memory[:, self.state_size],
                self.state_: batch_memory[:, -self.state_size:],
            })

            self.learn_step_counter += 1

        elif self.mode == 3:
            # # check to replace target parameters
            # if self.learn_step_counter % self.replace_target_iter == 0:
            #     self.sess.run(self.target_replace_op)

            v1, v2 = self.sess.run([self.q_next, self.q_eval], feed_dict={  # v1: Q'(s'); v2: Q(s')
                self.state_: batch_memory[:, -self.state_size:],
                self.state: batch_memory[:, -self.state_size:]})

            v3 = self.sess.run(self.q_eval, {self.state: batch_memory[:, :self.state_size]})  # v3: Q(s)
            q_target = v3.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.state_size].astype(int)
            reward = batch_memory[:, self.state_size + 1]
            max_act_next = np.argmax(v2, axis=1)

            selected_q_next = v1[batch_index, max_act_next]

            # ----------------------------------test part---------------------------------------
            terminal_set = []
            for i in range(32):
                if batch_memory[i, self.state_size + 2] + batch_memory[i, self.state_size + 3] == 2:
                    terminal_set.append(i)
            # ----------------------------------------------------------------------------------

            # 并不是4个全换，其中的一个换了
            # q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
            for i in range(32):
                if i in terminal_set:
                    q_target[i, eval_act_index[i]] = reward[i]
                else:
                    q_target[i, eval_act_index[i]] = reward[i] + self.gamma * selected_q_next[i]

            _, loss = self.sess.run([self.train_op, self.loss],
                                    feed_dict={self.state: batch_memory[:, :self.state_size],
                                               self.q_target: q_target})

            if loss <= 0.1:
                self.sess.run(self.target_replace_op)

            # for changeable epsilon
            # if self.epsilon_increment != 0 and self.learn_step_counter >= 40000 and self.epsilon >= 0.9:
            #     self.reset_epsilon()

            if self.observer_mode:
                rs = self.sess.run(self.merged, feed_dict={self.state: batch_memory[:, :self.state_size],
                                                           self.q_target: q_target})
                self.writer.add_summary(rs, self.learn_step_counter)
            
            if self.learn_step_counter > 40000 and self.learn_step_counter % 3000 == 0:
                print('learn step counter:', self.learn_step_counter)
                deceptive_map_test(10, 10, RL=self)
                self.saver()
            
            self.learn_step_counter += 1

        elif self.mode == 4:
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.target_replace_op)

            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

            q_next, q_eval = self.sess.run(  # q_next: Q'(s')
                [self.q_next, self.q_eval],
                feed_dict={self.state_: batch_memory[:, -self.state_size:],
                           self.state: batch_memory[:, :self.state_size]})

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.state_size].astype(int)
            reward = batch_memory[:, self.state_size + 1]
            max_act_next = np.argmax(q_eval, axis=1)

            selected_q_next = q_next[batch_index, max_act_next]

            # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            terminal_set = []
            for i in range(32):
                if batch_memory[i, self.state_size + 2] + batch_memory[i, self.state_size + 3] == 2:
                    terminal_set.append(i)

            for i in range(32):
                if i in terminal_set:
                    q_target[i, eval_act_index[i]] = reward[i]
                else:
                    q_target[i, eval_act_index[i]] = reward[i] + self.gamma * selected_q_next[i]

            _, abs_errors, loss = self.sess.run([self.train_op, self.abs_errors, self.loss],
                                                feed_dict={self.state: batch_memory[:, :self.state_size],
                                                           self.q_target: q_target,
                                                           self.ISWeights: ISWeights})

            self.memory.batch_update(tree_idx, abs_errors)
            self.learn_step_counter += 1

        elif self.mode == 5:
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.target_replace_op)

            q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.state_: batch_memory[:, -self.state_size:],
                           self.state: batch_memory[:, :self.state_size]})

            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.state_size].astype(int)
            reward = batch_memory[:, self.state_size + 1]

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            self.sess.run(self.train_op, feed_dict={
                self.state: batch_memory[:, :self.state_size],
                self.q_target: q_target})

            self.learn_step_counter += 1

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    # ------------special part----------------------------------------------

    def saver(self):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(self.sess, 'network_weight_bias/my_net.ckpt')

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'network_weight_bias/my_net.ckpt')

    def show_graph(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('network_graph/', self.sess.graph)

    def reset_epsilon(self):
        if self.epsilon_increment != 0:
            self.epsilon = 0


class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory:
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.008  # 4x4: 0.003; 10x10: 0.001; 7x7: 0.002
    abs_err_upper = 10000000  # 4x4: 10^6->10^7

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    # 返回n个transitions，样本优先级权重（ISW）
    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, self.tree.data[0].size))
        ISWeights = np.empty((n, 1))

        pri_seg = self.tree.total_p / n  # 进行区间划分
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        # 下面这个式子的推倒和IS的变体有关；一定要先把memory学满
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        # convert to abs and avoid 0
        abs_errors += self.epsilon
        # 怕加了epsilon后超过权重上限
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def reset_beta(self):
        self.beta = 0.4


if __name__ == '__main__':
    sess = tf.Session()
    tf.summary.merge_all()
    RL_beta = DeepQNetwork(8, 4, sess=sess, mode=5)
    RL_beta.show_graph()
