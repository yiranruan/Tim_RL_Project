# =======================1=======================
# from envR import envR

# env = envR()
# env.reset()
# while True:
#     a = input()
#     if a == 'w':
#         env.step('u')
#     elif a == 's':
#         env.step('d')
#     elif a == 'a': 
#         env.step('l')
#     elif a == 'd':
#         env.step('r')
#     elif a == 'r':
#         env.reset()

# =======================2=======================
import tensorflow as tf
import numpy as np
class test:
    def __init__(self):
        self.memory_size = 10
        self.memory = np.zeros((self.memory_size, 2 * 2 + 2))
        print('memory:', self.memory)
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s, [a,r], s_))
        
        index = self.memory_counter % self.memory_size
        print('index:', index)
        self.memory[index, :] = transition

        self.memory_counter += 1
        return self.memory
    def slice(self):
        return self.memory[6, :]

if __name__ == "__main__":
    t = test()
    n_l1, w_initializer, b_initializer = \
                10, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
    observation = np.array((2,3))
    observation_ = np.array((3,4))
    a = 0
    r = 4
    s = tf.placeholder(tf.float32, [None, 2], name='s')
    s_ = tf.placeholder(tf.float32, [None, 2], name='s_')
    w = tf.get_variable('w', [2,10], initializer=w_initializer)
    b = tf.get_variable('b', [1, 10], initializer=b_initializer)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("sess1:", sess.run(w))
        print("sess2:", sess.run(b))
        observation = observation[np.newaxis, :]
        observation_ = observation_[np.newaxis, :]
        print("observation:", observation)
        print('s', sess.run(s, feed_dict={s:observation}))
        print("sess3:", sess.run(tf.matmul(s,w), feed_dict={s:observation}))
        print("sess4:", sess.run(tf.matmul(s,w)+b, feed_dict={s:observation}))
        print("sess5:", sess.run(tf.nn.relu(tf.matmul(s,w)+b), feed_dict={s:observation}))
        print('transition',np.hstack(((2,1),[a,r],(3,2))))
        for i in range(10):
            print("store:", t.store_transition((i,1),a,r,(3,2)))
        print('memory', t.slice())
        batch_memory = t.slice()
        batch_memory = batch_memory[np.newaxis, :]
        print('batch_memory', batch_memory[:,2].astype(int))
        batch_index = np.arange(10, dtype = np.int32)
        q_target = q_eval.copy()
        eval_act_index = batch_memory[:, 2].astype(int)
        print('q_target:', q_target[batch_index, eval_act_index])

# =======================3=======================
# s = (2,3)
# print('s',s)
# s_ = s.copy()
# print('s_',s_)
