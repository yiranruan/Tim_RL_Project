import tensorflow as tf
import numpy as np
import env.env_a3c
import a3c.evaluate

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = np.array(s)
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = np.array(s)
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

class Critic(object):
    def __init__(self, sess, n_features, lr):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.gamma = 0.9

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s ,s_ = np.array(s), np.array(s_)
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


class Agent(object):
    def __init__(self):
        self.sess = tf.Session()
        self.train_episode = 1000
        self.play_episode = 10
        self.max_steps = 100
        self.actor = Actor(self.sess,n_actions=4, n_features=4,lr=0.00005)
        self.critic = Critic(self.sess,n_features=4,lr=0.0001)
        self.evaluator = a3c.evaluate.Evaluate(10, 10)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def reset(self):
        self.env = env.env_a3c.Maze(mode=2)
        return self.env.canvas.coords(self.env.rect)

    def evaluate(self,fake_goal,real_goal,path):
        self.evaluator.set_goals(real_pos=(real_goal[0], real_goal[1]),
                           fake_pos=(fake_goal[0], fake_goal[1]))
        self.evaluator.evaluate_path(path)

    def save(self,i_episode):
        self.saver.save(self.sess, 'my_model/save_net_%i.ckpt' % i_episode)

    def train(self):
        for i_episode in range(self.train_episode):
          #  print("episode:", i_episode)
            s = self.reset()
            step = 0
            track_r = []
            episode_reward = []
            while True:
                self.env.render()
                action = self.actor.choose_action(s)
                # print(action)
                s_, reward, done = self.env.step(action)
                #print(s_,reward,done)
                track_r.append(reward)
                td_error = self.critic.learn(s, reward, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
                self.actor.learn(s, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
                # print(state, action, reward, new_state)
                s = s_
                if done or step == 100:
                    ep_rs_sum = sum(track_r)
                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                    #episode_reward.append(ep_rs_sum)
                    print("episode:", i_episode, "  reward:", int(ep_rs_sum))
                    break
                step += 1
            self.env.destroy()

            if i_episode > 0 and i_episode % 10000 == 0:
                self.save(i_episode)

    def play(self):
        for i_episode in range(self.play_episode):
            print("episode:", i_episode)
            s = self.reset()
            step = 0
            track_r = []
            path = []
            while True:
                #self.env.render()
                real_goalx, real_goaly = int((self.env.canvas.coords(self.env.real_goal)[
                                                  0] - 5) / 40), int(
                    (self.env.canvas.coords(self.env.real_goal)[1] - 5) / 40)
                real_goal = [real_goalx,real_goaly]
                fake_goalx, fake_goaly = int((self.env.canvas.coords(self.env.fake_goal)[
                                                  0] - 5) / 40), int(
                    (self.env.canvas.coords(self.env.fake_goal)[1] - 5) / 40)
                fake_goal = [fake_goalx,fake_goaly]

                action = self.actor.choose_action(s)
                # print(action)
                s_, reward, done = self.env.step(action)
                track_r.append(reward)
                path.append(action)
                if done or step == self.env.MAZE_H * self.env.MAZE_W:
                    ep_rs_sum = sum(track_r)
                    print("episode:", i_episode, "  reward:", int(ep_rs_sum))
                    break
                step += 1
                s = s_
            self.evaluate(real_goal=real_goal,fake_goal=fake_goal,path = path)
            self.env.destroy()

if __name__ == '__main__':
    agent = Agent()
    agent.train()
    #agent.play()




