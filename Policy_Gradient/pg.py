import numpy as np
import random
import tensorflow as tf
import envR
import time
from evaluate import Evaluate

# reproducible
random.seed(0)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
            restore=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.output_graph = output_graph
        self.restore = restore

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

        if self.output_graph:
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(
                "D:/code/python/computing project/pg_env2/", self.sess.graph)

        if self.restore:
            self.saver.restore(
                self.sess, "D:/code/python/computing project/pg_env2/model288_200/model288.ckpt")
        else:
            self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

      
        with tf.name_scope("fc1"):
            fc1 = tf.contrib.layers.fully_connected(inputs=self.tf_obs,
                                                    num_outputs=288,
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

    
        with tf.name_scope("fc3"):
            all_act = tf.contrib.layers.fully_connected(inputs=fc1,
                                                        num_outputs=self.n_actions,
                                                        activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

        # use softmax to convert to probability
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
           
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
           
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
            self.tf_obs: observation[np.newaxis, :]})
        # select action w.r.t the actions prob
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def clear_transition(self):
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data

    def learn(self, episode):
        # no need to discount reward here, already do reward shaping in env
        discounted_ep_rs_norm = np.array(self.ep_rs)
        
        # discounted_ep_rs_norm = np.ones_like(self.ep_rs)
        # discounted_ep_rs_norm *= np.sum(np.array(self.ep_rs))     

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm  # shape=[None, ]
        })

        # tf summary
        if self.output_graph and episode % 200 == 0:
            summary = self.sess.run(self.merged, feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
                self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
                self.tf_vt: discounted_ep_rs_norm  # shape=[None, ]
            })
            self.writer.add_summary(summary, episode)

        return discounted_ep_rs_norm



"""
the following parameters need to be changed before every run:
n_features -> the number of features
n_polyfeatures -> the number of polyfeatures, this will include bias
self.r -> render or not
self.u -> update or not
need to change the comment to create new weights or load from file
need to change the save file at the end of the script
need to modify the features in FeatureExtractors class as needed
"""


class TrainingAgent():
    def __init__(self):
        self.train_episode = 1000
        self.r = False  # render or not
        self.u = False   # update or not
        self.env = envR.envR(rows=10, cols=10, n_features=10)
        self.max_steps = 30  # (self.env.maze.c - 2) * (self.env.maze.r - 2)
        self.brain = PolicyGradient(n_actions=4, n_features=(self.env.maze.c * self.env.maze.r),
                                    learning_rate=0.0001, reward_decay=0.95, output_graph=False, restore=True)

        # used for evaluation
        self.evaluate = Evaluate(rows=10, cols=10, start_pos=(10, 1))
        self.num_fail = 0
        self.num_find_target = 0
        self.cost, self.density = [], []  # dp is deceptive_percentage
        self.opt_cost, self.opt_dp = [], []  # optimal deceptive path
        self.path = []
        self.reward = []

    def train(self):
        for episode in range(self.train_episode):
            # print("episode:", episode)
            self.env.reset()
            step = 0

            # used for evaluation
            self.evaluate.set_goals(
                real_pos=self.env.maze.food_pos[0], fake_pos=self.env.maze.food_pos[1])

            while True:
                # refresh env
                if self.r:  # and (episode % 1000 == 0 or episode == 999)  # and (episode > 199990)
                    print(str(self.env.maze))
                    time.sleep(0.1)
                state = self.env.get_maps().flatten()
                action = self.brain.choose_action(state)

                # action is transfered from int to String, used for evaluation
                reward, done, str_action = self.env.step(action)
                self.brain.store_transition(state, action, float(reward))

                # used for evaluation
                self.path.append(str_action)

                # print(action, reward)
                if done or step == self.max_steps:
                    ep_rs_sum = sum(self.brain.ep_rs)
                    print("episode:", episode, "  reward:", int(ep_rs_sum), " step:", step)

                    # used for evaluation
                    # self.reward.append(ep_rs_sum)
                    if step == self.max_steps:  # episode > 980000 and
                        self.num_fail += 1
                    else:
                        self.evaluation()
                    # opt_cost, _ = self.evaluate.get_optimal()
                    # self.opt_cost.append(opt_cost)
                    self.path = []

                    # update
                    if self.u:
                        vt = self.brain.learn(episode)
                    self.brain.clear_transition()
                    break

                step += 1

    def evaluation(self):
        cost, density, find_target_node = self.evaluate.evaluate_path(self.path)
        opt_cost, _ = self.evaluate.get_optimal()

        self.cost.append(cost)
        self.density.append(density)
        if find_target_node:
            self.num_find_target += 1
        self.opt_cost.append(opt_cost)
        # self.opt_dp.append(opt_dp)


if __name__ == '__main__':
    noob = TrainingAgent()
    noob.train()
    # save_path = noob.brain.saver.save(
    #     noob.brain.sess, "D:/code/python/computing project/pg_env2/model288_300/model288.ckpt")
    # print("Save to path: ", save_path)
    success = noob.train_episode - noob.num_fail
    print('fails:', noob.num_fail)
    print('success rate:', success / noob.train_episode)
    # print('average reward:', np.mean(noob.reward))
    # print('optimal rate:', noob.num_opt / len(noob.cost))
    print('average cost:', np.mean(noob.cost), ' average density:', np.mean(
        noob.density), ' deceptive extent:', noob.num_find_target / success)
    print('optimal cost:', np.mean(noob.opt_cost))
