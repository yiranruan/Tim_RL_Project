from env_beta import Maze
from Neural_Network import *
import tensorflow as tf
from test import *


def model_based_version(x_r, y_r, x_f, y_f, h, w, episode_number,
                        learn_threshold, observer, RL):
    step_01 = 0
    episodes = []
    steps_per_episode = []
    total_steps = []

    env = Maze(mode=3,
               x_real=x_r,
               y_real=y_r,
               x_fake=x_f,
               y_fake=y_f,
               height=h,
               width=w)

    for episode in range(episode_number):
        step_02 = 0
        print('episode:', episode + 1, 'epsilon_start:', RL_nature.epsilon)
        # print('episode:', episode + 1, 'epsilon_start:', RL_nature.epsilon,
        #       'alpha:', RL.memory.alpha, 'beta:', RL.memory.beta)
        state = env.reset_03()

        while True:
            action = RL.choose_action(state)

            state_, reward, done = env.step_03(action)

            RL.store_transition(state, action, reward, state_)

            # 学习
            if (step_01 > learn_threshold) and (step_01 % 5 == 0):
                RL.learn()
                if observer:
                    rs = sess.run(RL.merged, feed_dict={
                        RL.state: state[np.newaxis, :],
                        RL.reward: np.array([reward]),
                        RL.action: np.array([action]),
                        RL.target_q: sess.run(RL.q_eval, feed_dict={RL.state: state_[np.newaxis, :]})})
                    RL.writer.add_summary(rs, step_01)

            state = state_

            if done[0] + done[1] == 2:
                episodes.append(episode)
                total_steps.append(step_01)
                steps_per_episode.append(step_02)
                break

            step_01 += 1
            step_02 += 1

        # print('epsilon_end:', RL_nature.epsilon, 'alpha:', RL.memory.alpha, 'beta:', RL.memory.beta)

    return np.vstack((episodes, steps_per_episode, total_steps))


def model_free_version(h, w, learn_threshold, RL=None):
    step_01 = 0
    np.random.seed(9)
    turns_count = []
    cost_count = []
    total_cost_count = []

    for turns in range(24000):
        step_02 = 0
        if (turns+1) % 10 == 0:
            print('turns:', turns+1)

        while True:
            r_goal = np.random.randint(h, size=2)
            f_goal = np.random.randint(h, size=2)
            if not (r_goal == f_goal).all():
                break

        # print('real_goal:', r_goal[0], r_goal[1], 'fake_goal:', f_goal[0], f_goal[1])

        env = Maze(mode=3,
                   x_real=r_goal[0],
                   y_real=r_goal[1],
                   x_fake=f_goal[0],
                   y_fake=f_goal[1],
                   height=h,
                   width=w)

        # update some configuration
        # RL.memory.reset_beta()
        # if turns < 35000:
        #     RL.reset_epsilon()
        state = env.reset_03()

        while True:

            action = RL.choose_action(state)

            state_, reward, done = env.step_03(action)

            RL.store_transition(state, action, reward, state_)

            # 学习
            if (step_01 > learn_threshold) and (step_01 % 5 == 0):
                RL.learn()

            state = state_

            if done == 2 or step_02 > 6000:  # 4x4: 3500; 10x10: 6000
                break

            step_01 += 1
            step_02 += 1

        turns_count.append(turns)
        cost_count.append(step_02)
        total_cost_count.append(step_01)

        # if turns > 500000 and (turns+1) % 10000 == 0:
        #     deceptive_map_test(10, 10, turns, RL=RL)

    # RL.my_file.close()
    # RL.my_file_02.close()
    return np.vstack((turns_count, cost_count, total_cost_count))


# ------------------------------------store data--------------------------------------
def store_data(natural=None, reference=None):
    my_file = open('storage/episodes', 'a')
    for i in natural[0, :]:
        my_file.write(str(i) + '\n')
    my_file.close()

    my_file = open('storage/training_steps_natural', 'a')
    for i in natural[1, :]:
        my_file.write(str(i) + '\n')
    my_file.close()

    my_file = open('storage/total_training_steps_natural', 'a')
    for i in natural[2, :]:
        my_file.write(str(i) + '\n')
    my_file.close()

    if reference is not None:
        my_file = open('storage/reference_training_steps', 'a')
        for i in reference[1, :]:
            my_file.write(str(i) + '\n')
        my_file.close()

        my_file = open('storage/reference_total_training_steps', 'a')
        for i in reference[2, :]:
            my_file.write(str(i) + '\n')
        my_file.close()


if __name__ == '__main__':
    sess = tf.Session()
    observer = True
    learn_threshold = 200
    RL_nature = DeepQNetwork(7, 4, memory_size=100000, epsilon=0.8,  # epsilon_increment=0.07,
                             observer_mode=observer, sess=sess, mode=3)
    # RL_nature.restore()
    sess.run(tf.global_variables_initializer())
    # RL_nature.show_graph()
    # his_natural = model_based_version(0, 9, 9, 0, 10, 10, 1000, learn_threshold=learn_threshold,
    #                                   observer=observer, RL=RL_nature)
    his_natural = model_free_version(4, 4, learn_threshold=learn_threshold, RL=RL_nature)
    # store_data(his_natural)
    RL_nature.saver()


