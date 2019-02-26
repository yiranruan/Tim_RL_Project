import time
from env_beta import *
# from env_beta_view import Maze_view
from Neural_Network import *
import numpy as np


def map_test(height, width):
    cost = 0
    dead_node = 0
    for i in range(width):
        for j in range(height):
            env = Maze(mode=2, x_real=i, y_real=j, height=height, width=width)
            opt = env.optimal_distance((0, 0), (i, j))
            s = env.reset_02()
            step = 0
            while True:

                action = RL.choose_action(s)

                s_, reward, done = env.step_02(action)
                step += 1

                if done:
                    cost += (step - opt)
                    print((i, j), ':', (step - opt))
                    break
                elif step > 50:
                    dead_node += 1
                    print('Dead goal:', (i, j))
                    break

                s = s_
    return cost, dead_node


# def single_goal_test(x, y, height, width):
#     env = Maze_view(mode=2, x_real=x, y_real=y, height=height, width=width)
#     s = env.reset()
#     step = 0
#     while True:
#
#         action = RL.choose_action(s)
#         print(action)
#
#         s_, reward, done = env.step_02(action)
#         step += 1
#
#         env.update()
#
#         if done:
#             break
#
#         time.sleep(0.3)
#
#         s = s_
#
#     env.destroy()
#
#
# def simple_deceptive_test(x_r, y_r, x_f, y_f, h, w):
#     env = Maze(mode=3, x_real=x_r, y_real=y_r, x_fake=x_f, y_fake=y_f, height=h, width=w)
#     env.pre_work()
#     step = 0
#     truthful_steps_count = 0
#     action_list = []
#     for times in range(1):
#         s = env.reset_03()
#         while True:
#             if [s[-2:][0], s[-2:][1]] in env.truthful_steps_area:
#                 truthful_steps_count += 1
#
#             action = RL.choose_action(s)
#             print(action)
#             action_list.append(action)
#
#             s_, reward, done = env.step_03(action)
#
#             step += 1
#
#             if done[0] + done[1] == 2:
#                 break
#
#             s = s_
#         print('一共走了几步：', len(action_list))
#         print('show animation')
#         env02 = Maze_view(mode=3,
#                           x_real=x_r,
#                           y_real=y_r,
#                           x_fake=x_f,
#                           y_fake=y_f,
#                           height=h,
#                           width=w)
#         env02.display(env.truthful_steps_area, env.R_M_P)
#         for action in action_list:
#             env02.step_04(action)
#             env02.update()
#             time.sleep(0.3)
#         # env02.mainloop()
#         env02.destroy()
#
#     print('show statistics')
#     print('optimal cost:', env.optimal_distance(env.origin, env.fake_goal)
#           + env.optimal_distance(env.fake_goal, env.real_goal))
#     print('cost:', step)
#     print('total truthful step:', truthful_steps_count)


def deceptive_map_test(h, w, turns=None, RL=None):
    np.random.seed(3)
    test_num = 240
    total_steps = 0
    lose_count = 0
    opt_count = 0
    opt_deceptive_count = 0

    for times in range(test_num):

        while True:
            r_goal = np.random.randint(h, size=2)
            f_goal = np.random.randint(h, size=2)
            if not (r_goal == f_goal).all():
                break

        env = Maze(mode=3,
                   x_real=r_goal[0],
                   y_real=r_goal[1],
                   x_fake=f_goal[0],
                   y_fake=f_goal[1],
                   height=h,
                   width=w)

        env.pre_work()

        # print('real_goal:', r_goal[0], r_goal[1], 'fake_goal:', f_goal[0], f_goal[1])

        step_01 = 0
        truthful_steps_count = False
        action_list = []
        s = env.reset_03()
        while True:

            if [s[-2:][0], s[-2:][1]] in env.truthful_steps_area and [s[-2:][0], s[-2:][1]] not in env.R_M_P:
                truthful_steps_count = True

            action = RL.choose_action(s)

            action_list.append(action)

            s_, reward, done = env.step_03(action)

            step_01 += 1

            # if done[0] + done[1] == 2 or step_01 > 50:
            if done == 2 or step_01 > 50:
                break

            s = s_

        if len(action_list) > (manhattan_distance(env.origin, env.fake_goal) + manhattan_distance(env.fake_goal,
                                                                                                  env.real_goal)):
            lose_count += 1
        else:
            print('real_goal:', r_goal[0], r_goal[1], 'fake_goal:', f_goal[0], f_goal[1])
            opt_count += 1
            total_steps += len(action_list)
            if not truthful_steps_count:
                opt_deceptive_count += 1

    if turns is None:
        print('optimal_path_rate:', opt_count / test_num)
        # print('optimal_deceptive_rate:', opt_deceptive_count / opt_count)
        # print('average_steps:', total_steps / test_num)
    else:
        optimal_path_rate = opt_count/test_num
        if optimal_path_rate >= 0.85:
            RL.saver()
            my_file = open('storage_02/report_01', 'a')
            my_file.write('turns: ' + str(turns) + ' optimal_path_rate: ' + str(optimal_path_rate) + '\n')
            my_file.close()
            RL.saver()


if __name__ == '__main__':
    # 激活🧠
    RL = DeepQNetwork(7, 4, memory_size=100000, epsilon=1, mode=3)
    # 唤起记忆
    RL.restore()

    # single_goal_test(5, 5, 6, 6)
    # print(map_test(8, 8))
    # simple_deceptive_test(2, 3, 3, 3, 4, 4)
    deceptive_map_test(4, 4, RL=RL)
    # 3, 15, 17, 13, 19, 19
