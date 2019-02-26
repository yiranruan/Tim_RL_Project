import matplotlib.pyplot as plt
from env_beta_view import *
import numpy as np

episodes = []
training_steps_natural = []
total_training_steps_natural = []
training_steps_fixed = []
total_training_steps_fixed = []
training_steps_double = []
total_training_steps_double = []
training_steps_prioritized = []
total_training_steps_prioritized = []

index = []
loss = []


# ---------------------------------------reload part------------------------------------
def reload_natural(episode):
    my_file = open('4x4_natural_mf/episodes', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        episodes.append(int(a))
    my_file.close()

    my_file = open('4x4_natural_mf/training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        training_steps_natural.append(int(a))
    my_file.close()

    my_file = open('4x4_natural_mf/total_training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        total_training_steps_natural.append(int(a))
    my_file.close()


def reload_fixed(episode):
    my_file = open('4x4_fixed_mf/training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        training_steps_fixed.append(int(a))
    my_file.close()

    my_file = open('4x4_fixed_mf/total_training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        total_training_steps_fixed.append(int(a))
    my_file.close()


def reload_double(episode):
    my_file = open('4x4_double_mf/training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        training_steps_double.append(int(a))
    my_file.close()

    my_file = open('4x4_double_mf/total_training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        total_training_steps_double.append(int(a))
    my_file.close()


def reload_prioritized(episode):
    my_file = open('4x4_prio_mf/training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        training_steps_prioritized.append(int(a))
    my_file.close()

    my_file = open('4x4_prio_mf/total_training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        num_01 = max((int(a) - 98000), 0)
        total_training_steps_prioritized.append(num_01)
    my_file.close()


def reload_storage(episode):
    my_file = open('storage/episodes', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        episodes.append(int(a))
    my_file.close()

    my_file = open('storage/training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        training_steps_natural.append(int(a))
    my_file.close()

    my_file = open('storage/total_training_steps_natural', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        total_training_steps_natural.append(int(a))
    my_file.close()


def reload_storage_02(episode):
    # my_file = open('storage_02/index', 'r')
    # for i in range(episode):
    #     a = my_file.readline()
    #     a = a[:(len(a))]
    #     index.append(int(a))
    # my_file.close()

    my_file = open('storage_02/loss', 'r')
    for i in range(episode):
        a = my_file.readline()
        a = a[:(len(a))]
        loss.append(float(a))
    my_file.close()


# --------------------------------------plot part------------------------------------------


def plot_natural():
    plt.figure(num=1)
    plt.plot(episodes, training_steps_natural, c='r', label='DQN')
    plt.legend(loc='best')
    plt.ylabel('training steps')
    plt.xlabel('episodes')
    plt.grid()

    plt.figure(num=2)
    plt.plot(episodes, total_training_steps_natural, c='r', label='total DQN')
    plt.legend(loc='best')
    plt.ylabel('total training steps')
    plt.xlabel('episodes')
    plt.grid()

    plt.show()


def plot_loss():
    index = np.arange(2000000)
    plt.figure(num=1)
    plt.scatter(index, loss, c='r', label='DQN', alpha=.1)
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('learning steps')
    plt.grid()

    plt.show()


def plot_graph(natural=False, fixed=False, double=False, prioritized=False):
    plt.figure(num=1)
    if natural:
        plt.plot(episodes, total_training_steps_natural, c='r', label='natural DQN')

    if fixed:
        plt.plot(episodes, total_training_steps_fixed, c='b', label='fixed DQN')

    if double:
        plt.plot(episodes, total_training_steps_double, c='g', label='double DQN')

    if prioritized:
        plt.plot(episodes, total_training_steps_prioritized, c='y', label='prioritized DQN')
    plt.legend(loc='best')
    plt.ylabel('total training steps')
    plt.xlabel('episodes')
    plt.grid()

    plt.show()


def plot_bar():
    n = 4
    x = np.arange(n)
    y = np.array([0.87, 0.9, 0.841, 0.863])

    plt.bar(x, y)
    plt.ylabel('optimal rate')
    plt.xticks([0, 1, 2, 3], [r'$natural$', r'$fixed$', r'$double$', r'$prioritized$'])

    for x, y in zip(x, y):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom')

    plt.show()


if __name__ == '__main__':
    episode_num = 2000000

    # reload_storage(episode_num)
    # reload_natural(episode_num)
    # reload_fixed(episode_num)
    # reload_double(episode_num)
    # reload_prioritized(episode_num)

    # plot_graph(natural=True, fixed=True, double=True, prioritized=True)
    # plot_natural()

    # reload_storage_02(episode_num)
    # print(loss)
    # plot_loss()

    plot_bar()

