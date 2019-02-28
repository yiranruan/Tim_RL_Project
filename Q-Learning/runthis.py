from envR import envR
from RL_brain import QLearningTable
import time
from evaluate import Evaluate
import numpy as np

def test(RL):
    env = envR(show=False)
    path, cost, density, num_find_target, opt_cost= [],[],[],0,[]
    evaluate = Evaluate(rows=10, cols=10)
    train = False
    succ = 0
    print("****************************************************")
    for episode in range(100):
        pre_maps = env.reset()
        step = 0
        evaluate.set_start(start_pos=env.agent)
        evaluate.set_goals(
                real_pos=env.maze.food_pos[0], fake_pos=env.maze.food_pos[1])
        # print("****************************************************")
        # print("EPISODE ", episode)
        # start_test = time.time()
        for step in range(100):

            action = RL.choose_action(str(pre_maps), train)

            reward, done,action_ = env.step(action)

            path.append(action_)

            step += 1
            if done:
                succ += 1
                cost, density, num_find_target, opt_cost = evaluation(evaluate, cost, density, num_find_target, opt_cost, path)
                path = []
                break
            pre_maps = env.get_maps()
    print('This is ',episode,'cost:', step,'succ', succ)
    print('average cost:', np.mean(cost), ' average density:', np.mean(density), ' deceptive extent:', num_find_target / succ)
    print('optimal cost:', np.mean(opt_cost))
    print()

def evaluation(evaluate, cost, density, num_find_target, opt_cost, path):
    cost_, density_, find_target_node_ = evaluate.evaluate_path(path)
    opt_cost_, _ = evaluate.get_optimal()

    cost.append(cost_)
    density.append(density_)
    if find_target_node_:
        num_find_target += 1
    opt_cost.append(opt_cost_)
    return cost, density, num_find_target, opt_cost

if __name__ == "__main__":
    # r = input('times: ')
    r = '50000'
    save_list = [100, 50000]
    # ,10000,50000,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000
    train = True
    env = envR(show=False)
    RL = QLearningTable(env.action_space, learning_rate=0.1)
    # step = 0
    # succ = 0
    # start = time.time()
    for episode in range(int(r)):
        pre_maps = env.reset()
        
        for i in range(100):

            action = RL.choose_action(str(pre_maps), train)

            reward, done, action_ = env.step(action)

            RL.learn(str(pre_maps),action, reward, str(env.get_maps()),done)

            pre_maps = env.get_maps()

            if done:
                break

            # step += 1
        print((episode+1))
        if (episode+1) in save_list:
            print("This is", episode+1)
            test(RL)
    print('Training Over!')