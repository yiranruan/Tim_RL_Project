import tensorflow as tf
from envR import envR
from RL_brain import DeepQNetwork
from evaluate import Evaluate
import numpy as np

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
    
    train = False
    env = envR(show=False)
    path, cost, density, num_find_target, opt_cost= [],[],[],0,[]
    evaluate = Evaluate(rows=10, cols=10)
    restore_list = [1000000]
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      rows=env.rows,
                      cols=env.cols,
                      learning_rate=0.00001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.005,
                      output_graph=True
                      )
    for restore in restore_list:
        RL.save_network(train, str(restore)+'_4')
        succ = 0
        
        for episode in range(1000):
                pre_maps = env.reset()
                step = 0
                evaluate.set_start(start_pos=env.agent)
                evaluate.set_goals(
                    real_pos=env.maze.food_pos[0], fake_pos=env.maze.food_pos[1])
                
                for step in range(100):
                
                    action = RL.choose_action(pre_maps, train)
    
                    reward, done, action_ = env.step(env.action_translate(action))
                    
                    path.append(action_)
                    step += 1
                    if done:
                        succ += 1
                        cost, density, num_find_target, opt_cost = evaluation(evaluate, cost, density, num_find_target, opt_cost, path)
                        
                        break
                    pre_maps = env.get_maps()
        print('This is ',episode,'cost:', step,'succ', succ)
        print('average cost:', np.mean(cost), ' average density:', np.mean(density), ' deceptive extent:', num_find_target / succ)
        print('optimal cost:', np.mean(opt_cost))
        print()