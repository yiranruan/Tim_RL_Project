from envR import envR
from RL_brain import DeepQNetwork

import time


if __name__ == "__main__":
    r = 1000000
    index_ = '_4'
    save_list = [10,20,30,40,50,60,70,80,90,100,1000,5000,10000,50000,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]

    train = True
    env = envR(show=False)

    RL = DeepQNetwork(env.n_actions, env.n_features,
                          rows=env.rows,
                          cols=env.cols,
                          learning_rate=0.00001,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=5000,
                          e_greedy_increment=0.0005,
                          output_graph=False
                          )

    step = 0
    succ = 0
    total_cost = 0
    for episode in range(int(r)):
        pre_maps = env.reset()
        for i in range(100):

            isVisited = True
                # print('episode:',episode)
            action = RL.choose_action(pre_maps, train)
                # print(RL.epsilon)
            reward, done, action_ = env.step(env.action_translate(action))

                # if not isVisited:
                # print("This is", episode)
                # print("This is step:", i, "rewards:",reward)
                # print("rewards:",reward)
            RL.store_transition(pre_maps, action, reward, env.get_maps())

            if (step > 2000) and (step % 5 == 0):
                RL.learn()

                # print('epsilon:',RL.epsilon)
            pre_maps = env.get_maps()

            step += 1
                
            if done:
                succ += 1
                break
        # if (episode+1) % 1000 == 0:
        total_cost += env.total_cost
        if ((episode+1) in save_list) or ((episode+1) % 1000 == 0):
            f = open("./reward_4.txt",'a')
            f.write('reward '+str(episode+1)+' '+str(total_cost)+'\n')
            f.write('succ '+str(episode+1)+' '+str(succ)+'\n')
            f.write('time '+str(episode+1)+' '+str(time.time())+'\n')
            f.close()
            total_cost = 0
            RL.save_network(train, str(episode+1)+index_)
        if (episode+1) in save_list:
            RL.plot_cost(str(episode+1))
    print('Training Over!')

 
