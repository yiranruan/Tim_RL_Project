from envR import envR
from RL_brain import DeepQNetwork

def update():
    pass

def action_translate(action):
    if action == 0:
        return 'u'
    elif action == 1:
        return 'd'
    elif action == 2:
        return 'l'
    elif action == 3:
        return 'r'

if __name__ == "__main__":
    env = envR()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    step = 0
    for episode in range(1000):
        observation = env.reset()
        
        while True:
            # env.render()
            print("This is", episode)

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action_translate(action))

            RL.store_transition(observation, action, reward, observation_)
            
            if (step >200) and (step % 5 == 0):
                RL.learn()
            
            observation = observation_

            if done:
                break
            step += 1
    print('Game Over!')
    # env.after(100, update)