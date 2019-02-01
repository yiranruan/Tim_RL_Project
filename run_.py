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
    train = True
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )

    step = 0
    for episode in range(300):
        env = envR()
        observation = env.reset()
        
        while True:
            # env.render()
            print("This is", episode)

            action = RL.choose_action(observation, train)

            observation_, reward, done = env.step(action_translate(action), train)

            RL.store_transition(observation, action, reward, observation_)
            
            if (step >200) and (step % 5 == 0):
                RL.learn()
            
            observation = observation_

            if done:
                break
            step += 1
    train = False
    print('Training Over!')
    # env.after(100, update)

    for episode in range(5):
        env = envR()
        s = env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(100):

            # Take the action (index) that have the maximum expected future reward given that state
            action = RL.choose_action(s, train)
            print(action)
            s_, reward, done = env.step(action_translate(action), train)
            
            if done:
                print('s_:',s_)
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                # We print the number of step it took.
                print("Number of steps",step)
                print('success')
                break
            s = s_