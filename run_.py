from envR import envR
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        s = env.reset()
        
        while True:
            # env.render()
            print("This is",episode)
            action = RL.choose_action(str(s))

            s_, reward, done = env.step(action_translate(action))
            RL.learn(str(s), action, reward, str(s_), done)
            s = s_
            if done:
                break
    print('Game Over!')

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
    RL = QLearningTable(actions = list(range(env.n_actions)))

    update()
    # env.after(100, update)