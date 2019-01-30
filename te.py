from envR import envR

env = envR()
env.reset()
while True:
    a = input()
    if a == 'w':
        env.step('u')
    elif a == 's':
        env.step('d')
    elif a == 'a':
        env.step('l')
    elif a == 'd':
        env.step('r')
    elif a == 'r':
        env.reset()