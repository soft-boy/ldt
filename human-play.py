from jericho import *

env = FrotzEnv("z-machine-games-master/jericho-game-suite/enchanter.z3")
observation, info = env.reset()
done = False

while not done:
    print(observation)
    valid_actions = env.get_valid_actions()
    print('hints: ', valid_actions)
    action = input('Take action: ')
    observation, reward, done, info = env.step(action)
    print('Current Score: ', info['score'])

print('Game complete. Scored', info['score'], 'out of', env.get_max_score())