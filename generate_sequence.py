import random
import argparse
from jericho import *

def play_game(env, use_walkthrough):
    description, info = env.reset()
    done = False
    sequence = []
    step = 0
    walkthrough = env.get_walkthrough()
    print(len(walkthrough))


    while not done:
        if use_walkthrough:
            # Use the walkthrough for the game if provided
            action = walkthrough[step]
            step += 1
        else:
            # Play randomly
            valid_actions = env.get_valid_actions()
            action = random.choice(valid_actions) if valid_actions else 'wait'

        obv = (env.get_valid_actions(), description, env.get_player_location().name, list(map(lambda x: x.name, env.get_inventory())))
        description, reward, done, info = env.step(action)
        sequence.append((obv, reward, action))
        print(step)

    return sequence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkthrough", help="Use the game's walkthrough instead of random actions", action="store_true")
    args = parser.parse_args()

    env = FrotzEnv("z-machine-games-master/jericho-game-suite/enchanter.z3")
    game_sequence = play_game(env, args.walkthrough)
    for step in game_sequence:
        print(step)

if __name__ == "__main__":
    main()

