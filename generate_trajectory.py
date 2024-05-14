import jericho
import random
import os
import argparse
import time
import pickle

from collections import defaultdict

class TrajectoryGenerator:
    def __init__(self, game_file_path, num_random_steps, num_repeats, num_seeds, output_dir, cache_dir="", gold_trajectory_only=False):
        self.game_file_path = game_file_path
        self.output_dir = output_dir

        self.num_random_steps = num_random_steps
        self.num_repeats = num_repeats
        self.num_seeds = num_seeds
        
        self.percentages = list(range(0, 101, 5))
        self.gold_trajectory_only = gold_trajectory_only
        self.trajectories = defaultdict(list)

        # Initialize the environment and get the walkthrough
        self.env = jericho.FrotzEnv(self.game_file_path)
        self.walkthrough = self.env.get_walkthrough()
        
        # Initialize the cache
        self.is_cache_save_required = False
        self.action_cache = {}
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_file = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(self.game_file_path))[0]}_valid_action_list_cache.pkl")
        self.load_cache()

        # Initialize the state and reward
        self.state = None
        self.info = None
        self.reward = None

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.action_cache = pickle.load(f)
                print(f"[SUCCESS] Cache loaded from {self.cache_file}")
                print("Number of cached states:", len(self.action_cache))
                print() # empty line
        else:
            print(f"[WARNING] Cache file not found, new cache file will be created in {self.cache_file}")
            print(f"Cache file: {self.cache_file}\n")

    def save_cache(self):
        if not self.is_cache_save_required:
            print("No cache save required")
            return
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.action_cache, f)
            print(f"Cache updated and saved")

    def reset_env(self, seed=None):
        self.reward = 0
        self.state, self.info  = self.env.reset()
        self.env.seed(seed)
        
    def run(self):
        seeds = [i for i in range(0, self.num_seeds)]
        seeds[0] = self.env.bindings['seed'] # seed for the golden path

        print("Using the following seeds:", seeds)
        print("Golden path seed:", seeds[0])
        
        for seed in seeds:
            print(f"\nGenerating trajectories for seed: {seed}")
            self.generate_trajectories_for_seed(seed)

        # Save the generated trajectories
        self.save_trajectories()

    def get_cached_valid_actions(self):
        state_hash = self.env.get_world_state_hash()
        if state_hash not in self.action_cache:
            self.action_cache[state_hash] = self.env.get_valid_actions(use_parallel=True)
            self.is_cache_save_required = True
        return self.action_cache[state_hash]

    def take_record_step(self, action, trajectory):
        """
        Execute an action in the environment and record the step.
        
        :param action: Action to take.
        :param trajectory: List to append the step information.
        :return: Tuple (done, reward).
        """
        hints = self.get_cached_valid_actions()
        next_state, next_reward, done, next_info = self.env.step(action)
        observation = {
            "msg": self.state,
            "hints": hints,
        }
        trajectory.append((observation, action, self.reward, self.info))
        
        # Update the state and reward
        self.state = next_state
        self.reward = next_reward
        self.info = next_info

        return done

    def follow_walkthrough(self, num_steps_to_follow):
        """
        Follow the walkthrough for a certain number of steps.
        
        :param num_steps_to_follow: Number of steps to follow the walkthrough.
        :return: List of recorded steps.
        """
        done = False
        trajectory = []
        for step in range(num_steps_to_follow):
            action = self.walkthrough[step]
            done = self.take_record_step(action, trajectory)
            if done:
                break
        return trajectory, done

    def take_random_steps(self):
        """
        Take random steps in the environment.

        :param trajectory: List to append the random step information.
        :return: List of recorded steps.
        """
        done = False
        trajectory = []
        for _ in range(self.num_random_steps):
            actions_list = self.get_cached_valid_actions()
            if len(actions_list) == 0:
                actions_list.append("QUIT")
                print("[WARNING] Manual action added: QUIT")
            random_action = random.choice(actions_list)
            done = self.take_record_step(random_action, trajectory)
            if done:
                break
        return trajectory, done

    def generate_trajectories_for_seed(self, seed):
        """
        Generate trajectories for a specific seed by following the golden path for a certain percentage and then taking random steps.
        
        :param seed: Seed for environment initialization.
        :return: List of generated trajectories.
        """
        # Append the golden trajectory
        if self.gold_trajectory_only:
            self.reset_env(seed)
            start_time = time.time()
            golden_trajectory, done = self.follow_walkthrough(len(self.walkthrough))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Golden trajectory (for seed: {seed}) generated in {elapsed_time:.2f} s,", \
                golden_trajectory[-1][-1], \
                end=", ")
            self.save_cache()
            self.trajectories[(seed, 100)].append(golden_trajectory)
            return
            
        count = 1
        total_iterations = (len(self.percentages) - 1) * self.num_repeats + 1
        for percentage in self.percentages:
            num_steps_to_follow = int(len(self.walkthrough) * (percentage / 100.0))
            for _ in range(self.num_repeats if percentage < 100 else 1):
                self.reset_env(seed)
                walkthrough_trajectory, random_trajectory  = [], []

                start_time = time.time()
                walkthrough_trajectory, done = self.follow_walkthrough(num_steps_to_follow)
                if not done:
                    random_trajectory, done = self.take_random_steps()
                end_time = time.time()
                elapsed_time = end_time - start_time

                trajectory = walkthrough_trajectory + random_trajectory
                print(f"(Seed: {seed}) Trajectory {count}/{total_iterations} generated in {elapsed_time:.2f} s,", \
                      f"gold steps ({percentage}%): {len(walkthrough_trajectory)}, rand steps: {len(random_trajectory)},", \
                      trajectory[-1][-1], \
                      end=", ")
                
                count += 1
                self.save_cache()
                self.trajectories[(seed, percentage)].append(trajectory)
                

    def save_trajectories(self):
        """
        Save trajectories to a file.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # save to text file
        text_output_file = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(self.game_file_path))[0]}.txt")
        with open(text_output_file, 'w') as file:
            for _, trajectories in self.trajectories.items():
                for trajectory in trajectories:
                    for observation, action, reward, info in trajectory:
                        file.write(f"{observation['msg']}\t{observation['hints']}\t{action}\t{reward}\t{info}\n\n")
                    file.write("\n")

        # save to pickle file
        pickle_output_file = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(self.game_file_path))[0]}.pkl")
        with open(pickle_output_file, 'wb') as file:
            pickle.dump(self.trajectories, file)

        print(f"\n[SUCCESS] Trajectories saved to {text_output_file} and {pickle_output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate training data for Jericho games.')
    parser.add_argument('-i', '--game_file', type=str, default="z-machine-games-master/jericho-game-suite/enchanter.z3", help='Path to the game file.')
    parser.add_argument('-o', '--output_dir', type=str, default='trajectories', help='Directory to save the generated trajectories.')
    parser.add_argument('-c', '--cache_dir', type=str, default='.env', help='Directory to save the valid action list cache file.')
    parser.add_argument('-s', '--num_seeds', type=int, default=5, help='Number of seeds for environment initialization.')
    parser.add_argument('-r', '--num_repeats', type=int, default=10, help='Number of repeats for each percentage of steps.')
    parser.add_argument('--num_random_steps', type=int, default=100, help='Number of random steps to take after following the walkthrough.')
    parser.add_argument('--gold', action='store_true', help='Generate golden trajectory only.')
    
    args = parser.parse_args()

    generator = TrajectoryGenerator(
        game_file_path=args.game_file,
        num_random_steps=args.num_random_steps,
        num_repeats=args.num_repeats,
        num_seeds=args.num_seeds,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        gold_trajectory_only=args.gold
    )
    generator.run()

if __name__ == "__main__":
    main()


# copy command: scp ./generate_trajectory.py flip:~/final-project/ldt