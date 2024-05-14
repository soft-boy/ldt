import jericho
import random
import os
import argparse
import numpy as np
import time
import pickle

class TrajectoryGenerator:
    def __init__(self, game_file_path, num_random_steps, num_repeats, num_seeds, output_file, gold_trajectory_only=False):
        self.game_file_path = game_file_path
        self.output_file = output_file

        self.num_random_steps = num_random_steps
        self.num_repeats = num_repeats
        self.num_seeds = num_seeds
        
        self.percentages = list(range(0, 100, 5))
        self.gold_trajectory_only = gold_trajectory_only
        self.trajectories = []

        # Initialize the environment and get the walkthrough
        self.env = jericho.FrotzEnv(self.game_file_path)
        self.walkthrough = self.env.get_walkthrough()
        
        # Initialize the cache
        self.action_cache = {}
        self.cache_file = f"{os.path.splitext(os.path.basename(self.game_file_path))[0]}_cache.pkl"
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.action_cache = pickle.load(f)
                print(f"Cache loaded from {self.cache_file}")

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.action_cache, f)
            print(f"Cache saved to {self.cache_file}")

    def run(self):
        print("Running on a single core.")

        seeds = [i for i in range(1, self.num_seeds + 1)]
        
        for seed in seeds:
            print(f"Generating trajectories for seed {seed}")
            result = self.generate_trajectories_for_seed(seed)
            self.trajectories.extend(result)

        # Save the generated trajectories
        self.save_trajectories()
        print(f"Trajectories saved to {self.output_file}")

    def get_cached_valid_actions(self):
        state_hash = self.env.get_world_state_hash()
        if state_hash not in self.action_cache:
            self.action_cache[state_hash] = self.env.get_valid_actions(use_parallel=True)
        return self.action_cache[state_hash]

    def record_step(self, action, trajectory):
        """
        Execute an action in the environment and record the step.
        
        :param action: Action to take.
        :param trajectory: List to append the step information.
        :return: Tuple (done, reward).
        """
        state, reward, done, info = self.env.step(action)
        # print(action)
        observation = {
            "cand": self.get_cached_valid_actions(),
            "msg": state
        }
        trajectory.append((observation, reward, action))
        return done, reward

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
            done, _ = self.record_step(action, trajectory)
            if done:
                break
        return trajectory, done

    def take_random_steps(self, num_random_steps, trajectory):
        """
        Take random steps in the environment.
        
        :param num_random_steps: Number of random steps to take.
        :param trajectory: List to append the random step information.
        :return: List of recorded steps.
        """
        done = False
        for _ in range(num_random_steps):
            actions_list = self.get_cached_valid_actions()
            if len(actions_list) == 0:
                print("should not happen")
                print("No valid actions found.")
                # print(self.env.)
            random_action = random.choice(actions_list)
            done, _ = self.record_step(random_action, trajectory)
            if done:
                break
        return trajectory, done

    def generate_trajectories_for_seed(self, seed):
        """
        Generate trajectories for a specific seed by following the golden path for a certain percentage and then taking random steps.
        
        :param seed: Seed for environment initialization.
        :return: List of generated trajectories.
        """
        trajectories = []

        # Append the golden trajectory
        np.random.seed(seed)
        random.seed(seed)
        self.env.reset()
        print(f"Generating golden trajectory for seed {seed}")
        start_time = time.time()
        golden_trajectory, done = self.follow_walkthrough(len(self.walkthrough))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Golden trajectory generated in {elapsed_time:.2f} seconds")
        trajectories.append(golden_trajectory)
        self.save_cache()
        if self.gold_trajectory_only:
            return trajectories

        total_iterations = len(self.percentages) * self.num_repeats
        count = 1
        for percentage in self.percentages:
            num_steps_to_follow = int(len(self.walkthrough) * (percentage / 100.0))
            for _ in range(self.num_repeats):
                self.env.reset()
                start_time = time.time()
                trajectory, done = self.follow_walkthrough(num_steps_to_follow)
                if not done:
                    trajectory, done = self.take_random_steps(self.num_random_steps, trajectory)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Trajectory {count}/{total_iterations} generated in {elapsed_time:.2f} seconds")
                count += 1
                trajectories.append(trajectory)
                self.save_cache()

        return trajectories

    def save_trajectories(self):
        """
        Save trajectories to a file.
        """
        with open(self.output_file, 'w') as file:
            for trajectory in self.trajectories:
                for observation, reward, action in trajectory:
                    file.write(f"{observation}\t{reward}\t{action}\n")
                file.write("\n")  # Separate trajectories by a newline

def main():
    parser = argparse.ArgumentParser(description='Generate training data for Jericho games.')
    parser.add_argument('-i', '--game_file', type=str, default="z-machine-games-master/jericho-game-suite/enchanter.z3", help='Path to the game file.')
    parser.add_argument('-o', '--output_file', type=str, default='trajectories.txt', help='Output file name for the trajectories.')
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
        output_file=args.output_file,
        gold_trajectory_only=args.gold
    )
    generator.run()

if __name__ == "__main__":
    main()
