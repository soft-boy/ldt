import jericho
import random
import os
import argparse
from multiprocessing import Pool

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
        self.walkthrough = self.env.get_walkthrough().split('\n')

    def run(self):
        # Prepare parameters for multiprocessing
        seeds = [i for i in range(1, self.num_seeds + 1)]
        
        # Generate trajectories using multiprocessing
        with Pool() as pool:
            all_trajectories = pool.map(self.generate_trajectories_for_seed, seeds)

        # Flatten the list of lists
        self.trajectories = [trajectory for sublist in all_trajectories for trajectory in sublist]

        # Save the generated trajectories
        self.save_trajectories()
        print(f"Trajectories saved to {self.output_file}")

    def record_step(self, action, trajectory):
        """
        Execute an action in the environment and record the step.
        
        :param action: Action to take.
        :param trajectory: List to append the step information.
        :return: Tuple (done, reward).
        """
        state, reward, done, info = self.env.step(action)
        observation = {
            "cand": self.env.get_valid_actions(),
            "msg": state,
            "desc": self.env.get_description(),
            "inv": self.env.get_inventory()
        }
        trajectory.append((observation, reward, action))
        return done, reward

    def follow_walkthrough(self, num_steps_to_follow):
        """
        Follow the walkthrough for a certain number of steps.
        
        :param num_steps_to_follow: Number of steps to follow the walkthrough.
        :return: List of recorded steps.
        """
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
        for _ in range(num_random_steps):
            random_action = random.choice(self.env.get_valid_actions())
            done, _ = self.record_step(random_action, trajectory)
            if done:
                break
        return trajectory

    def generate_trajectories_for_seed(self, seed):
        """
        Generate trajectories for a specific seed by following the golden path for a certain percentage and then taking random steps.
        
        :param seed: Seed for environment initialization.
        :return: List of generated trajectories.
        """
        trajectories = []

        # Append the golden trajectory
        self.env.reset(seed)
        golden_trajectory, done = self.follow_walkthrough(len(self.walkthrough))
        trajectories.append(golden_trajectory)
        if self.gold_trajectory_only:
            return trajectories
        
        for percentage in self.percentages:
            num_steps_to_follow = int(len(self.walkthrough) * (percentage / 100.0))
            for _ in range(self.num_repeats):
                self.env.reset(seed)
                trajectory, done = self.follow_walkthrough(num_steps_to_follow)
                if not done:
                    trajectory = self.take_random_steps(self.num_random_steps, trajectory)
                trajectories.append(trajectory)

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
    parser.add_argument('-i', '--game-file', type=str, default="z-machine-games-master/jericho-game-suite/enchanter.z3", help='Path to the game file.')
    parser.add_argument('-o', '--output-file', type=str, default='trajectories.txt', help='Output file name for the trajectories.')
    parser.add_argument('-s', '--num-seeds', type=int, default=5, help='Number of seeds for environment initialization.')
    parser.add_argument('-r', '--num-repeats', type=int, default=10, help='Number of repeats for each percentage of steps.')
    parser.add_argument('--num-random-steps', type=int, default=100, help='Number of random steps to take after following the walkthrough.')
    parser.add_argument('--gold', type="store_true", help='Generate golden trajectory only.')
    
    
    args = parser.parse_args()

    generator = TrajectoryGenerator(
        game_file_path=args.game_file_path,
        num_random_steps=args.num_random_steps,
        num_repeats=args.num_repeats,
        num_seeds=args.num_seeds,
        output_file=args.output_file
    )
    generator.run()

if __name__ == "__main__":
    main()
