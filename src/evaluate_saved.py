"""
Evaluate a saved genome using true SlimeVolley reward.
"""

import argparse

from src.evaluate import evaluate_game_score
from src.storage import load_genome


def main():
    """
    Load a saved genome and print raw win/loss evaluation stats.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="saved/best_genome.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    genome, saved_fitness = load_genome(args.path)
    stats = evaluate_game_score(
        genome,
        episodes=args.episodes,
        render=args.render,
    )

    print(f"Saved training fitness: {saved_fitness}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Average raw reward: {stats['average_reward']}")
    print(f"Best raw reward: {stats['best_reward']}")
    print(f"Worst raw reward: {stats['worst_reward']}")
    print(f"Wins: {stats['wins']}")
    print(f"Draws: {stats['draws']}")
    print(f"Losses: {stats['losses']}")


if __name__ == "__main__":
    main()
