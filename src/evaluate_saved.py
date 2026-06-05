"""
Evaluate a saved genome using true SlimeVolley reward.
"""

import argparse

from src.evaluate import evaluate_game_score, record_episode_mp4
from src.storage import load_genome_payload


def main():
    """
    Load a saved genome and print raw win/loss evaluation stats.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="saved/best_genome.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent-mode", default="baseline")
    parser.add_argument("--horizontal-margin", type=float, default=0.05)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--record-mp4")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    payload = load_genome_payload(args.path)
    genome = payload["genome"]
    saved_score = payload.get("fitness")
    metadata = payload.get("metadata", {})

    if args.record_mp4 is not None:
        reward = record_episode_mp4(
            genome,
            args.record_mp4,
            opponent_mode=args.opponent_mode,
            horizontal_margin=args.horizontal_margin,
            action_repeat=args.action_repeat,
        )
        print(f"Saved video: {args.record_mp4}")
        print(f"Episode raw reward: {reward}")
        return

    stats = evaluate_game_score(
        genome,
        episodes=args.episodes,
        opponent_mode=args.opponent_mode,
        horizontal_margin=args.horizontal_margin,
        action_repeat=args.action_repeat,
        render=args.render,
    )

    score_type = metadata.get("score_type", "unknown_score")
    print(f"Saved score ({score_type}): {saved_score}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Average raw reward: {stats['average_reward']}")
    print(f"Best raw reward: {stats['best_reward']}")
    print(f"Worst raw reward: {stats['worst_reward']}")
    print(f"Wins: {stats['wins']}")
    print(f"Draws: {stats['draws']}")
    print(f"Losses: {stats['losses']}")
    print(f"Points for: {stats['points_for']}")
    print(f"Points against: {stats['points_against']}")
    print(f"Point win rate: {stats['point_win_rate']}")


if __name__ == "__main__":
    main()
