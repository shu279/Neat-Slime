"""
Evaluate a saved genome using true SlimeVolley reward.
"""

import argparse

from src.evaluate import evaluate_game_score, record_episode_mp4
from src.storage import load_genome_payload


DEFAULT_ACTION_MODE = "exclusive"
DEFAULT_HORIZONTAL_MARGIN = 0.0
DEFAULT_ACTION_REPEAT = 1
DEFAULT_OPPONENT_MODE = "baseline"


def main():
    """
    Load a saved genome and print raw win/loss evaluation stats.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="saved/best_genome.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent-mode")
    parser.add_argument("--horizontal-margin", type=float)
    parser.add_argument("--action-repeat", type=int)
    parser.add_argument("--record-mp4")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    payload = load_genome_payload(args.path)
    genome = payload["genome"]
    saved_score = payload.get("fitness")
    metadata = payload.get("metadata", {})
    opponent_mode = (
        args.opponent_mode
        if args.opponent_mode is not None
        else metadata.get("opponent_mode", DEFAULT_OPPONENT_MODE)
    )
    horizontal_margin = (
        args.horizontal_margin
        if args.horizontal_margin is not None
        else metadata.get("horizontal_margin", DEFAULT_HORIZONTAL_MARGIN)
    )
    action_repeat = (
        args.action_repeat
        if args.action_repeat is not None
        else metadata.get("action_repeat", DEFAULT_ACTION_REPEAT)
    )

    if args.record_mp4 is not None:
        reward = record_episode_mp4(
            genome,
            args.record_mp4,
            opponent_mode=opponent_mode,
            horizontal_margin=horizontal_margin,
            action_repeat=action_repeat,
        )
        print(f"Saved video: {args.record_mp4}")
        print(f"Episode raw reward: {reward}")
        return

    stats = evaluate_game_score(
        genome,
        episodes=args.episodes,
        opponent_mode=opponent_mode,
        horizontal_margin=horizontal_margin,
        action_repeat=action_repeat,
        render=args.render,
    )

    score_type = metadata.get("score_type", "unknown_score")
    print(f"Saved score ({score_type}): {saved_score}")
    print(f"Opponent mode: {opponent_mode}")
    print(f"Horizontal margin: {horizontal_margin}")
    print(f"Action repeat: {action_repeat}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Average raw reward: {stats['average_reward']}")
    print(f"Best raw reward: {stats['best_reward']}")
    print(f"Worst raw reward: {stats['worst_reward']}")
    print(f"Average episode length: {stats['average_episode_length']}")
    print(f"Best episode length: {stats['best_episode_length']}")
    print(f"Worst episode length: {stats['worst_episode_length']}")
    print(f"Wins: {stats['wins']}")
    print(f"Draws: {stats['draws']}")
    print(f"Losses: {stats['losses']}")
    print(f"Points for: {stats['points_for']}")
    print(f"Points against: {stats['points_against']}")
    print(f"Point win rate: {stats['point_win_rate']}")


if __name__ == "__main__":
    main()
