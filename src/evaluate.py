"""
Measure genome fitness by running SlimeVolley episodes.
"""

import numpy as np

from src.network import forward, outputs_to_action
import gym
import slimevolleygym  # noqa: F401


def apply_complexity_penalty(raw_reward, genome, gene_penalty=0.001):
    """
    Penalize genomes with more connection genes.

    Args:
        raw_reward: Average episode reward before the size penalty.
        genome: Genome being scored.
        gene_penalty: Penalty strength for each connection gene.
    """

    connection_count = len(genome["connections"])
    penalty_scale = np.sqrt((connection_count * gene_penalty) + 1.0)
    if raw_reward >= 0:
        return raw_reward / penalty_scale
    return raw_reward * penalty_scale


def evaluate_fitness(
    genome,
    episodes=1,
    action_threshold=0.5,
    render=False,
    gene_penalty=0.001,
    survival_bonus_per_step=0.0,
):
    """
    Run the genome in SlimeVolley and return the average total reward.

    Args:
        genome: Genome to evaluate.
        episodes: Number of episodes to average over.
        action_threshold: Output threshold for binary button actions.
        render: Whether to render the environment during evaluation.
        gene_penalty: Penalty strength for each connection gene.
        survival_bonus_per_step: Extra reward for each timestep survived.
    """

    env = gym.make("SlimeVolley-v0")
    try:
        rewards = [
            run_episode(
                env,
                genome,
                action_threshold=action_threshold,
                render=render,
                survival_bonus_per_step=survival_bonus_per_step,
            )
            for _ in range(episodes)
        ]
    finally:
        env.close()

    average_reward = float(np.mean(rewards))
    return apply_complexity_penalty(average_reward, genome, gene_penalty)


def run_episode(
    env,
    genome,
    action_threshold=0.5,
    render=False,
    survival_bonus_per_step=0.0,
):
    """
    Run one SlimeVolley episode and return its total reward.
    """

    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    done = False
    episode_reward = 0.0

    while not done:
        outputs = forward(genome, obs)
        action = outputs_to_action(outputs, threshold=action_threshold)

        step_result = env.step(action)
        if len(step_result) == 5:
            # New Gym API (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, _info = step_result
            done = terminated or truncated
        else:
            # Old Gym API (obs, reward, done, info)
            obs, reward, done, _info = step_result

        episode_reward += reward + survival_bonus_per_step

        if render:
            env.render()

    return episode_reward


def evaluate_game_score(genome, episodes=100, action_threshold=0.5, render=False):
    """
    Evaluate true game score without survival bonus or complexity penalty.
    """

    env = gym.make("SlimeVolley-v0")
    try:
        rewards = np.asarray([
            run_episode(
                env,
                genome,
                action_threshold=action_threshold,
                render=render,
                survival_bonus_per_step=0.0,
            )
            for _ in range(episodes)
        ], dtype=np.float32)
    finally:
        env.close()

    return {
        "average_reward": float(np.mean(rewards)),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "wins": int(np.sum(rewards > 0)),
        "draws": int(np.sum(rewards == 0)),
        "losses": int(np.sum(rewards < 0)),
        "episodes": int(episodes),
    }
