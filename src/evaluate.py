"""
Measure genome fitness by running SlimeVolley episodes.
"""

import numpy as np

from src.network import forward, outputs_to_action
import gym
import slimevolleygym  # noqa: F401


def apply_complexity_penalty(raw_reward, genome, gene_penalty=0.0):
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
    action_mode="exclusive",
    horizontal_margin=0.05,
    action_repeat=3,
    opponent_mode="baseline",
    render=False,
    gene_penalty=0.001,
    survival_bonus_per_step=0.0,
    net_camping_penalty=0.0,
    ball_tracking_penalty=0.0,
    action_shaping_scale=0.0,
    ball_contact_reward=0.0,
    ball_approach_reward=0.0,
    missed_ball_penalty=0.0,
):
    """
    Run the genome in SlimeVolley and return the average total reward.

    Args:
        genome: Genome to evaluate.
        episodes: Number of episodes to average over.
        action_threshold: Output threshold for binary button actions.
        action_mode: How network outputs are converted to SlimeVolley buttons.
        horizontal_margin: Minimum output gap needed to move left/right.
        action_repeat: Number of environment steps to reuse each selected action.
        opponent_mode: Opponent policy: baseline, random, or mixed.
        render: Whether to render the environment during evaluation.
        gene_penalty: Penalty strength for each connection gene.
        survival_bonus_per_step: Extra reward for each timestep survived.
        net_camping_penalty: Penalty strength for staying near net while ball is behind.
        ball_tracking_penalty: Penalty strength for horizontal distance to own-side ball.
        action_shaping_scale: Reward/penalty strength for directional action choices.
        ball_contact_reward: Reward for likely touching/hitting the ball.
        ball_approach_reward: Reward for moving closer to ball on the agent's side.
        missed_ball_penalty: Penalty when opponent scores and ball was behind agent.
    """

    env = gym.make("SlimeVolley-v0")
    try:
        rewards = [
            run_episode(
                env,
                genome,
                action_threshold=action_threshold,
                action_mode=action_mode,
                horizontal_margin=horizontal_margin,
                action_repeat=action_repeat,
                opponent_mode=opponent_mode,
                render=render,
                survival_bonus_per_step=survival_bonus_per_step,
                net_camping_penalty=net_camping_penalty,
                ball_tracking_penalty=ball_tracking_penalty,
                action_shaping_scale=action_shaping_scale,
                ball_contact_reward=ball_contact_reward,
                ball_approach_reward=ball_approach_reward,
                missed_ball_penalty=missed_ball_penalty,
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
    action_mode="exclusive",
    horizontal_margin=0.05,
    action_repeat=3,
    opponent_mode="baseline",
    render=False,
    survival_bonus_per_step=0.0,
    net_camping_penalty=0.0,
    ball_tracking_penalty=0.0,
    action_shaping_scale=0.0,
    ball_contact_reward=0.0,
    ball_approach_reward=0.0,
    missed_ball_penalty=0.0,
    return_point_stats=False,
):
    """
    Run one SlimeVolley episode and return its total reward.
    """

    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    done = False
    episode_reward = 0.0
    points_for = 0
    points_against = 0
    episode_length = 0
    episode_opponent_mode = opponent_mode
    if opponent_mode == "mixed":
        episode_opponent_mode = "baseline" if np.random.random() < 0.5 else "random"
    repeated_action = np.zeros(3, dtype=np.int8)
    repeat_countdown = 0

    while not done:
        previous_obs = obs
        if repeat_countdown <= 0:
            outputs = forward(genome, obs)
            repeated_action = outputs_to_action(
                outputs,
                threshold=action_threshold,
                action_mode=action_mode,
                horizontal_margin=horizontal_margin,
            )
            repeat_countdown = action_repeat
        action = repeated_action
        repeat_countdown -= 1

        other_action = None
        if episode_opponent_mode == "random":
            other_action = env.action_space.sample()
        elif episode_opponent_mode != "baseline":
            raise ValueError(f"Unknown opponent_mode: {opponent_mode}")

        step_result = env.step(action, other_action)
        episode_length += 1
        if len(step_result) == 5:
            # New Gym API (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, _info = step_result
            done = terminated or truncated
        else:
            # Old Gym API (obs, reward, done, info)
            obs, reward, done, _info = step_result

        if reward > 0:
            points_for += 1
        elif reward < 0:
            points_against += 1

        shaped_reward = reward + survival_bonus_per_step
        shaped_reward += get_position_shaping_reward(
            obs,
            action,
            net_camping_penalty=net_camping_penalty,
            ball_tracking_penalty=ball_tracking_penalty,
            action_shaping_scale=action_shaping_scale,
        )
        shaped_reward += get_ball_interaction_shaping_reward(
            previous_obs,
            obs,
            reward,
            ball_contact_reward=ball_contact_reward,
            ball_approach_reward=ball_approach_reward,
            missed_ball_penalty=missed_ball_penalty,
        )
        episode_reward += shaped_reward

        if render:
            env.render()

    if return_point_stats:
        return {
            "reward": episode_reward,
            "points_for": points_for,
            "points_against": points_against,
            "episode_length": episode_length,
        }

    return episode_reward


def get_position_shaping_reward(
    obs,
    action,
    net_camping_penalty=0.0,
    ball_tracking_penalty=0.0,
    action_shaping_scale=0.0,
    net_x_threshold=0.65,
    ball_behind_margin=0.15,
):
    """
    Return a small training-only reward adjustment for horizontal positioning.
    """

    agent_x = obs[0]
    ball_x = obs[4]
    shaping_reward = 0.0

    # Agent is on the right side in normalized coordinates. Small x is near the net.
    if net_camping_penalty > 0 and agent_x < net_x_threshold and ball_x > agent_x:
        shaping_reward -= net_camping_penalty * (net_x_threshold - agent_x)

    # Only track horizontally when the ball is on the agent's side of the net.
    if ball_tracking_penalty > 0 and ball_x > 0:
        shaping_reward -= ball_tracking_penalty * abs(agent_x - ball_x)

    if action_shaping_scale > 0:
        ball_is_behind = ball_x > agent_x + ball_behind_margin
        agent_near_net = agent_x < net_x_threshold

        # Discourage pushing into the net once already camped there.
        if agent_near_net and action[0] > 0:
            shaping_reward -= action_shaping_scale

        # Encourage backing up when the ball is clearly behind the player.
        if ball_is_behind and action[1] > 0:
            shaping_reward += action_shaping_scale
        elif ball_is_behind and action[0] > 0:
            shaping_reward -= action_shaping_scale

    return shaping_reward


def get_ball_interaction_shaping_reward(
    previous_obs,
    obs,
    raw_reward,
    ball_contact_reward=0.0,
    ball_approach_reward=0.0,
    missed_ball_penalty=0.0,
    contact_distance=0.25,
    velocity_change_threshold=0.05,
    ball_behind_margin=0.12,
):
    """
    Return training-only reward for ball interaction.

    The environment observation is relative to the right-side agent and scaled
    by 10. Positive ball_x means the ball is on the trained agent's side.
    """

    previous_agent_x = previous_obs[0]
    previous_ball_x = previous_obs[4]
    previous_ball_y = previous_obs[5]
    previous_ball_vx = previous_obs[6]
    previous_ball_vy = previous_obs[7]

    agent_x = obs[0]
    agent_y = obs[1]
    ball_x = obs[4]
    ball_y = obs[5]
    ball_vx = obs[6]
    ball_vy = obs[7]

    shaping_reward = 0.0

    if ball_contact_reward > 0:
        ball_distance = np.hypot(ball_x - agent_x, ball_y - agent_y)
        velocity_change = np.hypot(
            ball_vx - previous_ball_vx,
            ball_vy - previous_ball_vy,
        )
        if ball_distance <= contact_distance and velocity_change >= velocity_change_threshold:
            shaping_reward += ball_contact_reward

    if ball_approach_reward > 0 and previous_ball_x > 0 and ball_x > 0:
        previous_distance = abs(previous_agent_x - previous_ball_x)
        current_distance = abs(agent_x - ball_x)
        distance_improvement = previous_distance - current_distance
        if distance_improvement > 0:
            shaping_reward += ball_approach_reward * min(distance_improvement, 0.1)

    ball_was_behind = previous_ball_x > previous_agent_x + ball_behind_margin
    if missed_ball_penalty > 0 and raw_reward < 0 and ball_was_behind:
        shaping_reward -= missed_ball_penalty

    return shaping_reward


def record_episode_mp4(
    genome,
    path,
    action_threshold=0.5,
    action_mode="exclusive",
    horizontal_margin=0.05,
    action_repeat=3,
    opponent_mode="baseline",
    fps=30,
    scale=4,
):
    """
    Record one SlimeVolley episode to MP4 without using the pyglet live renderer.
    """

    import cv2
    from slimevolleygym.slimevolley import setPixelObsMode

    setPixelObsMode()
    env = gym.make("SlimeVolley-v0")
    writer = None
    episode_reward = 0.0

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        done = False
        repeated_action = np.zeros(3, dtype=np.int8)
        repeat_countdown = 0

        while not done:
            frame = env.render(mode="state")
            if scale != 1:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] * scale, frame.shape[0] * scale),
                    interpolation=cv2.INTER_NEAREST,
                )

            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if repeat_countdown <= 0:
                outputs = forward(genome, obs)
                repeated_action = outputs_to_action(
                    outputs,
                    threshold=action_threshold,
                    action_mode=action_mode,
                    horizontal_margin=horizontal_margin,
                )
                repeat_countdown = action_repeat
            action = repeated_action
            repeat_countdown -= 1

            other_action = None
            if opponent_mode == "random":
                other_action = env.action_space.sample()
            elif opponent_mode != "baseline":
                raise ValueError(f"Unknown opponent_mode: {opponent_mode}")

            step_result = env.step(action, other_action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _info = step_result

            episode_reward += reward
    finally:
        if writer is not None:
            writer.release()
        env.close()

    return episode_reward


def evaluate_game_score(
    genome,
    episodes=100,
    action_threshold=0.5,
    action_mode="exclusive",
    horizontal_margin=0.05,
    action_repeat=3,
    opponent_mode="baseline",
    render=False,
):
    """
    Evaluate true game score without survival bonus or complexity penalty.
    """

    env = gym.make("SlimeVolley-v0")
    try:
        episode_stats = [
            run_episode(
                env,
                genome,
                action_threshold=action_threshold,
                action_mode=action_mode,
                horizontal_margin=horizontal_margin,
                action_repeat=action_repeat,
                opponent_mode=opponent_mode,
                render=render,
                survival_bonus_per_step=0.0,
                return_point_stats=True,
            )
            for _ in range(episodes)
        ]
    finally:
        env.close()

    rewards = np.asarray([
        episode["reward"]
        for episode in episode_stats
    ], dtype=np.float32)
    points_for = sum(episode["points_for"] for episode in episode_stats)
    points_against = sum(episode["points_against"] for episode in episode_stats)
    episode_lengths = np.asarray([
        episode["episode_length"]
        for episode in episode_stats
    ], dtype=np.float32)
    point_count = points_for + points_against

    return {
        "average_reward": float(np.mean(rewards)),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "average_episode_length": float(np.mean(episode_lengths)),
        "best_episode_length": int(np.max(episode_lengths)),
        "worst_episode_length": int(np.min(episode_lengths)),
        "wins": int(np.sum(rewards > 0)),
        "draws": int(np.sum(rewards == 0)),
        "losses": int(np.sum(rewards < 0)),
        "points_for": int(points_for),
        "points_against": int(points_against),
        "point_win_rate": 0.0 if point_count == 0 else points_for / point_count,
        "episodes": int(episodes),
    }
