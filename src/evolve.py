"""
Run the evolutionary training loop for the SlimeVolley agent.
"""

import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

from src.evaluate import evaluate_fitness, evaluate_game_score
from src.genome import (
    create_initial_genome,
    mutate_add_connection,
    mutate_add_node,
    mutate_change_weight,
    mutate_toggle_connection,
)
from src.storage import save_genome


def _evaluate_genome_worker(args):
    """
    Evaluate one genome in a worker process.
    """

    (
        genome,
        episodes_per_genome,
        survival_bonus_per_step,
        action_mode,
        horizontal_margin,
        action_repeat,
        opponent_mode,
        net_camping_penalty,
        ball_tracking_penalty,
        action_shaping_scale,
        ball_contact_reward,
        ball_approach_reward,
        missed_ball_penalty,
    ) = args
    try:
        return evaluate_fitness(
            genome,
            episodes=episodes_per_genome,
            survival_bonus_per_step=survival_bonus_per_step,
            action_mode=action_mode,
            horizontal_margin=horizontal_margin,
            action_repeat=action_repeat,
            opponent_mode=opponent_mode,
            net_camping_penalty=net_camping_penalty,
            ball_tracking_penalty=ball_tracking_penalty,
            action_shaping_scale=action_shaping_scale,
            ball_contact_reward=ball_contact_reward,
            ball_approach_reward=ball_approach_reward,
            missed_ball_penalty=missed_ball_penalty,
        )
    except ValueError as error:
        if "cyclic" not in str(error) and "recurrent" not in str(error):
            raise
        return float("-inf")


def create_population(size, n_in=12, n_out=3):
    """
    Create the first generation of minimal genomes.
    """

    # Start with many independent minimal genomes.
    return [create_initial_genome(n_in=n_in, n_out=n_out) for _ in range(size)]


def create_innovation_tracker(population):
    """
    Track shared innovation numbers across the whole population.
    """

    connections = {}
    next_innov = 1

    for genome in population:
        for connection in genome["connections"]:
            key = (connection["from"], connection["to"])
            connections[key] = connection["innov"]
            next_innov = max(next_innov, connection["innov"] + 1)

    return {
        "connections": connections,
        "next_innov": next_innov,
    }


def mutate_genome(
    genome,
    innovation_tracker,
    weight_mutation_rate=0.8,
    add_node_rate=0.03,
    add_connection_rate=0.08,
    toggle_connection_rate=0.02,
):
    """
    Apply the four mutation types: weight, add-node, add-connection, toggle.
    """

    mutate_change_weight(genome, mutation_rate=weight_mutation_rate)

    if np.random.random() < add_node_rate:
        mutate_add_node(genome, innovation_tracker=innovation_tracker)

    if np.random.random() < add_connection_rate:
        mutate_add_connection(genome, innovation_tracker=innovation_tracker)

    if np.random.random() < toggle_connection_rate:
        mutate_toggle_connection(genome)


def genome_size_summary(genome):
    """
    Return simple topology counts for progress logging.
    """

    hidden_count = sum(
        1
        for node in genome["nodes"].values()
        if node["type"] not in {"input", "bias", "output"}
    )
    enabled_connection_count = sum(
        1
        for connection in genome["connections"]
        if connection["enabled"]
    )
    return {
        "hidden": hidden_count,
        "enabled_connections": enabled_connection_count,
        "connections": len(genome["connections"]),
    }


def crossover(parent_a, parent_b):
    """
    Create a child by randomly choosing matching genes from either parent.
    """

    # Copy one parent as the base so node/input/output metadata stays valid.
    child = copy.deepcopy(parent_a)

    # Innovation numbers let us match the same connection in both parents.
    parent_b_by_innovation = {
        connection["innov"]: connection
        for connection in parent_b["connections"]
    }

    for child_connection in child["connections"]:
        matching_connection = parent_b_by_innovation.get(child_connection["innov"])
        if matching_connection is not None and np.random.random() < 0.5:
            # Randomly inherit this connection's values from the other parent.
            child_connection["weight"] = matching_connection["weight"]
            child_connection["enabled"] = matching_connection["enabled"]

    child["next_node_id"] = max(parent_a["next_node_id"], parent_b["next_node_id"])
    child["next_innov"] = max(parent_a["next_innov"], parent_b["next_innov"])
    return child


def genome_distance(genome_a, genome_b):
    """
    Distance used by k-medoids speciation.

    Matching genes compare weights. Missing genes count as structural distance.
    """

    connections_a = {
        connection["innov"]: connection
        for connection in genome_a["connections"]
    }
    connections_b = {
        connection["innov"]: connection
        for connection in genome_b["connections"]
    }
    innovations = set(connections_a) | set(connections_b)
    matching = set(connections_a) & set(connections_b)

    missing_gene_count = len(innovations) - len(matching)
    normalizer = max(len(connections_a), len(connections_b), 1)
    structural_distance = missing_gene_count / normalizer

    if matching:
        weight_diffs = np.asarray([
            abs(connections_a[innov]["weight"] - connections_b[innov]["weight"])
            for innov in matching
        ], dtype=np.float32)
        weight_distance = float(np.mean(weight_diffs))
    else:
        weight_distance = 0.0

    node_distance = abs(len(genome_a["nodes"]) - len(genome_b["nodes"])) / max(
        len(genome_a["nodes"]),
        len(genome_b["nodes"]),
        1,
    )

    return structural_distance + (0.4 * weight_distance) + (0.2 * node_distance)


def assign_to_medoids(scored_population, medoids):
    """
    Assign each scored genome to its closest medoid genome.
    """

    species = [{"medoid": medoid, "members": []} for medoid in medoids]

    for item in scored_population:
        closest_species = min(
            species,
            key=lambda cluster: genome_distance(item["genome"], cluster["medoid"]),
        )
        closest_species["members"].append(item)

    return [cluster for cluster in species if cluster["members"]]


def update_medoids(species):
    """
    Replace each species medoid with its most central member.
    """

    for cluster in species:
        # The medoid is the real genome with the smallest total distance to others.
        cluster["medoid"] = min(
            cluster["members"],
            key=lambda candidate: sum(
                genome_distance(candidate["genome"], other["genome"])
                for other in cluster["members"]
            ),
        )["genome"]


def speciate(scored_population, species_count=5, iterations=5):
    """
    Cluster genomes into species using a small k-medoids loop.
    """

    if len(scored_population) <= species_count:
        return [
            {"medoid": item["genome"], "members": [item]}
            for item in scored_population
        ]

    medoid_indices = np.random.choice(
        len(scored_population),
        size=species_count,
        replace=False,
    )
    medoids = [scored_population[index]["genome"] for index in medoid_indices]

    for _ in range(iterations):
        species = assign_to_medoids(scored_population, medoids)
        update_medoids(species)
        medoids = [cluster["medoid"] for cluster in species]

    return assign_to_medoids(scored_population, medoids)


def species_fitness(species):
    """
    Return the best member fitness inside one species.
    """

    return max(item["fitness"] for item in species["members"])


def select_species(surviving_species):
    """
    Choose a surviving species, weighted by its best fitness.
    """

    # Better species are more likely to produce the next child.
    fitnesses = np.asarray([
        species_fitness(cluster)
        for cluster in surviving_species
    ], dtype=np.float32)

    finite_mask = np.isfinite(fitnesses)
    if not np.any(finite_mask):
        probabilities = None
    else:
        finite_fitnesses = fitnesses[finite_mask]
        weights = np.zeros_like(fitnesses, dtype=np.float32)
        weights[finite_mask] = finite_fitnesses - np.min(finite_fitnesses) + 1e-6
        probabilities = weights / np.sum(weights)

    selected_index = np.random.choice(len(surviving_species), p=probabilities)
    return surviving_species[selected_index]


def select_parent(scored_population):
    """
    Select one parent with probability proportional to shifted fitness.
    """

    fitnesses = np.asarray([
        item["fitness"]
        for item in scored_population
    ], dtype=np.float32)

    finite_mask = np.isfinite(fitnesses)
    if not np.any(finite_mask):
        probabilities = None
    else:
        finite_fitnesses = fitnesses[finite_mask]
        weights = np.zeros_like(fitnesses, dtype=np.float32)
        weights[finite_mask] = finite_fitnesses - np.min(finite_fitnesses) + 1e-6
        probabilities = weights / np.sum(weights)

    selected_index = np.random.choice(len(scored_population), p=probabilities)
    return scored_population[selected_index]


def allocate_offspring_counts(species, child_count):
    """
    Decide how many children each species should produce.

    Each species gets one child first when possible, then extra children are
    sampled using shifted species fitness. This protects new architectures from
    being removed before their weights have time to improve.
    """

    if child_count <= 0:
        return [0 for _ in species]

    counts = np.zeros(len(species), dtype=np.int32)
    guaranteed_count = min(child_count, len(species))
    counts[:guaranteed_count] = 1
    remaining_count = child_count - guaranteed_count

    if remaining_count <= 0:
        return counts.tolist()

    fitnesses = np.asarray([
        species_fitness(cluster)
        for cluster in species
    ], dtype=np.float64)

    finite_mask = np.isfinite(fitnesses)
    if not np.any(finite_mask):
        probabilities = np.full(len(species), 1.0 / len(species), dtype=np.float64)
    else:
        finite_fitnesses = fitnesses[finite_mask]
        weights = np.zeros_like(fitnesses, dtype=np.float64)
        weights[finite_mask] = finite_fitnesses - np.min(finite_fitnesses) + 1e-6
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            probabilities = np.full(len(species), 1.0 / len(species), dtype=np.float64)
        else:
            probabilities = weights / weight_sum

    # np.random.multinomial is strict about probabilities summing to exactly 1.
    probabilities = probabilities / np.sum(probabilities)
    probabilities[-1] = 1.0 - np.sum(probabilities[:-1])

    counts += np.random.multinomial(remaining_count, probabilities)
    return counts.tolist()


def make_child_from_species(parent_species, innovation_tracker, mutation_rates):
    """
    Create one child using two parents from the same species.
    """

    parent_a = select_parent(parent_species["members"])
    parent_b = select_parent(parent_species["members"])

    # Use the fitter parent as the base for inherited structure.
    if parent_b["fitness"] > parent_a["fitness"]:
        parent_a, parent_b = parent_b, parent_a

    child = crossover(parent_a["genome"], parent_b["genome"])
    mutate_genome(
        child,
        innovation_tracker,
        weight_mutation_rate=mutation_rates["weight"],
        add_node_rate=mutation_rates["add_node"],
        add_connection_rate=mutation_rates["add_connection"],
        toggle_connection_rate=mutation_rates["toggle_connection"],
    )
    return child


def evaluate_population(
    population,
    episodes_per_genome,
    survival_bonus_per_step,
    action_mode,
    horizontal_margin,
    action_repeat,
    opponent_mode,
    net_camping_penalty,
    ball_tracking_penalty,
    action_shaping_scale,
    ball_contact_reward,
    ball_approach_reward,
    missed_ball_penalty,
    num_workers=1,
):
    """
    Evaluate all genomes, optionally in parallel across worker processes.
    """

    worker_args = [
        (
            genome,
            episodes_per_genome,
            survival_bonus_per_step,
            action_mode,
            horizontal_margin,
            action_repeat,
            opponent_mode,
            net_camping_penalty,
            ball_tracking_penalty,
            action_shaping_scale,
            ball_contact_reward,
            ball_approach_reward,
            missed_ball_penalty,
        )
        for genome in population
    ]

    if num_workers is None or num_workers <= 1:
        fitnesses = [_evaluate_genome_worker(args) for args in worker_args]
    else:
        # Use fork on macOS/Linux to avoid Python -m / script re-import issues.
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=context,
        ) as executor:
            fitnesses = list(executor.map(_evaluate_genome_worker, worker_args))

    return [
        {
            "genome": genome,
            "fitness": fitness,
        }
        for genome, fitness in zip(population, fitnesses)
    ]


def evolve(
    population_size=300,
    generations=300,
    target_fitness=None,
    episodes_per_genome=8,
    elite_count=22,
    species_count=10,
    best_genome_path="saved/best_genome.json",
    survival_bonus_per_step=0.00002,
    action_mode="exclusive",
    horizontal_margin=0.0,
    action_repeat=1,
    opponent_mode="baseline",
    net_camping_penalty=0.004,
    ball_tracking_penalty=0.001,
    action_shaping_scale=0.001,
    ball_contact_reward=0.02,
    ball_approach_reward=0.005,
    missed_ball_penalty=0.1,
    raw_eval_interval=5,
    raw_eval_episodes=10,
    save_best_by_raw_score=False,
    raw_save_episodes=20,
    num_workers=26,
    stagnation_generations=50,
    stagnation_reset_fraction=0.2,
    weight_mutation_rate=0.8,
    add_node_rate=0.1,
    add_connection_rate=0.1,
    toggle_connection_rate=0.02,
):
    """
    Run evolution and return the best genome and its selected score.

    Training fitness is used for evolution and saving by default. If
    save_best_by_raw_score is True, the saved genome is chosen by raw baseline
    reward instead.

    Train
    PYTHONPATH=.:./slimevolleygym .venv/bin/python src/evolve.py

    Check video
    PYTHONPATH=.:./slimevolleygym .venv/bin/python -m src.evaluate_saved --path saved/best_genome.json --record-mp4 saved/best_gameplay.mp4

    Check best genome
    PYTHONPATH=.:./slimevolleygym .venv/bin/python -m src.evaluate_saved --episodes 100

    Start roundtrip
    cd /Users/shusato/Desktop/neat-slime
    bash scripts/gce_roundtrip.sh

    Args:
        population_size:        #genomes in each generation.
        generations:            Max generations.
        target_fitness:         early stopping condition.
        episodes_per_genome:    #episodes used to score each genome.
        elite_count:            #genomes copied to next generation.
        species_count:          Number of k-medoids species to form.
        best_genome_path:       File path for saving the best genome. None disables saving.
        survival_bonus_per_step: Extra reward for each timestep survived.
        action_mode:            How outputs are converted to environment actions.
        horizontal_margin:      Minimum output gap needed to move left/right.
        action_repeat:          Number of environment steps to reuse each action.
        opponent_mode:          Training opponent: baseline, random, or mixed.
        net_camping_penalty:    Penalty for staying near net when ball is behind.
        ball_tracking_penalty:  Penalty for being horizontally far from own-side ball.
        action_shaping_scale:   Reward/penalty for directional action choices.
        ball_contact_reward:    Reward for likely touching/hitting the ball.
        ball_approach_reward:   Reward for moving closer to own-side ball.
        missed_ball_penalty:    Penalty when opponent scores from behind agent.
        raw_eval_interval:      Generations between true baseline score checks. None disables.
        raw_eval_episodes:      Episodes used for each true baseline score check.
        save_best_by_raw_score: If True, save by raw score instead of training fitness.
        raw_save_episodes:      Episodes for confirming raw baseline score before saving.
        num_workers:            Worker processes for parallel fitness evaluation.
        stagnation_generations: Generations without improvement before random reset.
        stagnation_reset_fraction: Fraction of worst genomes replaced after stagnation.
        weight_mutation_rate:   Chance that each connection weight mutates.
        add_node_rate:          Chance each child gets an inserted node.
        add_connection_rate:    Chance each child gets a new connection.
        toggle_connection_rate: Chance each child disables/enables one connection.
    """
     
    # Create the first generation. Each genome has 12 inputs, 3 outputs, no hidden nodes.
    population = create_population(population_size)
    innovation_tracker = create_innovation_tracker(population)
    best_genome = None
    best_fitness = float("-inf")
    best_saved_genome = None
    best_saved_raw_score = float("-inf")
    generations_since_improvement = 0

    for generation in range(generations):
        scored_population = evaluate_population(
            population,
            episodes_per_genome=episodes_per_genome,
            survival_bonus_per_step=survival_bonus_per_step,
            action_mode=action_mode,
            horizontal_margin=horizontal_margin,
            action_repeat=action_repeat,
            opponent_mode=opponent_mode,
            net_camping_penalty=net_camping_penalty,
            ball_tracking_penalty=ball_tracking_penalty,
            action_shaping_scale=action_shaping_scale,
            ball_contact_reward=ball_contact_reward,
            ball_approach_reward=ball_approach_reward,
            missed_ball_penalty=missed_ball_penalty,
            num_workers=num_workers,
        )

        # Highest fitness comes first.
        scored_population.sort(key=lambda item: item["fitness"], reverse=True)
        generation_best = scored_population[0]
        species = speciate(scored_population, species_count=species_count)
        species.sort(key=species_fitness, reverse=True)

        # Usually keep all species; sometimes remove only the weakest species.
        surviving_species = species
        if len(species) > 1 and np.random.random() < 0.8:
            surviving_species = species[:-1]

        raw_stats = None

        # Keep a copy of the best training-fitness genome seen across all generations.
        if generation_best["fitness"] > best_fitness:
            best_fitness = generation_best["fitness"]
            best_genome = copy.deepcopy(generation_best["genome"])
            generations_since_improvement = 0

            if save_best_by_raw_score and best_genome_path is not None:
                raw_stats = evaluate_game_score(
                    best_genome,
                    episodes=raw_save_episodes,
                    action_mode=action_mode,
                    horizontal_margin=horizontal_margin,
                    action_repeat=action_repeat,
                    opponent_mode="baseline",
                )
                if raw_stats["average_reward"] > best_saved_raw_score:
                    best_saved_raw_score = raw_stats["average_reward"]
                    best_saved_genome = copy.deepcopy(best_genome)
                    save_genome(
                        best_genome_path,
                        best_saved_genome,
                        best_saved_raw_score,
                        metadata={
                            "score_type": "raw_baseline_average_reward",
                            "raw_eval_episodes": raw_save_episodes,
                            "training_fitness": best_fitness,
                            "generation": generation,
                            "action_mode": action_mode,
                            "horizontal_margin": horizontal_margin,
                            "action_repeat": action_repeat,
                            "opponent_mode": "baseline",
                        },
                    )
            elif best_genome_path is not None:
                save_genome(
                    best_genome_path,
                    best_genome,
                    best_fitness,
                    metadata={
                        "score_type": "training_fitness",
                        "generation": generation,
                        "action_mode": action_mode,
                        "horizontal_margin": horizontal_margin,
                        "action_repeat": action_repeat,
                        "opponent_mode": "baseline",
                    },
                )
        else:
            generations_since_improvement += 1

        generation_best_size = genome_size_summary(generation_best["genome"])
        overall_best_size = genome_size_summary(best_genome)
        log_message = (
            f"Generation {generation}: "
            f"best={generation_best['fitness']}, "
            f"overall_best={best_fitness}, "
            f"hidden={generation_best_size['hidden']}"
            f"/{overall_best_size['hidden']}, "
            f"connections={generation_best_size['enabled_connections']}"
            f"/{generation_best_size['connections']}"
        )
        if raw_eval_interval is not None and generation % raw_eval_interval == 0:
            if raw_stats is None:
                raw_stats = evaluate_game_score(
                    generation_best["genome"],
                    episodes=raw_eval_episodes,
                    action_mode=action_mode,
                    horizontal_margin=horizontal_margin,
                    action_repeat=action_repeat,
                    opponent_mode="baseline",
                )
            log_message += (
                f", raw_avg={raw_stats['average_reward']:.2f}"
                f", w/d/l={raw_stats['wins']}/{raw_stats['draws']}/"
                f"{raw_stats['losses']}"
                f", avg_len={raw_stats['average_episode_length']:.0f}"
                f", raw_points={raw_stats['points_for']}/"
                f"{raw_stats['points_for'] + raw_stats['points_against']}"
            )
        if save_best_by_raw_score:
            log_message += f", saved_raw_best={best_saved_raw_score}"
        print(log_message)

        # Keep one strong representative from each surviving species when possible.
        next_population = []
        copied_genome_ids = set()
        for cluster in surviving_species:
            if len(next_population) >= elite_count:
                break
            species_best = max(cluster["members"], key=lambda item: item["fitness"])
            next_population.append(copy.deepcopy(species_best["genome"]))
            copied_genome_ids.add(id(species_best["genome"]))

        # Fill any remaining elite slots with the global best genomes.
        for item in scored_population:
            if len(next_population) >= elite_count:
                break
            if id(item["genome"]) in copied_genome_ids:
                continue
            next_population.append(copy.deepcopy(item["genome"]))
            copied_genome_ids.add(id(item["genome"]))

        if target_fitness is not None and best_fitness >= target_fitness:
            break

        mutation_rates = {
            "weight": weight_mutation_rate,
            "add_node": add_node_rate,
            "add_connection": add_connection_rate,
            "toggle_connection": toggle_connection_rate,
        }

        # Give each surviving species its own offspring budget.
        child_counts = allocate_offspring_counts(
            surviving_species,
            population_size - len(next_population),
        )
        for parent_species, child_count in zip(surviving_species, child_counts):
            for _ in range(child_count):
                if len(next_population) >= population_size:
                    break
                next_population.append(
                    make_child_from_species(
                        parent_species,
                        innovation_tracker,
                        mutation_rates,
                    )
                )

        while len(next_population) < population_size:
            parent_species = select_species(surviving_species)
            next_population.append(
                make_child_from_species(
                    parent_species,
                    innovation_tracker,
                    mutation_rates,
                )
            )

        if (
            stagnation_generations is not None
            and generations_since_improvement >= stagnation_generations
        ):
            reset_count = max(1, int(population_size * stagnation_reset_fraction))
            reset_count = min(reset_count, population_size - elite_count)
            next_population[-reset_count:] = create_population(reset_count)
            innovation_tracker = create_innovation_tracker(next_population)
            generations_since_improvement = 0
            print(f"Stagnation reset: replaced {reset_count} worst genomes")

        population = next_population

    if save_best_by_raw_score and best_saved_genome is not None:
        return best_saved_genome, best_saved_raw_score
    return best_genome, best_fitness


if __name__ == "__main__":
    #PYTHONPATH=.:./slimevolleygym .venv/bin/python src/evolve.py
    evolve()
