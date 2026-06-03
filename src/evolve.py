"""
Run the evolutionary training loop for the SlimeVolley agent.
"""

import copy

import numpy as np

from src.evaluate import evaluate_fitness
from src.genome import (
    create_initial_genome,
    mutate_add_connection,
    mutate_add_node,
    mutate_change_weight,
    mutate_toggle_connection,
)
from src.storage import save_genome


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
    min_fitness = min(species_fitness(cluster) for cluster in surviving_species)
    weights = np.asarray([
        species_fitness(cluster) - min_fitness + 1.0
        for cluster in surviving_species
    ], dtype=np.float32)
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
    weights = fitnesses - np.min(fitnesses) + 1e-6

    # If all fitnesses are effectively identical, sample uniformly.
    if np.sum(weights) <= 0:
        probabilities = None
    else:
        probabilities = weights / np.sum(weights)

    selected_index = np.random.choice(len(scored_population), p=probabilities)
    return scored_population[selected_index]


def evolve(
    population_size=100,
    generations=150,
    target_fitness=None,
    episodes_per_genome=3,
    elite_count=10,
    species_count=5,
    best_genome_path="saved/best_genome.json",
    survival_bonus_per_step=0.01,
    weight_mutation_rate=0.8,
    add_node_rate=0.04,
    add_connection_rate=0.05,
    toggle_connection_rate=0.01,
):
    """
    Run evolution and return the best genome and its fitness.

    Args:
        population_size:        #genomes in each generation.
        generations:            Max generations.
        target_fitness:         early stopping condition.
        episodes_per_genome:    #episodes used to score each genome.
        elite_count:            #genomes copied to next generation.
        species_count:          Number of k-medoids species to form.
        best_genome_path:       File path for saving the best genome. None disables saving.
        survival_bonus_per_step: Extra reward for each timestep survived.
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

    for generation in range(generations):
        scored_population = []

        # Run every genome in the environment and record its fitness.
        for genome in population:
            fitness = evaluate_fitness(
                genome,
                episodes=episodes_per_genome,
                survival_bonus_per_step=survival_bonus_per_step,
            )
            scored_population.append({
                "genome": genome,
                "fitness": fitness,
            })

        # Highest fitness comes first.
        scored_population.sort(key=lambda item: item["fitness"], reverse=True)
        generation_best = scored_population[0]
        species = speciate(scored_population, species_count=species_count)
        species.sort(key=species_fitness, reverse=True)

        # Usually keep all species; sometimes remove only the weakest species.
        surviving_species = species
        if len(species) > 1 and np.random.random() < 0.5:
            surviving_species = species[:-1]

        # Keep a copy of the best genome seen across all generations.
        if generation_best["fitness"] > best_fitness:
            best_fitness = generation_best["fitness"]
            best_genome = copy.deepcopy(generation_best["genome"])
            if best_genome_path is not None:
                save_genome(best_genome_path, best_genome, best_fitness)

        print(
            f"Generation {generation}: "
            f"best={generation_best['fitness']}, "
            f"overall_best={best_fitness}"
        )

        # Elites survive unchanged into the next generation.
        next_population = [
            copy.deepcopy(item["genome"])
            for item in scored_population[:elite_count]
        ]

        if target_fitness is not None and best_fitness >= target_fitness:
            break

        # Fill the rest of the next generation with mutated children.
        while len(next_population) < population_size:
            parent_species = select_species(surviving_species)
            parent_a = select_parent(parent_species["members"])
            parent_b = select_parent(parent_species["members"])

            # Use the fitter parent as the base for inherited structure.
            if parent_b["fitness"] > parent_a["fitness"]:
                parent_a, parent_b = parent_b, parent_a

            child = crossover(parent_a["genome"], parent_b["genome"])
            mutate_genome(
                child,
                innovation_tracker,
                weight_mutation_rate=weight_mutation_rate,
                add_node_rate=add_node_rate,
                add_connection_rate=add_connection_rate,
                toggle_connection_rate=toggle_connection_rate,
            )
            next_population.append(child)

        population = next_population

    return best_genome, best_fitness


if __name__ == "__main__":
    #PYTHONPATH=.:./slimevolleygym .venv/bin/python src/evolve.py
    evolve()
