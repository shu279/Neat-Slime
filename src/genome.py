"""
Create and store genome structures for the NEAT-style agent.
"""

import numpy as np


NODE_TYPES = {
    "input",
    "output",
    "bias",
    "add",
    "mult",
    "activation",
    "custom",
}

ACTIVATION_TYPES = {"sigmoid", "tanh", "relu"}
CUSTOM_TYPES = {"gaussian", "sin", "abs", "square"}


def create_nodes(n_in, n_out):
    """
    Create the input, bias, and output nodes for one genome.
    """

    nodes = {}
    input_ids = []

    # Input nodes
    for i in range(n_in):
        nodes[i] = {"type": "input"}
        input_ids.append(i)

    # Bias node
    bias_id = n_in
    nodes[bias_id] = {"type": "bias"}

    # Output nodes
    output_ids = []
    for i in range(n_out):
        node_id = n_in + 1 + i
        nodes[node_id] = {"type": "output", "activation": "sigmoid"}
        output_ids.append(node_id)

    return nodes, input_ids, bias_id, output_ids


def create_initial_connections(input_ids, bias_id, output_ids):
    """
    Create direct input-to-output and bias-to-output connection genes.
    """

    connections = []
    innov = 1

    # Input -> output edges
    for input_id in input_ids:
        for output_id in output_ids:
            connections.append({
                "from": input_id,
                "to": output_id,
                "weight": float(np.random.uniform(-1.0, 1.0)),
                "enabled": True,
                "innov": innov
            })
            innov += 1

    # Bias -> output edges
    for output_id in output_ids:
        connections.append({
            "from": bias_id,
            "to": output_id,
            "weight": float(np.random.uniform(-1.0, 1.0)),
            "enabled": True,
            "innov": innov
        })
        innov += 1

    return connections


def create_initial_genome(n_in=12, n_out=3):
    """
    Create the minimal starting genome for SlimeVolley state observations.

    The initial network has:
    - 12 input nodes for the observation vector
    - 1 bias node
    - 3 output nodes for the MultiBinary action
    - direct input/bias connections to each output
    """

    nodes, input_ids, bias_id, output_ids = create_nodes(n_in, n_out)
    connections = create_initial_connections(input_ids, bias_id, output_ids)

    return {
        "nodes": nodes,
        "connections": connections,
        "input_ids": input_ids,
        "bias_id": bias_id,
        "output_ids": output_ids,
        "next_node_id": max(nodes) + 1,
        "next_innov": len(connections) + 1,
    }


def connection_exists(genome, from_node, to_node):
    """
    Return True if the genome already has a connection between two nodes.
    """

    return any(
        connection["from"] == from_node and connection["to"] == to_node
        for connection in genome["connections"]
    )


def has_path(genome, start_node, target_node):
    """
    Return True if enabled connections already link start_node to target_node.
    """

    nodes_to_visit = [start_node]
    visited = set()

    while nodes_to_visit:
        node_id = nodes_to_visit.pop()
        if node_id == target_node:
            return True
        if node_id in visited:
            continue

        visited.add(node_id)
        for connection in genome["connections"]:
            if connection["enabled"] and connection["from"] == node_id:
                nodes_to_visit.append(connection["to"])

    return False


def creates_cycle(genome, from_node, to_node):
    """
    Return True if adding from_node -> to_node would create a cycle.
    """

    # A new from -> to connection creates a cycle if to can already reach from.
    return has_path(genome, to_node, from_node)


def get_innovation_number(innovation_tracker, from_node, to_node):
    """
    Reuse the same innovation number for the same structural connection.
    """

    if innovation_tracker is None:
        return None
    else:
        key = (from_node, to_node)
        innov = innovation_tracker["connections"].get(key)
        if innov is None:
            innov = innovation_tracker["next_innov"]
            innovation_tracker["connections"][key] = innov
            innovation_tracker["next_innov"] += 1
        return innov


def add_connection_gene(
    genome,
    from_node,
    to_node,
    weight=None,
    enabled=True,
    innovation_tracker=None,
):
    """
    Add one connection gene to the genome.
    """

    if weight is None:
        weight = float(np.random.uniform(-1.0, 1.0))

    innov = get_innovation_number(innovation_tracker, from_node, to_node)
    if innov is None:
        innov = genome["next_innov"]
        genome["next_innov"] += 1

    genome["connections"].append({
        "from": from_node,
        "to": to_node,
        "weight": weight,
        "enabled": enabled,
        "innov": innov,
    })
    genome["next_innov"] = max(genome["next_innov"], innov + 1)


def random_hidden_node_type():
    """
    Pick a random hidden node type and any activation metadata it needs.
    """

    node_type = str(np.random.choice(["add", "mult", "activation", "custom"]))
    if node_type == "activation":
        return {
            "type": node_type,
            "activation": str(np.random.choice(list(ACTIVATION_TYPES))),
        }
    if node_type == "custom":
        return {
            "type": node_type,
            "activation": str(np.random.choice(list(CUSTOM_TYPES))),
        }
    return {"type": node_type}


def mutate_add_node(genome, innovation_tracker=None):
    """
    Disable one connection and insert a new node between its endpoints.
    """

    enabled_connections = [
        connection for connection in genome["connections"]
        if connection["enabled"]
    ]
    if not enabled_connections:
        return False

    connection = enabled_connections[np.random.randint(len(enabled_connections))]
    connection["enabled"] = False

    new_node_id = genome["next_node_id"]
    genome["next_node_id"] += 1
    genome["nodes"][new_node_id] = random_hidden_node_type()

    # Preserve the old path: from -> new node -> to.
    add_connection_gene(
        genome,
        connection["from"],
        new_node_id,
        weight=1.0,
        innovation_tracker=innovation_tracker,
    )
    add_connection_gene(
        genome,
        new_node_id,
        connection["to"],
        weight=connection["weight"],
        innovation_tracker=innovation_tracker,
    )
    return True


def mutate_add_connection(genome, max_attempts=50, innovation_tracker=None):
    """
    Add random connection between two existing nodes.
    """

    node_ids = list(genome["nodes"])

    for _ in range(max_attempts):
        from_node = node_ids[np.random.randint(len(node_ids))]
        to_node = node_ids[np.random.randint(len(node_ids))]
        from_type = genome["nodes"][from_node]["type"]
        to_type = genome["nodes"][to_node]["type"]

        # Inputs and bias should only send values; outputs should only receive.
        if from_node == to_node or from_type == "output":
            continue
        if to_type in {"input", "bias"}:
            continue
        if connection_exists(genome, from_node, to_node):
            continue
        if creates_cycle(genome, from_node, to_node):
            continue

        add_connection_gene(
            genome,
            from_node,
            to_node,
            innovation_tracker=innovation_tracker,
        )
        return True

    return False


def mutate_change_weight(genome, mutation_rate=0.8, perturb_rate=0.9, step_size=0.5):
    """
    Change connection weights.

    mutation_rate: Chance each connection is selected for mutation
    perturb_rate: Chance selected weights are adjusted instead of reset --> stable optimization vs exploration
    step_size: Maximum amount added/subtracted during perturbation
    """

    for connection in genome["connections"]:
        if np.random.random() > mutation_rate:
            continue

        if np.random.random() < perturb_rate:
            connection["weight"] += float(np.random.uniform(-step_size, step_size))
        else:
            connection["weight"] = float(np.random.uniform(-1.0, 1.0))


def mutate_toggle_connection(genome):
    """
    Disable or re-enable one random connection.
    """

    if not genome["connections"]:
        return False

    connection = genome["connections"][np.random.randint(len(genome["connections"]))]
    if connection["enabled"]:
        connection["enabled"] = False
        return True

    # Only re-enable the connection if doing so keeps the graph feed-forward.
    if creates_cycle(genome, connection["from"], connection["to"]):
        return False

    connection["enabled"] = True
    return True


def count_genes(genome):
    """
    Count node genes plus connection genes for complexity penalties.
    """

    return len(genome["nodes"]) + len(genome["connections"])
