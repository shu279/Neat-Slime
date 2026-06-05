"""
Evaluate genomes as neural networks and convert outputs into actions.
"""

import numpy as np


def sigmoid(x):
    """
    Apply a numerically stable sigmoid activation.
    """

    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def activate(value, activation_name):
    """
    Apply one named activation/custom operation to a scalar value.
    """

    if activation_name == "sigmoid":
        return sigmoid(value)
    if activation_name == "tanh":
        return np.tanh(value)
    if activation_name == "relu":
        return np.maximum(0.0, value)
    if activation_name == "gaussian":
        return np.exp(-(value ** 2))
    if activation_name == "sin":
        return np.sin(value)
    if activation_name == "abs":
        return np.abs(value)
    if activation_name == "square":
        return value ** 2
    raise ValueError(f"Unknown activation: {activation_name}")


def forward(genome, observation):
    """
    Evaluate a feed-forward genome.

    Returns activated output values in genome["output_ids"] order.
    """

    input_ids = genome["input_ids"]
    observation = np.asarray(observation, dtype=np.float32)

    # Check observation length matches input nodes.
    if len(observation) != len(input_ids):
        raise ValueError(
            f"Expected {len(input_ids)} observation values, got {len(observation)}"
        )

    # Build a mapping of node_id to incoming connections for quick lookup during activation.
    incoming_connections = {node_id: [] for node_id in genome["nodes"]}
    for connection in genome["connections"]:
        if connection["enabled"]:
            incoming_connections[connection["to"]].append(connection)

    node_values = {}

    # Copy observation value to input nodes.
    for node_id, value in zip(input_ids, observation):
        node_values[node_id] = value

    # Bias is an input node that always has value 1.
    node_values[genome["bias_id"]] = 1.0

    def compute_node(node_id, visiting=None):
        """
        Recursively compute one node after computing its input nodes.
        """

        if node_id in node_values:
            return node_values[node_id]

        if visiting is None:
            visiting = set()
        if node_id in visiting:
            raise ValueError("Genome has a recurrent/cyclic connection")
        visiting.add(node_id)

        node = genome["nodes"][node_id]
        node_type = node["type"]
        weighted_inputs = np.asarray([
            compute_node(connection["from"], visiting) * connection["weight"]
            for connection in incoming_connections[node_id]
        ], dtype=np.float32)

        if node_type == "add":
            value = np.sum(weighted_inputs)
        elif node_type == "mult":
            # Empty multiplication would be unhelpful here, so use 0.0.
            value = 0.0 if weighted_inputs.size == 0 else np.prod(weighted_inputs)
        elif node_type in {"output", "activation", "custom"}:
            value = activate(np.sum(weighted_inputs), node.get("activation", "sigmoid"))
        else:
            raise ValueError(f"Cannot compute node type: {node_type}")

        visiting.remove(node_id)
        node_values[node_id] = float(value)
        return value

    # Activate and return outputs in action order.
    return np.asarray(
        [compute_node(output_id) for output_id in genome["output_ids"]],
        dtype=np.float32,
    )


def outputs_to_action(
    outputs,
    threshold=0.5,
    action_mode="exclusive",
    horizontal_margin=0.05,
):
    """
    Map continuous network outputs to binary button.
    """

    outputs = np.asarray(outputs)

    if action_mode == "independent":
        return (outputs >= threshold).astype(np.int8)

    if action_mode == "exclusive":
        action = np.zeros(3, dtype=np.int8)
        horizontal_difference = outputs[0] - outputs[1]

        # Forward/backward are mutually exclusive. The margin avoids jitter
        # when both movement outputs are nearly tied.
        if outputs[0] >= threshold and horizontal_difference > horizontal_margin:
            action[0] = 1
        elif outputs[1] >= threshold and horizontal_difference < -horizontal_margin:
            action[1] = 1

        action[2] = 1 if outputs[2] >= threshold else 0
        return action

    raise ValueError(f"Unknown action_mode: {action_mode}")
