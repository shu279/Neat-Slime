"""
Save and load evolved genomes.
"""

import copy
import json
from pathlib import Path


def _stringify_node_ids(genome):
    """
    Convert integer node IDs to strings so the genome is JSON-compatible.
    """

    genome_copy = copy.deepcopy(genome)
    genome_copy["nodes"] = {
        str(node_id): node
        for node_id, node in genome_copy["nodes"].items()
    }
    return genome_copy


def _restore_node_ids(genome):
    """
    Convert JSON string node IDs back to integers after loading.
    """

    genome_copy = copy.deepcopy(genome)
    genome_copy["nodes"] = {
        int(node_id): node
        for node_id, node in genome_copy["nodes"].items()
    }
    return genome_copy


def save_genome(path, genome, fitness=None, metadata=None):
    """
    Save one genome and optional fitness metadata as JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "fitness": None if fitness is None else float(fitness),
        "metadata": {} if metadata is None else metadata,
        "genome": _stringify_node_ids(genome),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_genome(path):
    """
    Load a genome JSON file saved by save_genome.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return _restore_node_ids(payload["genome"]), payload.get("fitness")


def load_genome_payload(path):
    """
    Load the full saved JSON payload, including metadata.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    payload["genome"] = _restore_node_ids(payload["genome"])
    payload.setdefault("metadata", {})
    return payload
