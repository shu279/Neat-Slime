#!/usr/bin/env python3
"""
Render a saved NEAT genome as a Graphviz graph.

The style is inspired by the compact Backprop NEAT diagrams from Otoro's demo:
colored node types, arbitrary NEAT topology, and weighted directed edges.
"""

import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess


INPUT_LABELS = [
    "self_x",
    "self_y",
    "self_vx",
    "self_vy",
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "opp_x",
    "opp_y",
    "opp_vx",
    "opp_vy",
]

OUTPUT_LABELS = [
    "move_left",
    "move_right",
    "jump",
]

NODE_STYLES = {
    "input": ("#f6e58d", "#665c00"),
    "bias": ("#dff9fb", "#0a5c61"),
    "output": ("#badc58", "#336600"),
    "sigmoid": ("#7ed6df", "#0c5460"),
    "tanh": ("#686de0", "#ffffff"),
    "relu": ("#ffbe76", "#5c2c00"),
    "gaussian": ("#c7ecee", "#155c63"),
    "sin": ("#e056fd", "#ffffff"),
    "abs": ("#f9ca24", "#5f4a00"),
    "square": ("#eb4d4b", "#ffffff"),
    "mult": ("#30336b", "#ffffff"),
    "add": ("#95afc0", "#132f3d"),
}


def quote(value):
    """
    Quote a string for DOT attributes.
    """

    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def dot_attrs(attributes):
    """
    Format a DOT attribute dictionary.
    """

    return ", ".join(f"{key}={quote(value)}" for key, value in attributes.items())


def node_kind(node):
    """
    Return the visual kind used to color a node.
    """

    node_type = node.get("type", "unknown")
    if node_type in {"activation", "custom", "output"}:
        return node.get("activation", "sigmoid")
    return node_type


def node_label(node_id, node, genome):
    """
    Build a readable node label.
    """

    if node.get("type") == "input":
        input_ids = genome.get("input_ids", [])
        if node_id in input_ids:
            index = input_ids.index(node_id)
            if index < len(INPUT_LABELS):
                return f"{INPUT_LABELS[index]}\\n#{node_id}"

    if node.get("type") == "bias":
        return f"bias\\n#{node_id}"

    if node.get("type") == "output":
        output_ids = genome.get("output_ids", [])
        if node_id in output_ids:
            index = output_ids.index(node_id)
            if index < len(OUTPUT_LABELS):
                return f"{OUTPUT_LABELS[index]}\\n#{node_id}"

    kind = node_kind(node)
    return f"{kind}\\n#{node_id}"


def edge_style(weight, enabled=True):
    """
    Convert an edge weight into Graphviz visual attributes.
    """

    magnitude = min(abs(weight), 5.0)
    penwidth = 0.8 + magnitude * 0.75
    if weight >= 0:
        color = "#1f78b4"
    else:
        color = "#e66101"

    attrs = {
        "color": color,
        "penwidth": f"{penwidth:.2f}",
        "arrowsize": "0.65",
    }
    if not enabled:
        attrs.update({
            "style": "dashed",
            "color": "#b2bec3",
            "penwidth": "0.8",
        })
    return attrs


def build_dot(saved_genome, show_disabled=False, show_weights=False):
    """
    Build Graphviz DOT text from a saved genome JSON object.
    """

    genome = saved_genome.get("genome", saved_genome)
    nodes = genome["nodes"]
    connections = genome["connections"]
    metadata = saved_genome.get("metadata", {})
    fitness = saved_genome.get("fitness")

    lines = [
        "digraph genome {",
        "  graph [",
        "    rankdir=LR,",
        "    bgcolor=\"#ffffff\",",
        "    pad=\"0.35\",",
        "    nodesep=\"0.45\",",
        "    ranksep=\"0.85\",",
        "    splines=true,",
        "    overlap=false,",
        "    fontname=\"Helvetica\"",
        "  ];",
        "  node [shape=circle, style=filled, fontname=\"Helvetica\", fontsize=10, margin=\"0.05\"];",
        "  edge [fontname=\"Helvetica\", fontsize=8];",
        "",
    ]

    title = "Best genome"
    if fitness is not None:
        title += f" | fitness={fitness:.3f}"
    if "generation" in metadata:
        title += f" | gen={metadata['generation']}"
    lines.append(f"  labelloc=t;")
    lines.append(f"  label={quote(title)};")
    lines.append("")

    for node_id_text, node in sorted(nodes.items(), key=lambda item: int(item[0])):
        node_id = int(node_id_text)
        kind = node_kind(node)
        fill, font_color = NODE_STYLES.get(kind, ("#dfe6e9", "#2d3436"))
        attrs = {
            "label": node_label(node_id, node, genome),
            "fillcolor": fill,
            "fontcolor": font_color,
            "color": "#2d3436",
        }
        if node.get("type") == "output":
            attrs["shape"] = "doublecircle"
        elif node.get("type") in {"add", "mult"}:
            attrs["shape"] = "box"
            attrs["width"] = "0.55"
            attrs["height"] = "0.38"
        lines.append(f"  n{node_id} [{dot_attrs(attrs)}];")

    input_nodes = [f"n{node_id}" for node_id in genome.get("input_ids", [])]
    output_nodes = [f"n{node_id}" for node_id in genome.get("output_ids", [])]
    if genome.get("bias_id") is not None:
        input_nodes.append(f"n{genome['bias_id']}")

    lines.append("")
    if input_nodes:
        lines.append(f"  {{ rank=source; {'; '.join(input_nodes)}; }}")
    if output_nodes:
        lines.append(f"  {{ rank=sink; {'; '.join(output_nodes)}; }}")
    lines.append("")

    for connection in connections:
        enabled = connection.get("enabled", True)
        if not enabled and not show_disabled:
            continue

        attrs = edge_style(connection["weight"], enabled=enabled)
        attrs["tooltip"] = f"w={connection['weight']:.4f}, innov={connection.get('innov', '?')}"
        if show_weights:
            attrs["label"] = f"{connection['weight']:.2f}"
        lines.append(
            f"  n{connection['from']} -> n{connection['to']} [{dot_attrs(attrs)}];"
        )

    lines.extend([
        "",
        "  subgraph cluster_legend {",
        "    label=\"Legend\";",
        "    color=\"#dfe6e9\";",
        "    fontsize=10;",
        "    key_pos [label=\"positive weight\", shape=plaintext];",
        "    key_neg [label=\"negative weight\", shape=plaintext];",
        "    key_a [label=\"\", width=0.01, height=0.01, style=invis];",
        "    key_b [label=\"\", width=0.01, height=0.01, style=invis];",
        "    key_a -> key_pos [color=\"#1f78b4\", penwidth=2.0, arrowsize=0.65];",
        "    key_b -> key_neg [color=\"#e66101\", penwidth=2.0, arrowsize=0.65];",
        "  }",
        "}",
        "",
    ])
    return "\n".join(lines)


def render_dot(dot_path, output_path, output_format):
    """
    Render DOT via the Graphviz dot binary.
    """

    dot_binary = shutil.which("dot")
    if dot_binary is None:
        return False

    subprocess.run(
        [dot_binary, f"-T{output_format}", str(dot_path), "-o", str(output_path)],
        check=True,
    )
    return True


def compute_node_positions(
    genome,
    width_step=210,
    y_step=58,
    margin_x=70,
    margin_y=70,
    max_nodes_per_column=7,
):
    """
    Compute a simple left-to-right layered layout for SVG fallback rendering.
    """

    nodes = genome["nodes"]
    enabled_connections = [
        connection
        for connection in genome["connections"]
        if connection.get("enabled", True)
    ]
    incoming = {int(node_id): [] for node_id in nodes}
    for connection in enabled_connections:
        incoming[int(connection["to"])].append(int(connection["from"]))

    input_ids = [int(node_id) for node_id in genome.get("input_ids", [])]
    output_ids = [int(node_id) for node_id in genome.get("output_ids", [])]
    bias_id = genome.get("bias_id")
    source_ids = set(input_ids)
    if bias_id is not None:
        source_ids.add(int(bias_id))

    depth_cache = {}

    def depth(node_id, visiting=None):
        if node_id in depth_cache:
            return depth_cache[node_id]
        if node_id in source_ids:
            depth_cache[node_id] = 0
            return 0
        if visiting is None:
            visiting = set()
        if node_id in visiting:
            return 1
        visiting.add(node_id)
        parent_depths = [
            depth(parent_id, visiting)
            for parent_id in incoming.get(node_id, [])
            if parent_id not in output_ids
        ]
        visiting.remove(node_id)
        node_depth = 1 + max(parent_depths, default=0)
        depth_cache[node_id] = node_depth
        return node_depth

    hidden_ids = [
        int(node_id)
        for node_id, node in nodes.items()
        if node.get("type") not in {"input", "bias", "output"}
    ]
    for node_id in hidden_ids:
        depth(node_id)

    output_depth = max([depth_cache.get(node_id, 0) for node_id in hidden_ids] + [0]) + 1
    columns = {}
    for node_id_text, node in nodes.items():
        node_id = int(node_id_text)
        if node_id in source_ids:
            column = 0
        elif node_id in output_ids:
            column = output_depth
        else:
            column = max(1, min(depth(node_id), output_depth - 1))
        columns.setdefault(column, []).append(node_id)

    expanded_columns = {}
    next_column = 0
    for column in sorted(columns):
        column_nodes = sorted(columns[column])
        for start in range(0, len(column_nodes), max_nodes_per_column):
            expanded_columns[next_column] = column_nodes[start:start + max_nodes_per_column]
            next_column += 1

    max_column = max(expanded_columns)
    max_count = max(len(column_nodes) for column_nodes in expanded_columns.values())
    width = (max_column * width_step) + (margin_x * 2)
    height = max(560, (max_count - 1) * y_step + margin_y * 2 + 120)
    positions = {}

    for column, column_nodes in expanded_columns.items():
        block_height = (len(column_nodes) - 1) * y_step
        y_start = (height - block_height) / 2
        x = margin_x + column * width_step
        for index, node_id in enumerate(column_nodes):
            positions[node_id] = (x, y_start + index * y_step)

    return positions, width, height


def build_svg(saved_genome, show_disabled=False, show_weights=False):
    """
    Build a simple standalone SVG when Graphviz is unavailable.
    """

    genome = saved_genome.get("genome", saved_genome)
    nodes = genome["nodes"]
    connections = genome["connections"]
    metadata = saved_genome.get("metadata", {})
    fitness = saved_genome.get("fitness")
    positions, width, height = compute_node_positions(genome)

    title = "Best genome"
    if fitness is not None:
        title += f" | fitness={fitness:.3f}"
    if "generation" in metadata:
        title += f" | gen={metadata['generation']}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow-pos" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L7,3 z" fill="#1f78b4"/>',
        "</marker>",
        '<marker id="arrow-neg" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L7,3 z" fill="#e66101"/>',
        "</marker>",
        '<marker id="arrow-disabled" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L7,3 z" fill="#b2bec3"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#2d3436">'
        f"{html.escape(title)}</text>",
    ]

    for connection in connections:
        enabled = connection.get("enabled", True)
        if not enabled and not show_disabled:
            continue

        source = int(connection["from"])
        target = int(connection["to"])
        if source not in positions or target not in positions:
            continue

        x1, y1 = positions[source]
        x2, y2 = positions[target]
        weight = connection["weight"]
        magnitude = min(abs(weight), 5.0)
        stroke_width = 0.8 + magnitude * 0.75
        if not enabled:
            color = "#b2bec3"
            marker = "arrow-disabled"
            dash = ' stroke-dasharray="5 4"'
            stroke_width = 0.8
        elif weight >= 0:
            color = "#1f78b4"
            marker = "arrow-pos"
            dash = ""
        else:
            color = "#e66101"
            marker = "arrow-neg"
            dash = ""

        dx = max(40.0, abs(x2 - x1) * 0.45)
        path = (
            f"M{x1 + 24:.1f},{y1:.1f} "
            f"C{x1 + dx:.1f},{y1:.1f} {x2 - dx:.1f},{y2:.1f} {x2 - 24:.1f},{y2:.1f}"
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width:.2f}"'
            f' opacity="0.72" marker-end="url(#{marker})"{dash}/>'
        )
        if show_weights:
            label_x = (x1 + x2) / 2
            label_y = (y1 + y2) / 2 - 4
            parts.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                'font-family="Helvetica, Arial, sans-serif" font-size="9" fill="#636e72">'
                f'{weight:.2f}</text>'
            )

    for node_id_text, node in sorted(nodes.items(), key=lambda item: int(item[0])):
        node_id = int(node_id_text)
        x, y = positions[node_id]
        kind = node_kind(node)
        fill, font_color = NODE_STYLES.get(kind, ("#dfe6e9", "#2d3436"))
        label_lines = node_label(node_id, node, genome).split("\\n")

        if node.get("type") in {"add", "mult"}:
            parts.append(
                f'<rect x="{x - 28:.1f}" y="{y - 20:.1f}" width="56" height="40" rx="5" '
                f'fill="{fill}" stroke="#2d3436" stroke-width="1.2"/>'
            )
        else:
            radius = 25 if node.get("type") == "output" else 23
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
                f'fill="{fill}" stroke="#2d3436" stroke-width="1.2"/>'
            )
            if node.get("type") == "output":
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius - 4}" '
                    'fill="none" stroke="#2d3436" stroke-width="1.0"/>'
                )

        for index, line in enumerate(label_lines):
            line_y = y - 4 + index * 11
            parts.append(
                f'<text x="{x:.1f}" y="{line_y:.1f}" text-anchor="middle" '
                'font-family="Helvetica, Arial, sans-serif" font-size="9" '
                f'fill="{font_color}">{html.escape(line)}</text>'
            )

    legend_y = height - 70
    parts.extend([
        f'<text x="28" y="{legend_y}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="11" fill="#2d3436">Legend</text>',
        f'<line x1="28" y1="{legend_y + 20}" x2="84" y2="{legend_y + 20}" '
        'stroke="#1f78b4" stroke-width="3"/>',
        f'<text x="94" y="{legend_y + 24}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="10" fill="#2d3436">positive weight</text>',
        f'<line x1="220" y1="{legend_y + 20}" x2="276" y2="{legend_y + 20}" '
        'stroke="#e66101" stroke-width="3"/>',
        f'<text x="286" y="{legend_y + 24}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="10" fill="#2d3436">negative weight</text>',
        "</svg>",
        "",
    ])
    return "\n".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a saved NEAT genome with Graphviz DOT.",
    )
    parser.add_argument(
        "--path",
        default="saved/best_genome.json",
        help="Saved genome JSON path.",
    )
    parser.add_argument(
        "--out",
        default="saved/best_genome_graph",
        help="Output path without extension, or with the final render extension.",
    )
    parser.add_argument(
        "--format",
        default="svg",
        choices=["svg", "png", "pdf"],
        help="Rendered output format when Graphviz dot is installed.",
    )
    parser.add_argument(
        "--show-disabled",
        action="store_true",
        help="Include disabled genes as dashed gray edges.",
    )
    parser.add_argument(
        "--show-weights",
        action="store_true",
        help="Print numeric weights on edges.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    genome_path = Path(args.path)
    out_path = Path(args.out)

    with genome_path.open("r", encoding="utf-8") as file:
        saved_genome = json.load(file)

    if out_path.suffix:
        render_path = out_path
        dot_path = out_path.with_suffix(".dot")
    else:
        render_path = out_path.with_suffix(f".{args.format}")
        dot_path = out_path.with_suffix(".dot")

    dot_text = build_dot(
        saved_genome,
        show_disabled=args.show_disabled,
        show_weights=args.show_weights,
    )
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(dot_text, encoding="utf-8")
    print(f"Wrote DOT: {dot_path}")

    rendered = render_dot(dot_path, render_path, args.format)
    if rendered:
        print(f"Wrote {args.format.upper()}: {render_path}")
    else:
        if args.format == "svg":
            svg_text = build_svg(
                saved_genome,
                show_disabled=args.show_disabled,
                show_weights=args.show_weights,
            )
            render_path.write_text(svg_text, encoding="utf-8")
            print(f"Wrote fallback SVG: {render_path}")
        else:
            print("Graphviz 'dot' was not found; install Graphviz to render this format.")


if __name__ == "__main__":
    main()
