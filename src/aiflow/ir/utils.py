from __future__ import annotations

from aiflow.ir.graph import Graph, Node, ValidationError
from aiflow.ir.graph import GraphValidator as _GraphValidator


def build_producer_map(graph: Graph) -> dict[str, int]:
    """
    Map tensor name -> producing node index. Graph inputs have no producer.
    Raises ValidationError on duplicate producers.
    """
    producer: dict[str, int] = {}
    for idx, node in enumerate(graph.nodes):
        for out in node.outputs:
            if out in producer:
                raise ValidationError(
                    f"Multiple producers for tensor '{out}' at node {idx} and {producer[out]}",
                    code="EDUP_PRODUCER",
                    node_index=idx,
                )
            producer[out] = idx
    return producer


def build_consumer_map(graph: Graph) -> dict[str, list[int]]:
    """
    Map tensor name -> list of consuming node indices.
    """
    consumers: dict[str, list[int]] = {}
    for idx, node in enumerate(graph.nodes):
        for inp in node.inputs:
            consumers.setdefault(inp, []).append(idx)
    return consumers


def find_linear_chains(graph: Graph, op_types: list[str]) -> list[list[int]]:
    """
    Find linear chains of nodes matching the exact op_types sequence.
    Constraints (simple and conservative):
    - Each node has exactly one output.
    - The output of each node is consumed by exactly one next node in the chain.
    - The consumer op_type matches the next op in op_types.
    Returns a list of index chains (each chain is a list of node indices).
    """
    if not op_types:
        return []

    consumers = build_consumer_map(graph)
    results: list[list[int]] = []

    for idx, node in enumerate(graph.nodes):
        if node.op_type != op_types[0]:
            continue
        if len(node.outputs) != 1:
            continue
        chain = [idx]
        ok = True
        current_output = node.outputs[0]
        # step through remaining ops
        for expect_op in op_types[1:]:
            use_list = consumers.get(current_output, [])
            if len(use_list) != 1:
                ok = False
                break
            next_idx = use_list[0]
            next_node = graph.nodes[next_idx]
            if next_node.op_type != expect_op:
                ok = False
                break
            if len(next_node.outputs) != 1:
                ok = False
                break
            chain.append(next_idx)
            current_output = next_node.outputs[0]
        if ok:
            results.append(chain)
    return results


def extract_subgraph(graph: Graph, node_indices: set[int]) -> Graph:
    """
    Create a new Graph that contains only the given node indices.
    - Nodes are ordered topologically (based on the original graph's dependencies).
    - Tensors include any referenced by the included nodes (inputs/outputs).
    - Graph inputs: tensors consumed by included nodes but not produced within the included set.
    - Graph outputs: tensors produced by included nodes that are consumed outside the set
      or are terminal.
    """
    if not node_indices:
        return Graph()

    # Ensure graph is valid and get topo order
    _GraphValidator(graph).toposort()
    # Filter order to included nodes preserving relative order from 'order'
    ordered_indices = [i for i in range(len(graph.nodes)) if i in node_indices]

    new_graph = Graph()
    # Collect tensors used/produced by included nodes
    included_producers = set()
    produced_tensors = set()
    used_tensors = set()
    consumer_map = build_consumer_map(graph)
    producer_map = build_producer_map(graph)

    for i in ordered_indices:
        node = graph.nodes[i]
        new_graph.add_node(
            Node(
                op_type=node.op_type,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                attributes=dict(node.attributes),
                metadata=dict(node.metadata),
            )
        )
        for t in node.inputs:
            used_tensors.add(t)
        for t in node.outputs:
            produced_tensors.add(t)
            included_producers.add(i)

    # Add tensors metadata
    referenced = used_tensors.union(produced_tensors)
    for name in referenced:
        t = graph.get_tensor(name)
        if t:
            new_graph.add_tensor(
                # shallow copy
                type(t)(
                    name=t.name,
                    dtype=t.dtype,
                    shape=list(t.shape),
                    layout=t.layout,
                    metadata=dict(t.metadata),
                )
            )

    # Determine graph inputs: used tensors that are not produced by included nodes
    inputs: list[str] = []
    for t in used_tensors:
        prod_idx = producer_map.get(t)
        if prod_idx is None or prod_idx not in node_indices:
            inputs.append(t)
    new_graph.inputs = inputs

    # Determine graph outputs:
    # produced tensors that are either consumed by nodes outside the set
    # or are original graph outputs
    outputs: list[str] = []
    for t in produced_tensors:
        consumers = consumer_map.get(t, [])
        any_outside = any(c not in node_indices for c in consumers)
        if any_outside or t in graph.outputs or not consumers:
            outputs.append(t)
    new_graph.outputs = outputs

    return new_graph
