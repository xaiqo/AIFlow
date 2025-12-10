from __future__ import annotations

from collections.abc import Callable
from math import prod

import numpy as np

from aiflow.ir.graph import Graph, GraphValidator, Node


class InferenceError(Exception):
    def __init__(self, message: str, code: str = "EINFER") -> None:
        super().__init__(message)
        self.code = code


ShapeInferFn = Callable[[Graph, Node], None]

_REGISTRY: dict[str, ShapeInferFn] = {}


def register_shape_inference(op_type: str) -> Callable[[ShapeInferFn], ShapeInferFn]:
    def wrapper(fn: ShapeInferFn) -> ShapeInferFn:
        _REGISTRY[op_type] = fn
        return fn

    return wrapper


def _broadcast_shape(a: list[int], b: list[int]) -> list[int]:
    ra = list(reversed(a))
    rb = list(reversed(b))
    result: list[int] = []
    for i in range(max(len(ra), len(rb))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == db or da == 1 or db == 1:
            result.append(max(da, db))
        else:
            raise InferenceError(f"Broadcast mismatch: {a} vs {b}", code="EBROADCAST")
    return list(reversed(result))


def _promote_dtype(dtype_a: str, dtype_b: str) -> str:
    # Use numpy's type promotion to resolve a result type name
    return str(np.result_type(dtype_a, dtype_b).name)


@register_shape_inference("Add")
def infer_add(graph: Graph, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise InferenceError("Add expects 2 inputs and 1 output", code="EADD_ARITY")
    a = graph.tensors[node.inputs[0]]
    b = graph.tensors[node.inputs[1]]
    out = graph.tensors[node.outputs[0]]
    out.shape = _broadcast_shape(a.shape, b.shape)
    out.dtype = _promote_dtype(a.dtype, b.dtype)


@register_shape_inference("Relu")
def infer_relu(graph: Graph, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise InferenceError("Relu expects 1 input and 1 output", code="ERELU_ARITY")
    x = graph.tensors[node.inputs[0]]
    out = graph.tensors[node.outputs[0]]
    out.shape = list(x.shape)
    out.dtype = x.dtype


@register_shape_inference("MatMul")
def infer_matmul(graph: Graph, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise InferenceError(
            "MatMul expects 2 inputs and 1 output", code="EMATMUL_ARITY"
        )
    a = graph.tensors[node.inputs[0]]
    b = graph.tensors[node.inputs[1]]
    out = graph.tensors[node.outputs[0]]
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise InferenceError(
            "MatMul requires tensors with rank >= 2", code="EMATMUL_RANK"
        )
    a_batch = a.shape[:-2]
    b_batch = b.shape[:-2]
    batch = _broadcast_shape(a_batch, b_batch)
    m, k1 = a.shape[-2], a.shape[-1]
    k2, n = b.shape[-2], b.shape[-1]
    if k1 != k2:
        raise InferenceError(
            f"Incompatible MatMul inner dims: {k1} vs {k2}", code="EMATMUL_DIMS"
        )
    out.shape = list(batch) + [m, n]
    out.dtype = _promote_dtype(a.dtype, b.dtype)


@register_shape_inference("Reshape")
def infer_reshape(graph: Graph, node: Node) -> None:
    if len(node.outputs) != 1 or len(node.inputs) not in (1, 2):
        raise InferenceError(
            "Reshape expects 1 or 2 inputs and 1 output", code="ERESHAPE_ARITY"
        )
    x = graph.tensors[node.inputs[0]]
    out = graph.tensors[node.outputs[0]]
    target = node.attributes.get("shape")
    if target is None and len(node.inputs) == 2:
        # Try to read shape from second input tensor's const metadata
        shape_input = graph.tensors.get(node.inputs[1])
        if shape_input is not None:
            const_val = shape_input.metadata.get("const")
            if isinstance(const_val, list):
                # Flatten one level if nested
                if const_val and isinstance(const_val[0], list):
                    flat = []
                    for v in const_val[0]:
                        try:
                            flat.append(int(v))
                        except Exception:
                            pass
                    const_val = flat
                try:
                    target = [int(v) for v in const_val]
                except Exception as _:
                    target = None
    if not isinstance(target, list) or not all(isinstance(d, int) for d in target):
        raise InferenceError(
            "Reshape requires 'shape' attribute (list[int])", code="ERESHAPE_ATTR"
        )
    neg_one_count = sum(1 for d in target if d == -1)
    if neg_one_count > 1:
        raise InferenceError(
            "Reshape 'shape' may contain at most one -1", code="ERESHAPE_NEG1"
        )
    known = [d for d in target if d != -1]
    if any(d <= 0 for d in known):
        raise InferenceError(
            "Reshape dims must be positive (or -1 for infer)",
            code="ERESHAPE_DIMS",
        )
    total_in = prod(x.shape) if x.shape else 0
    if neg_one_count == 0:
        if prod(target) != total_in:
            raise InferenceError(
                "Reshape element count mismatch", code="ERESHAPE_COUNT"
            )
        out.shape = list(target)
    else:
        known_prod = prod(known) if known else 1
        if total_in % known_prod != 0:
            raise InferenceError(
                "Reshape cannot infer -1 dimension", code="ERESHAPE_INF"
            )
        inferred = total_in // known_prod
        new_shape: list[int] = []
        used_infer = False
        for d in target:
            if d == -1 and not used_infer:
                new_shape.append(inferred)
                used_infer = True
            else:
                new_shape.append(d)
        out.shape = new_shape
    out.dtype = x.dtype


@register_shape_inference("Transpose")
def infer_transpose(graph: Graph, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise InferenceError(
            "Transpose expects 1 input and 1 output", code="ETRANSPOSE_ARITY"
        )
    x = graph.tensors[node.inputs[0]]
    out = graph.tensors[node.outputs[0]]
    rank = len(x.shape)
    perm = node.attributes.get("perm")
    if perm is None:
        perm = list(reversed(range(rank)))
    if (
        not isinstance(perm, list)
        or len(perm) != rank
        or any(not isinstance(p, int) or p < 0 or p >= rank for p in perm)
    ):
        raise InferenceError("Invalid Transpose perm", code="ETRANSPOSE_PERM")
    out.shape = [x.shape[i] for i in perm]
    out.dtype = x.dtype


def _pool2d_out_dim(input_dim: int, k: int, s: int, p0: int, p1: int) -> int:
    # floor((in + pad0 + pad1 - kernel)/stride) + 1
    return (input_dim + p0 + p1 - k) // s + 1


def _conv2d_out_dim(input_dim: int, k: int, s: int, d: int, p0: int, p1: int) -> int:
    # floor((in + pad0 + pad1 - dilation*(k - 1) - 1)/stride + 1)
    return (input_dim + p0 + p1 - d * (k - 1) - 1) // s + 1


def _get_list_attr(node: Node, name: str, default: list[int]) -> list[int]:
    val = node.attributes.get(name)
    if val is None:
        return default
    if not isinstance(val, list) or not all(isinstance(x, int) for x in val):
        raise InferenceError(f"Attribute '{name}' must be list[int]", code="EATTR")
    return val


def _pool2d_shape(
    x_shape: list[int],
    kernel: list[int],
    strides: list[int],
    pads: list[int],
) -> list[int]:
    if len(x_shape) < 4:
        raise InferenceError("Pool2D expects rank >= 4", code="EPOOL_RANK")
    h_in = x_shape[-2]
    w_in = x_shape[-1]
    k_h, k_w = kernel if len(kernel) == 2 else (kernel[0], kernel[0])
    s_h, s_w = strides if len(strides) == 2 else (1, 1)
    if len(pads) == 2:
        p_top = p_bottom = pads[0]
        p_left = p_right = pads[1]
    elif len(pads) == 4:
        p_top, p_left, p_bottom, p_right = pads
    else:
        p_top = p_left = p_bottom = p_right = 0
    h_out = _pool2d_out_dim(h_in, k_h, s_h, p_top, p_bottom)
    w_out = _pool2d_out_dim(w_in, k_w, s_w, p_left, p_right)
    return list(x_shape[:-2]) + [h_out, w_out]


@register_shape_inference("Conv")
def infer_conv(graph: Graph, node: Node) -> None:
    if len(node.inputs) < 2 or len(node.outputs) != 1:
        raise InferenceError("Conv expects at least 2 inputs and 1 output", code="ECONV_ARITY")
    x = graph.tensors[node.inputs[0]]
    w = graph.tensors[node.inputs[1]]
    out = graph.tensors[node.outputs[0]]
    if len(x.shape) != 4 or len(w.shape) != 4:
        raise InferenceError("Conv expects rank-4 tensors (NCHW, OIHW)", code="ECONV_RANK")

    n, c_in, h_in, w_in = x.shape
    c_out, c_per_group, k_h, k_w = w.shape

    groups = node.attributes.get("groups", 1)
    if not isinstance(groups, int) or groups <= 0:
        raise InferenceError("Conv groups must be positive integer", code="ECONV_GROUPS")
    if c_in % groups != 0:
        raise InferenceError("Conv input channels not divisible by groups", code="ECONV_GROUPS_DIV")
    if c_per_group * groups != c_in:
        # According to ONNX, W shape is [Cout, Cin/groups, kH, kW]
        raise InferenceError("Conv weight shape mismatch Cin/groups", code="ECONV_W_SHAPE")

    strides = _get_list_attr(node, "strides", [1, 1])
    dilations = _get_list_attr(node, "dilations", [1, 1])
    pads = _get_list_attr(node, "pads", [0, 0, 0, 0])  # [pad_top, pad_left, pad_bottom, pad_right]

    if len(strides) != 2 or len(dilations) != 2:
        raise InferenceError("Conv strides/dilations must be length-2", code="ECONV_ATTRS")
    if len(pads) not in (2, 4):
        raise InferenceError("Conv pads must be length-2 or -4", code="ECONV_PADS")

    s_h, s_w = strides
    d_h, d_w = dilations
    if len(pads) == 2:
        p_top = p_bottom = pads[0]
        p_left = p_right = pads[1]
    else:
        p_top, p_left, p_bottom, p_right = pads

    h_out = _conv2d_out_dim(h_in, k_h, s_h, d_h, p_top, p_bottom)
    w_out = _conv2d_out_dim(w_in, k_w, s_w, d_w, p_left, p_right)

    out.shape = [n, c_out, h_out, w_out]
    out.dtype = x.dtype

@register_shape_inference("MaxPool")
def infer_maxpool(graph: Graph, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise InferenceError("MaxPool expects 1 input and 1 output", code="EPOOL_ARITY")
    x = graph.tensors[node.inputs[0]]
    out = graph.tensors[node.outputs[0]]
    kernel = _get_list_attr(node, "kernel_shape", [])
    if not kernel:
        raise InferenceError("MaxPool requires kernel_shape", code="EPOOL_KERNEL")
    strides = _get_list_attr(node, "strides", [1, 1])
    pads = _get_list_attr(node, "pads", [0, 0, 0, 0])
    out.shape = _pool2d_shape(x.shape, kernel, strides, pads)
    out.dtype = x.dtype


@register_shape_inference("AveragePool")
def infer_avgpool(graph: Graph, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise InferenceError(
            "AveragePool expects 1 input and 1 output", code="EPOOL_ARITY"
        )
    x = graph.tensors[node.inputs[0]]
    out = graph.tensors[node.outputs[0]]
    kernel = _get_list_attr(node, "kernel_shape", [])
    if not kernel:
        raise InferenceError("AveragePool requires kernel_shape", code="EPOOL_KERNEL")
    strides = _get_list_attr(node, "strides", [1, 1])
    pads = _get_list_attr(node, "pads", [0, 0, 0, 0])
    out.shape = _pool2d_shape(x.shape, kernel, strides, pads)
    out.dtype = x.dtype


@register_shape_inference("Concat")
def infer_concat(graph: Graph, node: Node) -> None:
    if len(node.outputs) != 1 or len(node.inputs) < 1:
        raise InferenceError(
            "Concat expects N inputs and 1 output", code="ECONCAT_ARITY"
        )
    tensors = [graph.tensors[name] for name in node.inputs]
    out = graph.tensors[node.outputs[0]]
    rank = len(tensors[0].shape)
    if any(len(t.shape) != rank for t in tensors):
        raise InferenceError("Concat inputs must have same rank", code="ECONCAT_RANK")
    axis = node.attributes.get("axis", 0)
    if not isinstance(axis, int):
        raise InferenceError("Concat axis must be int", code="ECONCAT_AXIS")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise InferenceError("Concat axis out of range", code="ECONCAT_AXIS")
    out_shape = list(tensors[0].shape)
    out_shape[axis] = 0
    for t in tensors:
        for d in range(rank):
            if d == axis:
                continue
            if t.shape[d] != out_shape[d]:
                raise InferenceError("Concat dim mismatch", code="ECONCAT_DIMS")
        out_shape[axis] += t.shape[axis]
    out.shape = out_shape
    out.dtype = tensors[0].dtype


def infer_graph(graph: Graph) -> None:
    """
    Run shape/dtype inference over the graph in topological order.
    """
    order = GraphValidator(graph).toposort()
    # Map nodes to their indices to walk in order
    for node in order:
        fn = _REGISTRY.get(node.op_type)
        if fn is None:
            # Unknown op: leave shapes as-is
            continue
        fn(graph, node)
