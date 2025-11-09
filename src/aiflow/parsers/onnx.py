from __future__ import annotations

from typing import Any

import onnx
from onnx import numpy_helper

from aiflow.ir import Graph, GraphValidator, Node, Tensor, infer_graph

_DTYPE_MAP = {
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.UINT16: "uint16",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.DOUBLE: "float64",
    onnx.TensorProto.UINT32: "uint32",
    onnx.TensorProto.UINT64: "uint64",
    onnx.TensorProto.BFLOAT16: "bfloat16",
}


def _dtype_from_value_info(vi: onnx.ValueInfoProto) -> str | None:
    t = vi.type.tensor_type
    elem = t.elem_type
    return _DTYPE_MAP.get(elem)


def _shape_from_value_info(vi: onnx.ValueInfoProto) -> list[int] | None:
    dims = vi.type.tensor_type.shape.dim
    out: list[int] = []
    for d in dims:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        else:
            # unknown -> use 1 as placeholder
            out.append(1)
    return out if out else None


def _parse_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            attrs[a.name] = int(a.i)
        elif a.type == onnx.AttributeProto.FLOAT:
            attrs[a.name] = float(a.f)
        elif a.type == onnx.AttributeProto.STRING:
            attrs[a.name] = a.s.decode("utf-8", errors="ignore")
        elif a.type == onnx.AttributeProto.INTS:
            attrs[a.name] = [int(x) for x in a.ints]
        elif a.type == onnx.AttributeProto.FLOATS:
            attrs[a.name] = [float(x) for x in a.floats]
        elif a.type == onnx.AttributeProto.TENSOR:
            # Store as numpy array in metadata if needed
            attrs[a.name] = numpy_helper.to_array(a.t).tolist()
        else:
            # skip other attribute types for MVP
            continue
    return attrs


class OnnxParser:
    """Parse an ONNX model into AIFlow IR Graph (MVP subset)."""

    def parse(self, model_or_path: Any, *, validate_and_infer: bool = True) -> Graph:
        model = self._load_model(model_or_path)
        g = Graph()

        # Initializers -> tensors with const metadata
        init_names: set[str] = set()
        init_arrays: dict[str, list] = {}
        for init in model.graph.initializer:
            name = init.name
            init_names.add(name)
            arr = numpy_helper.to_array(init)
            dtype = str(arr.dtype.name)
            shape = list(arr.shape) if arr.shape else [1]
            g.add_tensor(
                Tensor(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    metadata={"const": arr.tolist()},
                )
            )
            init_arrays[name] = arr.tolist()

        # Inputs -> tensors (skip ones that are initializers)
        for inp in model.graph.input:
            name = inp.name
            if name in init_names:
                continue
            dtype = _dtype_from_value_info(inp) or "float32"
            shape = _shape_from_value_info(inp) or [1]
            g.add_tensor(Tensor(name=name, dtype=dtype, shape=shape))
            g.inputs.append(name)

        # Outputs -> tensors (ensure existence and shapes)
        for out in model.graph.output:
            name = out.name
            dtype = _dtype_from_value_info(out) or "float32"
            shape = _shape_from_value_info(out) or [1]
            if name not in g.tensors:
                g.add_tensor(Tensor(name=name, dtype=dtype, shape=shape))
            else:
                # Update missing dtype/shape if needed
                t = g.tensors[name]
                t.dtype = t.dtype or dtype
                t.shape = t.shape or shape
            g.outputs.append(name)

        # ValueInfo (intermediate tensors with shapes/dtypes)
        for vi in model.graph.value_info:
            name = vi.name
            dtype = _dtype_from_value_info(vi) or "float32"
            shape = _shape_from_value_info(vi) or [1]
            if name not in g.tensors:
                g.add_tensor(Tensor(name=name, dtype=dtype, shape=shape))
            else:
                t = g.tensors[name]
                t.dtype = t.dtype or dtype
                t.shape = t.shape or shape

        # Nodes -> IR nodes and ensure output tensors exist
        for n in model.graph.node:
            attrs = _parse_attributes(n)
            # Special-case: Reshape often uses the 2nd input as the target shape tensor
            if n.op_type == "Reshape" and len(n.input) >= 2:
                shape_name = n.input[1]
                if shape_name in init_arrays:
                    attrs["shape"] = list(init_arrays[shape_name])
            g.add_node(
                Node(
                    op_type=n.op_type,
                    inputs=list(n.input),
                    outputs=list(n.output),
                    attributes=attrs,
                )
            )
            for out_name in n.output:
                if out_name and out_name not in g.tensors:
                    # Create placeholder tensor; shapes may be inferred later
                    g.add_tensor(Tensor(name=out_name, dtype="float32", shape=[1]))

        if validate_and_infer:
            # Basic validation then shape/dtype inference to populate internal tensors
            GraphValidator(g).validate()
            infer_graph(g)
        return g

    def _load_model(self, model_or_path: Any) -> onnx.ModelProto:
        if isinstance(model_or_path, onnx.ModelProto):
            return model_or_path
        if isinstance(model_or_path, (bytes, bytearray)):
            return onnx.load_model_from_string(model_or_path)
        if isinstance(model_or_path, str):
            return onnx.load(model_or_path)
        raise TypeError("Unsupported model type for ONNX parser")
