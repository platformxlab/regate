import csv
from functools import lru_cache
import glob
from math import ceil, sqrt
import os
from typing import Any, Sequence

# from npusim.xla_program import XLAProgram
# from npusim.npusim import npu_config_DEFAULT
import trace_util.llm_ops_generator.Operator as Operator
import trace_util.xla_hlo_parser.xla_hlo_trace_parser as hlo_parser
import trace_util.xla_hlo_parser.xla_hlo_structures as hlo_struct


benchmark_names: list[str] = [
    'bert',
    'dlrm',
    'efficientnet',
    'mask-rcnn',
    'mnist',
    'ncf',
    'resnet',
    'resnet-rs',
    'retinanet',
    'shapemask',
    'transformer',
    'llama13b',
    'llava-llm-prefill',
    'llava-llm-decode',
    'llava-vit',
    'gligen',
    'seem',
]


inference_module_name: dict[str, str] = {
    'bert': 'cluster_test_step',
    'dlrm': 'cluster_predict_function',
    'efficientnet': 'cluster_predict_function',
    'mask-rcnn': 'cluster_test_step',
    'mnist': 'cluster_predict_function',
    'ncf': 'cluster_predict_function',
    'resnet': 'cluster_eval_step',
    'resnet-rs': 'cluster_eval_step',
    'retinanet': 'cluster_test_step',
    'shapemask': 'cluster_test_step',
    'transformer': 'cluster_predict_function',
    'llama13b': '',
    'llava-llm-prefill': '',
    'llava-llm-decode': '',
    'llava-vit': '',
    'gligen': '',
    'seem': '',
}


inference_batch_size: dict[str, list[int]] = {
    'bert':                 [1, 8, 32, 64, 128, 256, 512],
    'dlrm':                 [1, 8, 32, 64, 128, 256, 512, 1024],
    'efficientnet':         [1, 8, 32, 64, 128, 256, 512, 1024],
    'mask-rcnn':            [1, 8],
    'mnist':                [1, 8, 32, 64, 128, 256, 512, 1024],
    'ncf':                  [1, 8, 32, 64, 128, 256, 512, 1024],
    'resnet':               [1, 8, 32, 64, 128, 256, 512, 1024],
    'resnet-rs':            [1, 8, 32, 64, 128, 256, 512, 1024],
    'retinanet':            [1, 8, 32, 64],
    'shapemask':            [1, 8],
    'transformer':          [1, 8, 32, 64],
    'llama13b':             [1, 8, 32, 64, 128, 256, 512, 1024],
    'llava-llm-prefill':    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    'llava-llm-decode':     [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    'llava-vit':            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    'gligen':               [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    'seem':                 [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
}


def get_size_bytes_from_dtype(dtype: str) -> int:
    if "8" in dtype:
        return 1
    elif "16" in dtype:
        return 2
    elif "32" in dtype:
        return 4
    elif "64" in dtype:
        return 8
    elif "BOOL" in dtype:
        return 1
    elif "DT_FLOAT" == dtype:
        return 4
    elif "DT_INT" == dtype:
        return 4
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


@lru_cache(maxsize=None)
def get_factors(num: int) -> list[int]:
    '''return all positive factors of @num (in ascending order)'''
    factors: set[int] = set()
    for i in range(1, ceil(sqrt(num)) + 1):
        if num % i == 0:
            factors.update({i, num // i})
    return sorted(list(factors))


def get_hlo_module(workload_dir: str, bn: str, bs: int) -> hlo_struct.HLOModule:
    ### parse model from HLO source codes
    b = f"{bn}.{bs}"
    xla_hlo_dir = f"{workload_dir}/{bn}/xla_hlo_{bn}_{bs}"
    model = hlo_parser.parseHLOModel(b, xla_hlo_dir)

    # we only care about the prediction function
    modules = model.searchModuleByName(inference_module_name[b.split('.')[0]])
    assert len(modules) == 1 and inference_module_name[b.split('.')[0]] in modules[0].name, \
        f"benchmark '{b}': module search failed: model={b}, query={inference_module_name[b.split('.')[0]]}, \
            result={str(modules)}"
    module = modules[0]
    module.name = b
    return module


def construct_hlo_instruction_from_node_cost(node_cost: dict[str, Any] | Operator.Operator) -> hlo_struct.HLOInstruction:
    '''construct an HLOInstruction from @node_cost. This function is currently only used for tf-sim analytical LLM traces.'''
    if isinstance(node_cost, dict):
        # convert dict to Operator
        node_cost = Operator.from_csv_dict(node_cost)

    ### output tensor
    output_shape_str = node_cost.output_tensor_shape_str
    scalar_type, shape_str = output_shape_str.split(":")[0:2]
    shape_str = shape_str.removeprefix("(").removesuffix(")]")
    output_shape = [int(s) for s in shape_str.split(",")]
    output_value = hlo_struct.HLOValue(
        type=hlo_struct.HLOTensorType(
            scalar_type=scalar_type,
            shape=output_shape,
        ),
        name=node_cost.name,
    )

    ### input tensors
    inputs_shape_str = node_cost.input_tensor_shape_str
    input_shape_str_list = inputs_shape_str.removesuffix("]").split("],")
    input_values = []
    for shape_str in input_shape_str_list:
        scalar_type, shape_str = shape_str.split(":[")[0:2]
        input_shape = [int(s) for s in shape_str.split(",")]
        input_value = hlo_struct.HLOValue(
            type=hlo_struct.HLOTensorType(
                scalar_type=scalar_type,
                shape=input_shape,
            )
        )
        input_values.append(input_value)

    op_config = node_cost.config_str

    ### opcode
    opcode = op_config.split("(")[0]

    metadata = {
        "op_type": "Einsum" if opcode in ["XlaEinsum", "BatchMatMul"] else opcode,
    }
    if opcode == "Conv2D":
        metadata["dim_labels"] = op_config.split("eq=")[1].split(",")[0]
        metadata["window"] = op_config.split("window=")[1].split(",")[0]
    I = hlo_struct.HLOInstruction(
        result=output_value,
        operands=input_values,
        opcode=(
            "convolution"
            if opcode in ["XlaEinsum", "BatchMatMul", "Conv2D", "FlashAttention"]
            else opcode
        ),
        metadata=metadata,
        raw_string=str(node_cost), #json.dumps(node_cost, indent=4),
    )

    ### input_axes and output_axes
    if opcode in ["XlaEinsum"]:
        ### einsum dim labels
        dim_labels = op_config.split("eq=")[1].split(",")[0]
        in_out_dim_label_list = dim_labels.split("->")
        input_dim_label_list = in_out_dim_label_list[0].split(";")
        input0_str = input_dim_label_list[0]
        input1_str = input_dim_label_list[1]
        output_str = in_out_dim_label_list[1]

        I.input_axes = [[], []]
        I.output_axes = []

        # lhs
        for i, c in enumerate(input0_str):
            shape_type = I.operands[0].type
            assert isinstance(shape_type, hlo_struct.HLOTensorType)  # for pylint checking
            I.input_axes[0].append(hlo_struct.HLOAxis(c, i, shape_type.shape[i], shape_type.scalar_type))
        # rhs
        for i, c in enumerate(input1_str):
            shape_type = I.operands[1].type
            assert isinstance(shape_type, hlo_struct.HLOTensorType)
            I.input_axes[1].append(hlo_struct.HLOAxis(c, i, shape_type.shape[i], shape_type.scalar_type))
        # out
        for i, c in enumerate(output_str):
            shape_type = I.result.type
            assert isinstance(shape_type, hlo_struct.HLOTensorType)
            I.output_axes.append(hlo_struct.HLOAxis(c, i, shape_type.shape[i], shape_type.scalar_type))
    elif opcode == "BatchMatMul":
        I.input_axes = [[], []]
        I.output_axes = []

        input1_shape_type = I.operands[0].type
        assert isinstance(input1_shape_type, hlo_struct.HLOTensorType)
        input2_shape_type = I.operands[1].type
        assert isinstance(input2_shape_type, hlo_struct.HLOTensorType)
        output_shape_type = I.result.type
        assert isinstance(output_shape_type, hlo_struct.HLOTensorType)

        # batch
        I.input_axes[0].append(hlo_struct.HLOAxis("batch", 0, input1_shape_type.shape[0], input1_shape_type.scalar_type))
        I.input_axes[1].append(hlo_struct.HLOAxis("batch", 0, input2_shape_type.shape[0], input2_shape_type.scalar_type))
        I.output_axes.append(hlo_struct.HLOAxis("batch", 0, output_shape_type.shape[0], output_shape_type.scalar_type))

        # m
        I.input_axes[0].append(hlo_struct.HLOAxis("m", 1, input1_shape_type.shape[1], input1_shape_type.scalar_type))
        I.output_axes.append(hlo_struct.HLOAxis("m", 1, output_shape_type.shape[1], output_shape_type.scalar_type))

        # k
        I.input_axes[0].append(hlo_struct.HLOAxis("k", 2, input1_shape_type.shape[2], input1_shape_type.scalar_type))
        I.input_axes[1].append(hlo_struct.HLOAxis("k", 1, input2_shape_type.shape[1], input2_shape_type.scalar_type))

        # n
        I.input_axes[1].append(hlo_struct.HLOAxis("n", 2, input2_shape_type.shape[2], input2_shape_type.scalar_type))
        I.output_axes.append(hlo_struct.HLOAxis("n", 2, output_shape_type.shape[2], output_shape_type.scalar_type))

    return I


def construct_hlo_module_from_node_costs(node_costs: Sequence[dict[str, Any] | Operator.Operator], name: str = "") -> hlo_struct.HLOModule:
    '''construct an HLOModule from @node_costs. This module contains a single ENTRY "forward" function.'''
    module = hlo_struct.HLOModule(name)
    module.addHLOFunction(hlo_struct.HLOFunction("forward"))
    for node_cost in node_costs:
        I = construct_hlo_instruction_from_node_cost(node_cost)
        module.ENTRY.instructions.append(I)
    return module


# def construct_hlo_module_from_ops(ops: Sequence[Operator.Operator]) -> hlo_struct.HLOModule:
#     '''construct an HLOModule from @ops. This module contains a single ENTRY "forward" function.'''
#     ops_dict = [Operator.to_csv_dict(op) for op in ops]
#     module = construct_hlo_module_from_node_costs(ops_dict)
#     return module


## TODO: unused for now. Fix it when needed.
# def get_xla_program(tfsim_dir: str, bn: str, bs: int) -> XLAProgram:
#     b = f"{bn}.{bs}"
#     prog = XLAProgram(b)
#     prog.loadTraceFromFile(
#         os.path.join(tfsim_dir, f"xla_hlo_{bn}_{bs}"),
#         npu_config_DEFAULT,
#         analyze_only=True,
#     )
#     return prog


def get_tfsim_node_costs(tfsim_dir: str, bn: str, bs: int, sa: int = 4, vu: int = 1) -> list[dict[str, Any]]:
    if bn in ["llama13b"]:  # llama2-13b from tf-sim analytical
        node_cost_csv_path = os.path.join(tfsim_dir, f"xla_hlo_{bn}_{bs}", f"sa{sa}_vu{vu}", "JellyFish-TPU_Conv-Opt-4SA-4VU-LLaMA-13B-serving-fwd_bwd_ops.csv")
    # elif bn in ["clip-vit", "vicuna13b", "seem", "lama", "gligen"]:  # multimodal benchmarks
    #     node_cost_csv_path = os.path.join(tfsim_dir, f"xla_hlo_{bn}_{bs}", f"sa{sa}_vu{vu}", "cluster*", "node_costs.csv")
    else:  # other benchmarks from tf-sim
        node_cost_csv_path = os.path.join(tfsim_dir, f"xla_hlo_{bn}_{bs}", f"sa{sa}_vu{vu}", "cluster*", "node_costs.csv")
    nc_glob = glob.glob(node_cost_csv_path)
    assert len(nc_glob) == 1, f"benchmark '{bn}.{bs}': node_costs.csv not found in directory {node_cost_csv_path}"
    node_costs_file_path = nc_glob[0]
    with open(node_costs_file_path, "r") as f:
        reader = csv.DictReader(f)
        node_costs = list(reader)

    # allocate new field names
    def init_new_field(field_name, default_value = None):
        if field_name not in nc:
            nc[field_name] = default_value
    for nc in node_costs:
        init_new_field("parsed_op_type")
        init_new_field("dim_labels")
        init_new_field("tile_shapes")
        init_new_field("num_tiles")
        init_new_field("max_vmem_demand_bytes")
        init_new_field("num_mxu_ops", 0)
        init_new_field("einsum_B_size")
        init_new_field("einsum_M_size")
        init_new_field("einsum_N_size")
        init_new_field("einsum_K_size")

    return node_costs


def get_top_level_node_op_name(node: dict[str, Any]) -> str:
    '''return the op name of the top level node of @node'''
    if node["Top Level Node"] == "True":
        return node["Op Name"]

    name_dir = str(node["Op Name"]).split("/")
    tln_name = name_dir[0]

    # hack for While node in transformer model
    if "while" in tln_name:
        return "While"

    return tln_name


def get_top_level_node(node: dict[str, Any], node_costs: list[dict[str, Any]]) -> dict[str, Any]:
    '''return the top level node of @node'''
    if node["Top Level Node"] == "True":
        return node

    tln_name = get_top_level_node_op_name(node)
    for nc in node_costs:
        if nc["Op Name"] == tln_name:
            return nc
    raise ValueError(f"Top level node not found for node: {node}")


def get_total_execution_time_ns_from_tfsim_node_costs(node_costs: list[dict[str, Any]]) -> int:
    '''return the total execution time in ns from @node_costs'''
    return sum([
        int(node["Total execution time (ns)"]) for node in node_costs if node["Top Level Node"] == "True"
    ])


def get_total_execution_time_ns_from_ops(node_costs: list[Operator.Operator]) -> int:
    '''return the total execution time in ns from @node_costs'''
    return sum([
        int(node.stats.execution_time_ns) for node in node_costs
    ])


def get_total_mxu_vpu_time_ns_from_tfsim_node_costs(node_costs: list[dict[str, Any]]) -> tuple[int, int]:
    '''return the total MXU and VPU time in ns from @node_costs'''
    mxu_time = sum([
        int(node["Total MXU time (ns)"]) for node in node_costs if node["Top Level Node"] == "True"
    ])
    vpu_time = sum([
        int(node["Total VPU time (ns)"]) for node in node_costs if node["Top Level Node"] == "True"
    ])
    return mxu_time, vpu_time


def get_total_compute_time_ns_from_tfsim_node_costs(node_costs: list[dict[str, Any]]) -> int:
    '''return the total compute time in ns from @node_costs'''
    return sum([
        int(node["Compute Time (ns)"]) for node in node_costs if node["Top Level Node"] == "True"
    ])


def get_total_memory_time_ns_from_tfsim_node_costs(node_costs: list[dict[str, Any]]) -> int:
    '''return the total memory time in ns from @node_costs'''
    return sum([
        int(node["Memory Time (ns)"]) for node in node_costs if node["Top Level Node"] == "True"
    ])


def get_total_bytes_accessed_from_tfsim_node_costs(node_costs: list[dict[str, Any]]) -> int:
    '''return the total bytes accessed from @node_costs'''
    return sum([
        int(node["Bytes Accessed"]) for node in node_costs if node["Top Level Node"] == "True"
    ])


def get_total_flops_from_tfsim_node_costs(node_costs: list[dict[str, Any]]) -> int:
    '''return the total FLOPs from @node_costs'''
    return sum([
        int(node["FLOP Count"]) for node in node_costs if node["Top Level Node"] == "True"
    ])


def get_num_bound_by_from_tfsim_node_costs(node_costs: list[dict[str, Any]], bound_by: str) -> int:
    '''return the number of nodes bound by @bound_by from @node_costs'''
    return sum([
        1 for node in node_costs if node["Top Level Node"] == "True" and node["Bound By"] == bound_by and int(node["Total execution time (ns)"]) > 0
    ])


def get_num_top_level_nodes_from_tfsim_node_costs(node_costs: list[dict[str, Any]]) -> int:
    '''return the number of top level nodes from @node_costs'''
    return sum([
        1 for node in node_costs if node["Top Level Node"] == "True" and int(node["Total execution time (ns)"]) > 0
    ])

def get_ici_traffic_from_tfsim_node_cost(node_costs: list[dict[str, Any]]) -> tuple[int, int, float]:
    '''return the total number of bytes sent/received and total ICI time (@max bw) from node costs.'''
    outbound_bytes = sum([
        nc["ICI/NVLink outbound traffic"] for nc in node_costs if nc["Top Level Node"] == "True"
    ])
    inbound_bytes = sum([
        nc["ICI/NVLink inbound traffic"] for nc in node_costs if nc["Top Level Node"] == "True"
    ])
    ici_time_total = sum([
        nc["ICI/NVLink time"] for nc in node_costs if nc["Top Level Node"] == "True"
    ])
    return outbound_bytes, inbound_bytes, ici_time_total


def get_model_weight_size(  node_costs: list[dict[str, Any]],
                            model: str = "llama") -> int:

    '''
    Compute sum of model weights listed in ops generator. For certain models, some adjustments may
    need to be made outside of this function.
    '''
    if model in ["llama", "llama-prefill", "vit"]:
        return sum([
            nc["Weight Size"] for nc in node_costs if (nc["Top Level Node"] == "True" and "decode" not in nc["Op Name"])
            ])
    elif model in ["llama-decode"]:
        return sum([
            nc["Weight Size"] for nc in node_costs if (nc["Top Level Node"] == "True")
            ])
    elif model in ["gligen", "seem"]:
        return sum([
            nc["Weight Size"] for nc in node_costs if (nc["Top Level Node"] == "True")
            ])
    else:
        raise NotImplementedError("Can't compute model weights for this model!")
