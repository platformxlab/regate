### Helper functions for triggering the performance simulation of operators.
### fill_operators_execution_info() can be called on a list of Operator objects
### to simulate their execution times.

from copy import deepcopy
from typing import Any

from absl import flags, logging
import numpy as np

from trace_util.llm_ops_generator.Operator import Operator, OpType
from trace_util.npusim_backend import npusim_lib
from trace_util.npusim_backend import util as npusim_util
from trace_util.llm_ops_generator.configs.chips.ChipConfig import ChipConfig


def analyze_operator_component_util(
    op: Operator, config: ChipConfig
) -> Operator:
    '''
    Fill out op.stats.flops_util and op.stats.hbm_bw_util.
    '''
    if op.op_type == OpType.MXU:
        op.stats.flops_util = op.stats.tflops_per_sec / config.peak_tflops_per_sec
    else:
        op.stats.flops_util = op.stats.tflops_per_sec / config.peak_VU_tflops_per_sec

    op.stats.hbm_bw_util = op.stats.hbm_bw_GBps / config.hbm_bw_GBps

    return op


def fill_operators_execution_info(
    ops: list[Operator],
    config: ChipConfig,
) -> list[Operator]:
    '''
    Fill in the execution info (exe time, flops, bytes accessed, etc.) for each op.
    '''
    converted_ops = []

    # hlo_module = mem_util.construct_hlo_module_from_node_costs(node_costs)
    hlo_module = npusim_util.construct_hlo_module_from_node_costs(ops)

    for op in ops:
        I, converted_op = npusim_lib.parse_tensor_shapes_for_node_cost(op, hlo_module)

        mxu_time, vpu_time = npusim_lib.compute_node_cost_compute_time(
            I, converted_op, config # num_sa, num_vu, freq_GHz
        )
        bytes_accessed = npusim_lib.compute_bytes_accessed_from_vmem_size(
            I, converted_op, config # vmem_size_mb, hbm_bw_GBps, freq_GHz
        )
        compute_time = max(mxu_time, vpu_time)
        memory_time = max(
            int(np.ceil(bytes_accessed / (config.hbm_bw_GBps * 1024 * 1024 * 1024 / 1e9))),
            config.hbm_latency_ns,
        )
        if bytes_accessed == 0:
            memory_time = 0
        ici_time = converted_op.stats.ici_time_ns
        exe_time = max(compute_time, ici_time, memory_time)
        converted_op.stats.sa_time_ns = mxu_time
        converted_op.stats.vu_time_ns = vpu_time
        converted_op.stats.compute_time_ns = compute_time
        converted_op.stats.memory_time_ns = memory_time
        converted_op.stats.execution_time_ns = exe_time
        converted_op.stats.memory_traffic_bytes = bytes_accessed
        if compute_time == exe_time:
            converted_op.stats.bounded_by = "Compute"
        elif memory_time == exe_time and compute_time < exe_time:
            converted_op.stats.bounded_by = "Memory"
        else:
            converted_op.stats.bounded_by = "ICI/NVLink"
        # fill_additional_fields(converted_op)

        analyze_operator_component_util(converted_op, config)

        converted_ops.append(converted_op)

    return converted_ops
