### Helper library for running a single operator simulation

from trace_util.llm_ops_generator.Operator import Operator
from trace_util.llm_ops_generator import op_analysis_lib as analysis_lib
from trace_util.llm_ops_generator.configs.chips.ChipConfig import ChipConfig


def run_sim_single_op(op: Operator, cfg: ChipConfig) -> Operator:
    ops = analysis_lib.fill_operators_execution_info(
        [op], cfg
    )
    return ops[0]
