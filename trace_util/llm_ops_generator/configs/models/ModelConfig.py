from pydantic.dataclasses import dataclass
from pydantic import TypeAdapter

from trace_util.llm_ops_generator.configs.chips.ChipConfig import ChipConfig
from trace_util.llm_ops_generator.configs.systems.SystemConfig import SystemConfig


# @dataclass(kw_only=True)
class ModelConfig(ChipConfig, SystemConfig):
    model_type: str
    global_batch_size: int = 1

    num_chips: int = 1
    data_parallelism_degree: int = 1
    tensor_parallelism_degree: int = 1
    pipeline_parallelism_degree: int = 1
    num_data_parallel_axes: int = 1
    num_tensor_parallel_axes: int = 1
    num_pipeline_parallel_axes: int = 1
    data_parallel_degree_dcn: int = 1
    tensor_parallel_degree_dcn: int = 1
    pipeline_parallel_degree_dcn: int = 1
    microbatch_size_dcn: int = 1
    microbatch_size_ici: int = 1

    output_file_path: str
