### class Operator:
###     Represents a single operator (e.g., Softmax, MatMul).
###     Contains fields for operator semantics (e.g., input/output shapes, operator type),
###     as well as fields for simulation results (e.g., execution time, memory traffic).
### In addition, this module contains utility functions for serializing and deserializing
### operators to/from python dictionaries, which can be used for dumping to CSV files.


from typing import Any, Sequence
from enum import Enum
import uuid

import numpy as np
from pydantic import BaseModel


def get_base_op_dict() -> dict[str, Any]:
    return {
        "Fusion index": 0,
        "Description": "",
        "Config": "",
        "Name": "",
        "OpType": "",
        "Count": 0,
        "Bounded-by": "",
        "Execution time": 0,
        "Compute time": 0,
        "Memory time": 0,
        "ICI/NVLink time": 0,
        "ICI/NVLink outbound traffic": 0,
        "ICI/NVLink inbound traffic": 0,
        "Aggregated DCN time": 0,
        "DCN 0 time": 0,
        "PCIe time": 0,
        "Temporary memory size": 0,
        "Persistent memory size": 0,
        "MXU time": 0,
        "VPU time": 0,
        "Transpose time": 0,
        "Permute time": 0,
        "Bytes accessed": 0,
        "Input Tensor Shapes": "",
        "Output Tensor Shapes": "",
        "FLOP Count": 0,
        "Op Name": "",
        "Op Code": "",
        "Weight Size": 0,      # HBM capacity needed to store parameters in bytes

        # analysis fields
        # (may not exist in some ops)
        "Output Shape": [],
        "parsed_op_type": "",
        "dim_labels": "",
        "tile_shapes": "",
        "num_tiles": "",
        "max_vmem_demand_bytes": "",
        "num_mxu_ops": 0,
        "einsum_B_size": "",
        "einsum_M_size": "",
        "einsum_N_size": "",
        "einsum_K_size": "",

        # tf-sim fields
        # (may be repetitive with the above fields)
        "Compute Time (ns)": 0,
        "Bytes Accessed": 0,
    }


class OpcodeType(Enum):
    CONV2D = "Conv2D"
    EINSUM = "Einsum"
    FLASH_ATTENTION = "FlashAttention"
    ELEMENTWISE = "Elementwise"
    UP_DOWN_SAMPLE = "UpDownSample"
    COLLECTIVE_REDUCE = "CollectiveReduce"
    COLLECTIVE_NO_COMPUTE = "CollectiveNoCompute"
    EMBEDDING_BAG = "EmbeddingBag"
    OTHER = "Other"

    @classmethod
    def from_opcode(cls, opcode: str):
        '''
        Converts an opcode to an OpType.
        '''
        if opcode == "Conv2D":
            return cls.CONV2D
        elif opcode in ["Einsum", "MatMul", "BatchMatMulV2"]:
            return cls.EINSUM
        elif opcode == "FlashAttention":
            return cls.FLASH_ATTENTION
        elif opcode in ["LayerNorm", "GroupNorm", "RMSNorm", "Softmax", "Add", "Mul", "Abs", "Pointwise Mul."]:
            return cls.ELEMENTWISE
        elif opcode in ["Upsample", "AvgPool2d"]:
            return cls.UP_DOWN_SAMPLE
        elif opcode in ["ReduceScatter", "AllReduce"]:
            return cls.COLLECTIVE_REDUCE
        elif opcode in ["AllGather", "InterChipCommInput", "InterChipCommOutput", "AllToAll"]:
            return cls.COLLECTIVE_NO_COMPUTE
        elif opcode in ["EmbeddingBag"]:
            return cls.EMBEDDING_BAG
        else:
            return cls.OTHER


class OpType(Enum):
    MXU = "MXU"
    VPU = "VPU"
    ICI_NO_COMPUTE = "ICINoCompute"
    OTHER = "Other"


    @classmethod
    def from_string(cls, op_type: str):
        '''
        Converts a string to an OpType.
        '''
        if op_type == "MXU":
            return cls.MXU
        elif op_type == "VPU":
            return cls.VPU
        elif op_type == "ICINoCompute":
            return cls.ICI_NO_COMPUTE
        else:
            return cls.OTHER


class Axis(BaseModel):
    '''
    Represents a single axis.
    '''

    name: str
    '''Name'''
    size: int
    '''Size'''
    index: int
    '''Index'''
    parallelism: list[int]
    '''Parallelism degrees. Default is 1 (no parallelism).'''
    tile_size: int
    '''Tile Size (not tiled by default)'''

    def __init__(
        self,
        name: str = "axis",
        size: int = 1,
        index: int = 0,
        parallelism: Sequence[int] | None = None,
        tile_size: int | None = None,
    ) -> None:
        parallelism = list(parallelism) if parallelism else [1]
        tile_size = size if tile_size == None else tile_size
        super().__init__(
            name=name,
            size=size,
            index=index,
            parallelism=parallelism,
            tile_size=tile_size,
        )

    @property
    def num_shards(self) -> int:
        '''
        Returns the number of shards.
        '''
        return int(np.prod(self.parallelism))

    @property
    def shard_size(self) -> int:
        '''
        Returns the local shard size.
        '''
        return int(np.ceil(self.size / self.num_shards))

    @property
    def num_tiles(self) -> int:
        '''
        Returns the number of tiles (for the local shard).
        '''
        return int(np.ceil(self.shard_size / self.tile_size))


class Tensor(BaseModel):
    '''
    Represents a single tensor.
    '''

    name: str
    '''Name'''
    axes: list[Axis]
    '''Axes'''
    dtype: str
    '''Data Type'''
    uuid: str
    '''Auto-generated UUID'''

    def __init__(
        self,
        name: str = "tensor",
        axes: list[Axis] | None = None,
        dtype: str = "BFLOAT16",
    ) -> None:
        axes = axes or [Axis()]
        _uuid = uuid.uuid4().hex
        super().__init__(
            name=name,
            axes=axes,
            dtype=dtype,
            uuid=_uuid,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        '''
        Returns the shape of the tensor.
        '''
        return tuple(axis.size for axis in self.axes)


    @classmethod
    def from_shape(cls, name: str, shape: Sequence[int], dtype: str = "BFLOAT16") -> "Tensor":
        '''
        Creates a Tensor from a shape.
        '''
        axes = [Axis(name=f"{i}", size=size, index=i) for i, size in enumerate(shape)]
        return cls(name=name, axes=axes, dtype=dtype)


class OperatorStatistics(BaseModel):
    '''
    Contains simulation/analysis statistics for a single operator.
    '''
    count: int = 1
    '''Count'''
    bounded_by: str = ""
    '''Bounded-by'''
    execution_time_ns: int = 0
    '''Execution time'''
    compute_time_ns: int = 0
    '''Compute time, Compute Time'''
    memory_time_ns: int = 0
    '''Memory time'''
    ici_time_ns: int = 0
    '''ICI/NVLink time'''
    sa_time_ns: int = 0
    '''MXU time'''
    vu_time_ns: int = 0
    '''VPU time'''
    vmem_time_ns: int = 0
    '''Vmem time (for now, this is only used for energy analysis)'''
    transpose_time_ns: int = 0
    '''Transpose time (unused)'''
    permute_time_ns: int = 0
    '''Permute time (unused)'''
    memory_traffic_bytes: int = 0
    '''Bytes accessed, Bytes Accessed'''
    ici_traffic_outbound_bytes: int = 0
    '''ICI/NVLink outbound traffic'''
    ici_traffic_inbound_bytes: int = 0
    '''ICI/NVLink inbound traffic'''
    dcn_time_ns: int = 0
    '''Aggregated DCN time'''
    pcie_time_ns: int = 0
    '''PCIe time (unused)'''

    flop_count: int = 0
    '''FLOP Count'''
    weight_size_bytes: int = 0
    '''Weight Size'''

    parsed_op_type: str = ""
    '''Parsed_op_type'''

    # power stats
    static_energy_sa_J: float = 0
    '''Static energy consumption of SA in Joules'''
    static_energy_vu_J: float = 0
    '''Static energy consumption of VU in Joules'''
    static_energy_sram_J: float = 0
    '''Static energy consumption of SRAM (Vmem) in Joules'''
    static_energy_hbm_J: float = 0
    '''Static energy consumption of HBM controller+PHY in Joules'''
    static_energy_ici_J: float = 0
    '''Static energy consumption of ICI/NVLink in Joules'''
    static_energy_other_J: float = 0
    '''Static energy consumption of other components in Joules'''

    dynamic_energy_sa_J: float = 0
    '''Dynamic energy consumption of SA in Joules'''
    dynamic_energy_vu_J: float = 0
    '''Dynamic energy consumption of VU in Joules'''
    dynamic_energy_sram_J: float = 0
    '''Dynamic energy consumption of SRAM (Vmem) in Joules'''
    dynamic_energy_hbm_J: float = 0
    '''Dynamic energy consumption of HBM controller+PHY+data transfer in Joules'''
    dynamic_energy_ici_J: float = 0
    '''Dynamic energy consumption of ICI/NVLink in Joules'''
    dynamic_energy_other_J: float = 0
    '''Dynamic energy consumption of other components in Joules'''

    # power gating stats
    num_setpm_sa: int = 0
    '''Number of setpm calls on SAs'''
    num_setpm_vu: int = 0
    '''Number of setpm calls on VUs'''
    num_setpm_sram: int = 0
    '''Number of setpm calls on SRAM (Vmem)'''
    num_setpm_hbm: int = 0
    '''Number of setpm calls on HBM controller+PHY'''
    num_setpm_ici: int = 0
    '''Number of setpm calls on ICI/NVLink'''

    # utilization stats
    flops_util: float = 0
    '''
    FLOPS Utilization (0-1), normalized to SA/VU type.
    E.g., for SA op, this is self.tflops_per_sec / peak SA tflops;
    for VU op, this is self.tflops_per_sec / peak VU tflops.
    '''
    hbm_bw_util: float = 0
    '''
    HBM BW Utilization (0-1).
    '''

    @property
    def ici_traffic_bytes(self) -> int:
        '''
        Returns the total ICI/NVLink traffic bytes (outbound + inbound).
        '''
        return self.ici_traffic_outbound_bytes + self.ici_traffic_inbound_bytes

    @property
    def tflops_per_sec(self) -> float:
        '''
        Returns the TFLOPs per second of this operator.
        '''
        if self.execution_time_ns == 0:
            return 0
        return self.flop_count * 1e9 / self.execution_time_ns / 1e12

    @property
    def hbm_bw_GBps(self) -> float:
        '''
        Returns the HBM bandwidth utilization in GBps.
        '''
        if self.execution_time_ns == 0:
            return 0    
        return self.memory_traffic_bytes / 1024 / 1024 / 1024 * 1e9 / self.execution_time_ns

    @property
    def static_energy_J(self) -> float:
        '''
        Returns the total static energy consumption in Joules.
        '''
        return (
            self.static_energy_sa_J
            + self.static_energy_vu_J
            + self.static_energy_sram_J
            + self.static_energy_hbm_J
            + self.static_energy_ici_J
            + self.static_energy_other_J
        )

    @property
    def static_power_W(self) -> float:
        '''
        Returns the average static power consumption in Watts.
        '''
        if self.execution_time_ns == 0:
            return -1
        return self.static_energy_J / self.execution_time_ns * 1e9

    @property
    def dynamic_energy_J(self) -> float:
        '''
        Returns the total dynamic energy consumption in Joules.
        '''
        return (
            self.dynamic_energy_sa_J
            + self.dynamic_energy_vu_J
            + self.dynamic_energy_sram_J
            + self.dynamic_energy_hbm_J
            + self.dynamic_energy_ici_J
            + self.dynamic_energy_other_J
        )

    @property
    def dynamic_power_W(self) -> float:
        '''
        Returns the average dynamic power consumption in Watts.
        '''
        if self.execution_time_ns == 0:
            return -1
        return self.dynamic_energy_J / self.execution_time_ns * 1e9

    @property
    def total_energy_J(self) -> float:
        '''
        Returns the total energy consumption in Joules.
        '''
        return self.static_energy_J + self.dynamic_energy_J

    @property
    def total_power_W(self) -> float:
        '''
        Returns the average total power consumption in Watts.
        '''
        if self.execution_time_ns == 0:
            return -1
        return self.total_energy_J / self.execution_time_ns * 1e9


class EinsumStatistics(OperatorStatistics):
    '''
    Statistics for einsum operators.
    '''

    dim_labels_str: str = ""
    '''Dim_labels'''
    tile_shapes_str: str = ""
    '''Tile_shapes'''
    num_tiles: int = 1
    '''Num_tiles'''
    max_vmem_demand_bytes: int = 0
    '''Max_vmem_demand_bytes'''
    num_sa_ops: int = 0
    '''Num_mxu_ops'''
    einsum_B_size: int = 0
    '''Einsum_B_size'''
    einsum_M_size: int = 0
    '''Einsum_M_size'''
    einsum_N_size: int = 0
    '''Einsum_N_size'''
    einsum_K_size: int = 0
    '''Einsum_K_size'''


class FlashAttentionStatistics(OperatorStatistics):
    '''
    Statistics for flash attention operators.
    '''

    tile_shapes_str: str = ""
    '''Tile_shapes'''
    max_vmem_demand_bytes: int = 0
    '''Max_vmem_demand_bytes'''
    num_sa_ops: int = 0
    '''Num_mxu_ops'''
    einsum_B_size: int = 0
    '''Einsum_B_size'''
    einsum_M_size: int = 0
    '''Einsum_M_size'''
    einsum_N_size: int = 0
    '''Einsum_N_size'''
    einsum_K_size: int = 0
    '''Einsum_K_size'''


class Operator(BaseModel):
    '''
    Contains semantic information of an operator.
    '''

    stats: OperatorStatistics = OperatorStatistics()
    '''Statistics for the operator'''

    fusion_id: int = 0
    '''Fusion index'''
    description: str = ""
    '''Description'''
    name: str = ""
    '''Name'''
    op_type: OpType = OpType.OTHER
    '''OpType (MXU or VPU or ICI_NO_COMPUTE)'''
    opcode_type: OpcodeType = OpcodeType.OTHER
    '''Opcode Type'''
    opcode: str = ""

    input_tensors: list[Tensor] = []
    '''Input Tensors'''
    output_tensors: list[Tensor] = []
    '''Output Tensors'''

    # temporary fields for compatibility with csv traces
    input_tensor_shape_str: str = ""
    output_tensor_shape_str: str = ""
    config_str: str = ""

    def to_csv_dict(self) -> dict[str, Any]:
        '''
        Adapter for outputing CSV traces that are compatible
        with TF-Sim and TF-Sim-Analytical.
        '''
        op_dict = get_base_op_dict()
        op_dict.update({
            "Fusion index": self.fusion_id,
            "Description": self.description,
            "Config": self.config_str,
            "Name": self.name,
            "OpType": self.op_type.value,
            "Count": self.stats.count,
            "Bounded-by": self.stats.bounded_by,
            "Execution time": self.stats.execution_time_ns,
            "Compute time": self.stats.compute_time_ns,
            "Memory time": self.stats.memory_time_ns,
            "Vmem time": self.stats.vmem_time_ns,
            "ICI/NVLink time": self.stats.ici_time_ns,
            "ICI/NVLink outbound traffic": self.stats.ici_traffic_outbound_bytes,
            "ICI/NVLink inbound traffic": self.stats.ici_traffic_inbound_bytes,
            "Aggregated DCN time": self.stats.dcn_time_ns,
            "DCN 0 time": self.stats.dcn_time_ns,
            "PCIe time": self.stats.pcie_time_ns,
            "MXU time": self.stats.sa_time_ns,
            "VPU time": self.stats.vu_time_ns,
            "Transpose time": self.stats.transpose_time_ns,
            "Permute time": self.stats.permute_time_ns,
            "Bytes accessed": self.stats.memory_traffic_bytes,
            "Input Tensor Shapes": self.input_tensor_shape_str,
            "Output Tensor Shapes": self.output_tensor_shape_str,
            "FLOP Count": self.stats.flop_count,
            "Op Name": self.name,
            "Op Code": self.opcode,
            "Weight Size": self.stats.weight_size_bytes, # HBM capacity needed to store parameters in bytes
            "parsed_op_type": self.stats.parsed_op_type,

            # tf-sim fields
            # (may be repetitive with the above fields)
            "Compute Time (ns)": self.stats.compute_time_ns,
            "Bytes Accessed": self.stats.memory_traffic_bytes,

            # energy/power stats
            "static_energy_sa_J": self.stats.static_energy_sa_J,
            "static_energy_vu_J": self.stats.static_energy_vu_J,
            "static_energy_sram_J": self.stats.static_energy_sram_J,
            "static_energy_hbm_J": self.stats.static_energy_hbm_J,
            "static_energy_ici_J": self.stats.static_energy_ici_J,
            "static_energy_other_J": self.stats.static_energy_other_J,
            "dynamic_energy_sa_J": self.stats.dynamic_energy_sa_J,
            "dynamic_energy_vu_J": self.stats.dynamic_energy_vu_J,
            "dynamic_energy_sram_J": self.stats.dynamic_energy_sram_J,
            "dynamic_energy_hbm_J": self.stats.dynamic_energy_hbm_J,
            "dynamic_energy_ici_J": self.stats.dynamic_energy_ici_J,
            "dynamic_energy_other_J": self.stats.dynamic_energy_other_J,
            "static_energy_J": self.stats.static_energy_J,
            "dynamic_energy_J": self.stats.dynamic_energy_J,
            "total_energy_J": self.stats.total_energy_J,
            "static_power_W": self.stats.static_power_W,
            "dynamic_power_W": self.stats.dynamic_power_W,
            "total_power_W": self.stats.total_power_W,

            # power gating stats
            "num_setpm_sa": self.stats.num_setpm_sa,
            "num_setpm_vu": self.stats.num_setpm_vu,
            "num_setpm_sram": self.stats.num_setpm_sram,
            "num_setpm_hbm": self.stats.num_setpm_hbm,
            "num_setpm_ici": self.stats.num_setpm_ici,

            # util stats
            "tflops_per_sec": self.stats.tflops_per_sec,
            "hbm_bw_GBps": self.stats.hbm_bw_GBps,
            "flops_util": self.stats.flops_util,
            "hbm_bw_util": self.stats.hbm_bw_util,
        })
        return op_dict

    @classmethod
    def from_csv_dict(cls, opdict: dict[str, Any]) -> "Operator":
        '''
        Adapter for reading CSV traces that are compatible
        with TF-Sim and TF-Sim-Analytical.
        '''
        op = cls()

        op.fusion_id = int(opdict["Fusion index"])
        op.description = opdict["Description"]
        op.name = opdict["Name"]
        op.config_str = opdict["Config"]

        op.stats.count = int(opdict["Count"])
        op.stats.bounded_by = opdict["Bounded-by"]
        op.stats.execution_time_ns = int(opdict["Execution time"])
        op.stats.compute_time_ns = int(opdict["Compute time"])
        op.stats.memory_time_ns = int(opdict["Memory time"])
        op.stats.vmem_time_ns = int(opdict.get("Vmem time", 0))
        op.stats.ici_time_ns = int(opdict["ICI/NVLink time"])
        op.stats.ici_traffic_outbound_bytes = int(opdict["ICI/NVLink outbound traffic"])
        op.stats.ici_traffic_inbound_bytes = int(opdict["ICI/NVLink inbound traffic"])
        op.stats.dcn_time_ns = int(opdict["Aggregated DCN time"])
        op.stats.pcie_time_ns = int(opdict["PCIe time"])
        op.stats.sa_time_ns = int(opdict["MXU time"])
        op.stats.vu_time_ns = int(opdict["VPU time"])
        op.stats.transpose_time_ns = int(opdict["Transpose time"])
        op.stats.permute_time_ns = int(opdict["Permute time"])
        op.stats.memory_traffic_bytes = int(opdict["Bytes accessed"])
        op.stats.flop_count = int(opdict["FLOP Count"])
        op.stats.weight_size_bytes = int(opdict["Weight Size"])
        op.stats.flops_util = float(opdict.get("flops_util", 0))
        op.stats.hbm_bw_util = float(opdict.get("hbm_bw_util", 0))

        op.stats.parsed_op_type = opdict["parsed_op_type"]
        op.opcode = opdict["Op Code"]
        opcode_type_parsed = OpcodeType.from_opcode(opdict["parsed_op_type"])
        opcode_type_native = OpcodeType.from_opcode(opdict["Op Code"])
        opcode_type = opcode_type_parsed
        if opcode_type == OpcodeType.OTHER:
            opcode_type = opcode_type_native
        op.opcode_type = opcode_type

        op.op_type = OpType.from_string(opdict["OpType"])
        # override op type based on execution times
        if op.stats.sa_time_ns > 0:
            op.op_type = OpType.MXU
        elif op.stats.vu_time_ns > 0:
            op.op_type = OpType.VPU

        # TODO: parse input/output tensor shapes
        # ignored for now since we do not need them for energy analysis
        # op.input_tensors
        # op.output_tensors
        op.input_tensor_shape_str = opdict["Input Tensor Shapes"]
        op.output_tensor_shape_str = opdict["Output Tensor Shapes"]

        return op


class EinsumOperator(Operator):
    stats: EinsumStatistics = EinsumStatistics()  # type: ignore

    op_type: OpType = OpType.MXU
    opcode_type: OpcodeType = OpcodeType.EINSUM

    @classmethod
    def from_csv_dict(cls, opdict: dict[str, Any]) -> "EinsumOperator":
        '''
        Adapter for reading CSV traces that are compatible
        with TF-Sim and TF-Sim-Analytical.
        '''
        op = super(EinsumOperator, cls).from_csv_dict(opdict)
        op = cls(**op.model_dump())

        op.stats.dim_labels_str = opdict["dim_labels"]
        op.stats.tile_shapes_str = opdict["tile_shapes"]
        op.stats.num_tiles = int(opdict["num_tiles"])
        op.stats.max_vmem_demand_bytes = int(opdict["max_vmem_demand_bytes"])
        op.stats.num_sa_ops = int(opdict["num_mxu_ops"])
        op.stats.einsum_B_size = int(opdict["einsum_B_size"])
        op.stats.einsum_M_size = int(opdict["einsum_M_size"])
        op.stats.einsum_N_size = int(opdict["einsum_N_size"])
        op.stats.einsum_K_size = int(opdict["einsum_K_size"])

        return op

    def to_csv_dict(self) -> dict[str, Any]:
        opdict = super().to_csv_dict()
        opdict.update({
            "parsed_op_type": self.stats.parsed_op_type,
            "dim_labels": self.stats.dim_labels_str,
            "tile_shapes": self.stats.tile_shapes_str,
            "num_tiles": self.stats.num_tiles,
            "max_vmem_demand_bytes": self.stats.max_vmem_demand_bytes,
            "num_mxu_ops": self.stats.num_sa_ops,
            "einsum_B_size": self.stats.einsum_B_size,
            "einsum_M_size": self.stats.einsum_M_size,
            "einsum_N_size": self.stats.einsum_N_size,
            "einsum_K_size": self.stats.einsum_K_size,
        })
        return opdict


class Conv2DOperator(Operator):
    stats: EinsumStatistics = EinsumStatistics()  # type: ignore

    op_type: OpType = OpType.MXU
    opcode_type: OpcodeType = OpcodeType.CONV2D

    @classmethod
    def from_csv_dict(cls, opdict: dict[str, Any]) -> "Conv2DOperator":
        '''
        Adapter for reading CSV traces that are compatible
        with TF-Sim and TF-Sim-Analytical.
        Currently, this function is the same as EinsumOperator.from_csv_dict.
        '''
        op = super(Conv2DOperator, cls).from_csv_dict(opdict)
        op = cls(**op.model_dump())

        op.stats.dim_labels_str = opdict["dim_labels"]
        op.stats.tile_shapes_str = opdict["tile_shapes"]
        op.stats.num_tiles = int(opdict["num_tiles"])
        op.stats.max_vmem_demand_bytes = int(opdict["max_vmem_demand_bytes"])
        op.stats.num_sa_ops = int(opdict["num_mxu_ops"])
        op.stats.einsum_B_size = int(opdict["einsum_B_size"])
        op.stats.einsum_M_size = int(opdict["einsum_M_size"])
        op.stats.einsum_N_size = int(opdict["einsum_N_size"])
        op.stats.einsum_K_size = int(opdict["einsum_K_size"])

        return op

    def to_csv_dict(self) -> dict[str, Any]:
        '''
        Currently, this function is the same as EinsumOperator.to_csv_dict.
        '''
        opdict = super().to_csv_dict()
        opdict.update({
            "parsed_op_type": self.stats.parsed_op_type,
            "dim_labels": self.stats.dim_labels_str,
            "tile_shapes": self.stats.tile_shapes_str,
            "num_tiles": self.stats.num_tiles,
            "max_vmem_demand_bytes": self.stats.max_vmem_demand_bytes,
            "num_mxu_ops": self.stats.num_sa_ops,
            "einsum_B_size": self.stats.einsum_B_size,
            "einsum_M_size": self.stats.einsum_M_size,
            "einsum_N_size": self.stats.einsum_N_size,
            "einsum_K_size": self.stats.einsum_K_size,
        })
        return opdict


class FlashAttentionOperator(Operator):
    '''
    FlashAttention operator.
    '''
    stats: FlashAttentionStatistics = FlashAttentionStatistics()  # type: ignore

    op_type: OpType = OpType.MXU
    opcode_type: OpcodeType = OpcodeType.FLASH_ATTENTION

    @classmethod
    def from_csv_dict(cls, opdict: dict[str, Any]) -> "FlashAttentionOperator":
        '''
        Adapter for reading CSV traces that are compatible
        with TF-Sim and TF-Sim-Analytical.
        '''
        op = super(FlashAttentionOperator, cls).from_csv_dict(opdict)
        op = cls(**op.model_dump())

        op.stats.tile_shapes_str = opdict["tile_shapes"]
        op.stats.max_vmem_demand_bytes = int(opdict["max_vmem_demand_bytes"])
        op.stats.num_sa_ops = int(opdict["num_mxu_ops"])
        op.stats.einsum_B_size = int(opdict["einsum_B_size"])
        op.stats.einsum_M_size = int(opdict["einsum_M_size"])
        op.stats.einsum_N_size = int(opdict["einsum_N_size"])
        op.stats.einsum_K_size = int(opdict["einsum_K_size"])

        return op

    def to_csv_dict(self) -> dict[str, Any]:
        opdict = super().to_csv_dict()
        opdict.update({
            "parsed_op_type": self.stats.parsed_op_type,
            "tile_shapes": self.stats.tile_shapes_str,
            "max_vmem_demand_bytes": self.stats.max_vmem_demand_bytes,
            "num_mxu_ops": self.stats.num_sa_ops,
            "einsum_B_size": self.stats.einsum_B_size,
            "einsum_M_size": self.stats.einsum_M_size,
            "einsum_N_size": self.stats.einsum_N_size,
            "einsum_K_size": self.stats.einsum_K_size,
        })
        return opdict


class FusedOperator(Operator):
    '''
    Fused operator. Contains a list of @operators.
    '''

    operators: list[Operator] = []
    '''Operators'''


def from_csv_dict(opdict: dict[str, Any]) -> (
    Operator | EinsumOperator |
    Conv2DOperator | FlashAttentionOperator
):
    '''
    Adapter for reading CSV traces that are compatible
    with TF-Sim and TF-Sim-Analytical.
    '''
    opcode_type_parsed = OpcodeType.from_opcode(opdict["parsed_op_type"])
    opcode_type_native = OpcodeType.from_opcode(opdict["Op Code"])
    opcode_type = opcode_type_parsed
    if opcode_type == OpcodeType.OTHER:
        opcode_type = opcode_type_native

    if opcode_type == OpcodeType.EINSUM:
        op = EinsumOperator.from_csv_dict(opdict)
    elif opcode_type == OpcodeType.CONV2D:
        op = Conv2DOperator.from_csv_dict(opdict)
    elif opcode_type == OpcodeType.FLASH_ATTENTION:
        op = FlashAttentionOperator.from_csv_dict(opdict)
    else:
        op = Operator.from_csv_dict(opdict)

    return op


def to_csv_dict(
    op: Operator | EinsumOperator |
        Conv2DOperator | FlashAttentionOperator
) -> dict[str, Any]:
    '''
    Adapter for outputing CSV traces that are compatible
    with TF-Sim and TF-Sim-Analytical.
    Just a wrapper for @op.to_csv_dict().
    '''
    return op.to_csv_dict()
