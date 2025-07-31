### Backend for the power simulation. The entry function is analyze_operator_energy(),
### which can be called on each tensor operator to update its static and dynamic
### energy consumption stats as well as the execution time impact.
### The function get_power_gating_config() contains several pre-defined power-gating
### configurations, which can be used to analyze the power gating impact on the operator.

from copy import deepcopy
from enum import Enum
from math import ceil
from tkinter import SW
from typing import Any

from absl import flags, logging
import numpy as np
import json

from pydantic import BaseModel

import trace_util.llm_ops_generator.Operator as Operator
from trace_util.llm_ops_generator.configs.chips.ChipConfig import ChipConfig


class PowerGatingConfig(BaseModel):
    """
    Power-gating configuration.
    """

    class TemporalGranularity(Enum):
        INSTRUCTION = 1
        OPERATOR = 2
        APPLICATION = 3

    class SASpatialGranularity(Enum):
        PE = 1
        PARTITION = 2
        COMPONENT = 3

    class VUSpatialGranularity(Enum):
        ALU = 1
        PARTITION = 2
        COMPONENT = 3

    class VmemSpatialGranularity(Enum):
        REGISTER_SIZE = 1
        PARTITION = 2

    class ICISpatialGranularity(Enum):
        LINK = 1
        COMPONENT = 2

    class VoltageGranularity(Enum):
        TWO_LEVEL = 1  # only on/off
        MULTI_LEVEL = 2  # on/off + sleep modes

    class PowerGatingPolicy(Enum):
        HW = 1  # HW-managed (auto mode)
        SW = 2  # SW-managed

    name: str = "PowerGatingConfig"
    SA_PG_enabled: bool = False
    SA_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    SA_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    SA_spatial_granularity: SASpatialGranularity = SASpatialGranularity.COMPONENT
    sa_partition_shapes: list[int] = [128, 128]
    """partition shapes in number of PEs (128*128 by default)"""
    sa_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    sa_pe_pg_delay_cycles: int = 1
    """Delay in cycles of power gating and waking up a single PE."""
    sa_pg_delay_cycles: int = 10
    """Delay in cycles of power gating and waking up the entire SA."""

    VU_PG_enabled: bool = False
    VU_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    VU_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    VU_spatial_granularity: VUSpatialGranularity = VUSpatialGranularity.COMPONENT
    vu_partition_shapes: list[int] = [8, 128]
    """partition shapes in number of ALUs (8*128 by default)"""
    vu_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    vu_pg_delay_cycles: int = 2
    """Delay in cycles of power gating and waking up a VU."""

    vmem_PG_enabled: bool = False
    vmem_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    vmem_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    vmem_spatial_granularity: VmemSpatialGranularity = VmemSpatialGranularity.PARTITION
    vmem_voltage_granularity: VoltageGranularity = VoltageGranularity.TWO_LEVEL
    vmem_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    vmem_partition_size_bytes: int = 2 * 1024 * 1024
    """partition size in bytes (2MB by default if spatial granularity is PARTITION)"""
    vmem_partition_pg_delay_cycles: int = 10
    """Delay in cycles of power gating and waking up a vmem partition."""
    vmem_HW_drowsy_period_cycles: int = 2000
    """Period at which all vmem partitions are put into sleep."""

    ici_PG_enabled: bool = False
    ici_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    ici_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    ici_spatial_granularity: ICISpatialGranularity = ICISpatialGranularity.COMPONENT
    ici_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    ici_pg_delay_cycles: int = 10
    """Delay in cycles of power gating and waking up an ICI."""

    hbm_PG_enabled: bool = False
    hbm_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    hbm_power_level_factors: list[float] = [1.0, 0.1]  # 0.1 takes into account the auto refresh cost
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    hbm_refresh_interval_ns: int = 3900
    hbm_refresh_delay_ns: int = 400  # for 12H device
    hbm_pg_delay_cycles: int = 60

    other_PG_enabled: bool = False
    other_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    other_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""


def get_power_gating_config(pg_config_name: str) -> PowerGatingConfig:
    """
    'disabled', 'NoPG': no power gating. \n
    'ideal_inst_component': ideal power gating with instruction-level temporal granularity and component-level spatial granularity. \n
    'ideal_op_component': ideal power gating with operator-level temporal granularity and component-level spatial granularity. \n
    'ideal_inst_PE_ALU', 'Ideal': ideal power gating with instruction-level temporal granularity and PE/ALU-level spatial granularity. This should result in the most power savings. \n
    'Full': Same as 'Ideal' but with non-zero power-gating factor (power_level_factors) and delay cycles. \n
    '\\<base_config\\>\\_vary_Vth_\\<value\\>_\\<value_sram\\>': vary Vth_low and Vth_sram for sensitivity analysis. The values are the percentage over Vdd. \n
    '\\<base_config\\>\\_vary_PG_delay_\\<value\\>': vary PG delay for sensitivity analysis. The value is specified as the ratio over base config. \n
    """
    pg_config = PowerGatingConfig
    if pg_config_name in ["disabled", "NoPG"]:
        pg_config = PowerGatingConfig(name="NoPG")
    elif pg_config_name == "ideal_inst_component":
        pg_config = PowerGatingConfig(
            name="ideal_inst_component",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.COMPONENT,
            sa_partition_shapes=[128, 128],
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.PARTITION,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
        )
    elif pg_config_name == "ideal_op_component":
        pg_config = PowerGatingConfig(
            name="ideal_op_component",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.COMPONENT,
            sa_partition_shapes=[128, 128],
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.PARTITION,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
        )
    elif pg_config_name in ["ideal_inst_PE_ALU", "Ideal"]:
        pg_config = PowerGatingConfig(
            name="Ideal",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.PE,
            sa_partition_shapes=[128, 128],
            sa_power_level_factors=[1.0, 0.0],
            sa_pe_pg_delay_cycles=0,
            sa_pg_delay_cycles=0,
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.ALU,
            vu_power_level_factors=[1.0, 0.0],
            vu_pg_delay_cycles=0,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=0,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.0],
            ici_pg_delay_cycles=0,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
            hbm_pg_delay_cycles=0,
        )
    elif pg_config_name.startswith("Base"):
        pg_config = PowerGatingConfig(
            name="Base",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.COMPONENT,
            sa_partition_shapes=[128, 128],
            # 0.03 -> 0.05 accounts for the fact that weight registers cannot be power gated
            sa_power_level_factors=[1.0, 0.05],
            sa_pe_pg_delay_cycles=1,
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vu_power_level_factors=[1.0, 0.03],
            vu_pg_delay_cycles=2,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.25],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=4,
            vmem_HW_drowsy_period_cycles=2000,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.03],
            ici_pg_delay_cycles=60,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.HW,
        )
    elif pg_config_name.startswith("HW"):
        pg_config = PowerGatingConfig(
            name="HW",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.PE,
            sa_partition_shapes=[128, 128],
            sa_power_level_factors=[1.0, 0.03],
            sa_pe_pg_delay_cycles=1,
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vu_power_level_factors=[1.0, 0.03],
            vu_pg_delay_cycles=2,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.25],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=4,
            vmem_HW_drowsy_period_cycles=2000,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.03],
            ici_pg_delay_cycles=60,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.HW,
        )
    elif pg_config_name.startswith("Full"):
        pg_config = PowerGatingConfig(
            name="Full",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.PE,
            sa_partition_shapes=[128, 128],
            sa_power_level_factors=[1.0, 0.03],
            sa_pe_pg_delay_cycles=1,
            VU_PG_enabled=True,
            VU_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vu_power_level_factors=[1.0, 0.03],
            vu_pg_delay_cycles=2,
            vmem_PG_enabled=True,
            vmem_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0002],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=10,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.03],
            ici_pg_delay_cycles=60,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
        )
    else:
        raise ValueError(f"Unsupported power gating configuration: {pg_config_name}")

    # vary Vth_low and PG delay for sensitivity analysis
    if "vary_Vth" in pg_config_name:
        # name scheme: "<base_config_name>_vary_Vth_<value>_<value_sram>"
        pg_config.name = pg_config_name
        Vth = float(pg_config_name.split("_")[-2])
        Vth_sram = float(pg_config_name.split("_")[-1])
        pg_config.sa_power_level_factors[-1] = Vth
        pg_config.vu_power_level_factors[-1] = Vth
        pg_config.vmem_power_level_factors[-1] = Vth_sram
        pg_config.ici_power_level_factors[-1] = Vth
        pg_config.hbm_power_level_factors[-1] = Vth
        pg_config.other_power_level_factors[-1] = Vth
    if "vary_PG_delay" in pg_config_name:
        # name scheme: "<base_config_name>_vary_PG_delay_<value>"
        # <value> is the extra delay ratio: new delay = old delay * <value>
        # This do not apply to PEs in the SA.
        pg_config.name = pg_config_name
        pg_delay = float(pg_config_name.split("_")[-1])
        pg_config.sa_pg_delay_cycles = ceil(pg_config.sa_pg_delay_cycles * pg_delay)
        pg_config.vu_pg_delay_cycles = ceil(pg_config.vu_pg_delay_cycles * pg_delay)
        pg_config.vmem_partition_pg_delay_cycles = ceil(
            pg_config.vmem_partition_pg_delay_cycles * pg_delay
        )
        pg_config.ici_pg_delay_cycles = ceil(pg_config.ici_pg_delay_cycles * pg_delay)

    return pg_config


def compute_peak_sa_flops_per_sec_from_chip_config(config: ChipConfig) -> float:
    freq = config.freq_GHz * 1e9
    num_sa = config.num_sa
    sa_dim_size = config.sa_dim
    return 2 * (sa_dim_size**2) * freq * num_sa


def compute_peak_vu_flops_per_sec_from_chip_config(config: ChipConfig) -> float:
    freq = config.freq_GHz * 1e9
    num_vu = config.num_vu
    vu_num_alus = 128 * 8  # TODO: make this a parameter in chip config
    return vu_num_alus * freq * num_vu


def compute_sa_flops_util(op: Operator.Operator, config: ChipConfig) -> float:
    """
    Compute SA flops utilization for an operator.
    """
    peak_sa_flops_per_sec = compute_peak_sa_flops_per_sec_from_chip_config(config)
    sa_time_ns = op.stats.sa_time_ns
    if op.op_type == Operator.OpType.MXU:
        assert sa_time_ns > 0, f"SA time is 0 for op: {op.to_csv_dict()}"
        sa_flops_util = min(
            (op.stats.flop_count / sa_time_ns * 1e9) / peak_sa_flops_per_sec,
            1.0,
        )
    else:
        sa_flops_util = 0
    return sa_flops_util


def compute_vu_flops_util(op: Operator.Operator, config: ChipConfig) -> float:
    """
    Compute VU flops utilization for an operator.
    """
    peak_vu_flops_per_sec = compute_peak_vu_flops_per_sec_from_chip_config(config)
    vu_time_ns = op.stats.vu_time_ns
    if op.op_type == Operator.OpType.MXU:
        # assumes vu flops is at least 1/8 of sa flops for accmulation
        vu_flops_util = min(
            (op.stats.flop_count / 8 / vu_time_ns * 1e9) / peak_vu_flops_per_sec,
            1.0,
        )
    else:
        if peak_vu_flops_per_sec > 0 and vu_time_ns > 0:
            vu_flops_util = min(
                (op.stats.flop_count / vu_time_ns * 1e9) / peak_vu_flops_per_sec,
                1.0,
            )
        else:
            vu_flops_util = 0
    return vu_flops_util


def cycle_to_ns(cycles: int, freq_GHz: float) -> float:
    """
    Convert cycles to nanoseconds.
    """
    return cycles / freq_GHz


def ns_to_cycle(ns: float, freq_GHz: float) -> float:
    """
    Convert nanoseconds to cycles.
    """
    return ns * freq_GHz


def analyze_dynamic_energy(
    op: Operator.Operator, config: ChipConfig
) -> Operator.Operator:
    """
    Analyze dynamic power and energy for an operator.
    Must be called after analyze_*_static_energy since it relies on the new execution times.
    """
    dynamic_sa_peak_W = config.dynamic_power_sa_W
    dynamic_vu_peak_W = config.dynamic_power_vu_W
    dynamic_vmem_peak_W = config.dynamic_power_vmem_W
    dynamic_ici_peak_W = config.dynamic_power_ici_W
    dynamic_hbm_peak_W = config.dynamic_power_hbm_W
    dynamic_other_W = config.dynamic_power_other_W

    exe_time_ns = op.stats.execution_time_ns
    sa_time_ns = op.stats.sa_time_ns
    vu_time_ns = op.stats.vu_time_ns
    ici_time_ns = op.stats.ici_time_ns
    hbm_time_ns = op.stats.memory_time_ns
    vmem_time_ns = op.stats.vmem_time_ns

    # compute flops utilization for SA and VU
    sa_flops_util = compute_sa_flops_util(op, config)
    vu_flops_util = compute_vu_flops_util(op, config)

    # Dynamic power is dissipated only when the corresponding component is active.
    op.stats.dynamic_energy_sa_J = dynamic_sa_peak_W * sa_time_ns / 1e9 * sa_flops_util
    op.stats.dynamic_energy_vu_J = dynamic_vu_peak_W * vu_time_ns / 1e9 * vu_flops_util
    op.stats.dynamic_energy_sram_J = dynamic_vmem_peak_W * vmem_time_ns / 1e9
    op.stats.dynamic_energy_ici_J = dynamic_ici_peak_W * ici_time_ns / 1e9
    op.stats.dynamic_energy_hbm_J = dynamic_hbm_peak_W * hbm_time_ns / 1e9
    op.stats.dynamic_energy_other_J = dynamic_other_W * exe_time_ns / 1e9

    return op


def analyze_sa_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for SA.
    """
    static_sa_W = config.static_power_sa_W
    pg_power_W = static_sa_W * pg_config.sa_power_level_factors[-1]

    # No power-gating
    if not pg_config.SA_PG_enabled:
        op.stats.static_energy_sa_J = static_sa_W * op.stats.execution_time_ns / 1e9
        return op

    if op.stats.sa_time_ns > 0:
        # assumes in the worst case, idle intervals are evenly distributed
        # over the entire execution time
        worst_case_sa_idle_interval_ns = ceil(
            (op.stats.execution_time_ns - 1) / (op.stats.sa_time_ns / config.sa_dim)
        )
        if worst_case_sa_idle_interval_ns == 0:
            # if SA is not idle (op is SA-bound), then no power gating
            op.stats.static_energy_sa_J = static_sa_W * op.stats.execution_time_ns / 1e9
            return op
    else:
        worst_case_sa_idle_interval_ns = 0

    sa_flops_util = compute_sa_flops_util(op, config)

    # calculate PG delay overhead and update op stats
    if (
        op.stats.sa_time_ns > 0
        and pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):  # used by HW and Full
        assert isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics))
        overhead_ns_1 = ceil(
            op.stats.sa_time_ns / config.sa_dim * cycle_to_ns(
                pg_config.sa_pe_pg_delay_cycles, config.freq_GHz
            )
        )
        overhead_ns_2 = ceil(
            op.stats.num_sa_ops * cycle_to_ns(pg_config.sa_pe_pg_delay_cycles, config.freq_GHz)
        )
        overhead_ns = min(overhead_ns_1, overhead_ns_2)
        op.stats.sa_time_ns += overhead_ns
    elif (
        op.stats.sa_time_ns > 0
        and pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):  # used by Base (idle-detect policy)
        if worst_case_sa_idle_interval_ns > 4 * cycle_to_ns(
            pg_config.sa_pg_delay_cycles, config.freq_GHz
        ):
            pg_delay_ns = ceil(
                cycle_to_ns(pg_config.sa_pg_delay_cycles, config.freq_GHz)
            )
            # sa_time_ns/sa_dim is the worst case number of idle intervals
            op.stats.sa_time_ns += ceil(pg_delay_ns * (op.stats.sa_time_ns / config.sa_dim))
    # if op.stats.sa_time_ns > op.stats.execution_time_ns:
    #     op.stats.execution_time_ns = op.stats.sa_time_ns
    #     op.stats.bounded_by = "Compute"

    sa_time_ns = op.stats.sa_time_ns
    exe_time_ns = max(op.stats.execution_time_ns, sa_time_ns)

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):
        pg_energy = pg_power_W * (exe_time_ns - sa_time_ns) / 1e9
        static_energy = static_sa_W * sa_time_ns / 1e9

        # Base policy: do not power gate if idle interval is smaller than 2x pg delay time
        if sa_time_ns > 0 and worst_case_sa_idle_interval_ns < 2 * cycle_to_ns(
            pg_config.sa_pg_delay_cycles, config.freq_GHz
        ):
            pg_energy = 0
            static_energy = static_sa_W * exe_time_ns / 1e9

        op.stats.static_energy_sa_J = pg_energy + static_energy
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):
        pg_energy = (
            pg_power_W * sa_time_ns / 1e9 * (1 - sa_flops_util)
            + pg_power_W * (exe_time_ns - sa_time_ns) / 1e9
        )
        static_energy = static_sa_W * sa_time_ns / 1e9 * sa_flops_util
        op.stats.static_energy_sa_J = pg_energy + static_energy
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):
        if op.op_type == Operator.OpType.MXU:
            op.stats.static_energy_sa_J = static_sa_W * exe_time_ns / 1e9
        else:
            op.stats.static_energy_sa_J = 0
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):
        op.stats.static_energy_sa_J = static_sa_W * exe_time_ns / 1e9 * sa_flops_util
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid SA power gating configuration")


def analyze_vu_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for VU.
    """
    static_vu_W = config.static_power_vu_W
    pg_power_W = static_vu_W * pg_config.vu_power_level_factors[-1]

    # No power-gating
    if not pg_config.VU_PG_enabled:
        op.stats.static_energy_vu_J = static_vu_W * op.stats.execution_time_ns / 1e9
        return op

    if op.stats.vu_time_ns > 0:
        # assumes in the worst case, idle intervals are evenly distributed
        # over the entire execution time
        worst_case_vu_idle_interval_ns = ceil(
            (op.stats.execution_time_ns - 1) / op.stats.vu_time_ns
        )
        if worst_case_vu_idle_interval_ns == 0:
            # if VU is not idle (op is VU-bound), then no power gating
            op.stats.static_energy_vu_J = static_vu_W * op.stats.execution_time_ns / 1e9
            return op
    else:
        worst_case_vu_idle_interval_ns = 0

    vu_flops_util = compute_vu_flops_util(op, config)

    # calculate PG delay overhead and update op stats
    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW
    ):  # used by Base and HW (idle-detect policy)
        if worst_case_vu_idle_interval_ns > 4 * cycle_to_ns(
            pg_config.vu_pg_delay_cycles, config.freq_GHz
        ):
            pg_delay_ns = ceil(
                cycle_to_ns(pg_config.vu_pg_delay_cycles, config.freq_GHz)
            )
            # vu_time_ns is the worst case number of idle intervals
            op.stats.vu_time_ns += pg_delay_ns * op.stats.vu_time_ns
            # if op.stats.vu_time_ns > op.stats.execution_time_ns:
            #     op.stats.execution_time_ns = op.stats.vu_time_ns
            #     op.stats.bounded_by = "Compute"

    vu_time_ns = op.stats.vu_time_ns
    exe_time_ns = max(op.stats.execution_time_ns, vu_time_ns)

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.COMPONENT
    ):  # used by Base, HW, and Full pg_config
        pg_energy = pg_power_W * (exe_time_ns - vu_time_ns) / 1e9
        static_energy = static_vu_W * vu_time_ns / 1e9

        # HW policy: do not power gate if idle interval is smaller than 2x pg delay time
        if pg_config.VU_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW:
            if worst_case_vu_idle_interval_ns < 2 * cycle_to_ns(
                pg_config.vu_pg_delay_cycles, config.freq_GHz
            ):
                pg_energy = 0
                static_energy = static_vu_W * exe_time_ns / 1e9

        op.stats.static_energy_vu_J = pg_energy + static_energy

        # Full policy: calculate number of setpm instructions
        if pg_config.VU_PG_policy == PowerGatingConfig.PowerGatingPolicy.SW:
            if exe_time_ns == vu_time_ns:  # VU bound; no VU idle intervals
                op.stats.num_setpm_vu = 0
            elif vu_time_ns == 0:  # VU idle; set VUs to be PG'ed only once
                op.stats.num_setpm_vu = 1
            else:  # use the number of idle intervals as an estimate
                op.stats.num_setpm_vu = min(
                    round(exe_time_ns / worst_case_vu_idle_interval_ns),
                    (exe_time_ns - vu_time_ns) // 32,  # 32 cycles BET for VU with wake-up delay of 2 cycles on TPUv5p
                                                       # This division estimates the max number of setpm instructions
                                                       # TODO: make this a parameter in PG config
                )
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.ALU
    ):  # only used by Ideal pg_config for now
        pg_energy = (
            pg_power_W * vu_time_ns / 1e9 * (1 - vu_flops_util)
            + pg_power_W * (exe_time_ns - vu_time_ns) / 1e9
        )
        static_energy = static_vu_W * vu_time_ns / 1e9 * vu_flops_util
        op.stats.static_energy_vu_J = pg_energy + static_energy
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.COMPONENT
    ):
        if vu_time_ns > 0:
            op.stats.static_energy_vu_J = static_vu_W * exe_time_ns / 1e9
        else:
            op.stats.static_energy_vu_J = 0
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.ALU
    ):
        op.stats.static_energy_vu_J = static_vu_W * exe_time_ns / 1e9 * vu_flops_util
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.COMPONENT
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.ALU
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid VU power gating configuration")


def analyze_vmem_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for vmem.
    """
    static_vmem_W = config.static_power_vmem_W
    pg_power_W = static_vmem_W * pg_config.vmem_power_level_factors[-1]

    # No power-gating
    if not pg_config.vmem_PG_enabled:
        op.stats.static_energy_sram_J = static_vmem_W * op.stats.execution_time_ns / 1e9
        op.stats.vmem_time_ns = op.stats.execution_time_ns
        return op

    partition_granularity = pg_config.vmem_partition_size_bytes
    if (
        pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
    ):
        partition_granularity = 4 * 1024  # 4KB

    vmem_size = config.vmem_size_MB * 1024 * 1024

    # compute vmem capacity utilization
    if op.op_type == Operator.OpType.MXU:
        assert isinstance(
            op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics)
        ), f"op_name: {op.name} :: op_type: {op.op_type}, opcode_type: {op.opcode_type}, opcode: {op.opcode} not supported for op.stats type {type(op.stats)}\nconfig: {op.config_str}"
        max_vmem_demand = op.stats.max_vmem_demand_bytes
        vmem_capacity_util = min(max_vmem_demand / vmem_size, 1.0)
    else:
        # only use 2MB per core (4MB in total) for operators w/o data reuse
        vmem_capacity_util = 4 / config.vmem_size_MB
    vmem_demand_ceiled = (
        int(np.ceil(vmem_capacity_util * vmem_size / partition_granularity))
        * partition_granularity
    )
    vmem_capacity_util = vmem_demand_ceiled / vmem_size

    exe_time_ns = op.stats.execution_time_ns

    # calculate PG delay overhead and update op stats
    if pg_config.vmem_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW:
        pg_delay_overhead = ceil(
            op.stats.execution_time_ns
            / cycle_to_ns(pg_config.vmem_HW_drowsy_period_cycles, config.freq_GHz)
            * cycle_to_ns(pg_config.vmem_partition_pg_delay_cycles, config.freq_GHz)
        )
        exe_time_ns += pg_delay_overhead

    # TODO: better vmem simulation
    # For now, we assume vmem is always on when doing compute or HBM DMA or ICI DMA
    op.stats.vmem_time_ns = exe_time_ns
    vmem_time_ns = op.stats.vmem_time_ns
    exe_time_ns = max(exe_time_ns, vmem_time_ns)

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy

        # Full policy: calculate number of setpm instructions
        if pg_config.vmem_PG_policy == PowerGatingConfig.PowerGatingPolicy.SW:
            # only set once per operator as we assume fixed tile size per operator for now
            op.stats.num_setpm_sram = 1

        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy
        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy
        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy
        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid Vmem power gating configuration")


def analyze_ici_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for ICI.
    """
    static_ici_W = config.static_power_ici_W
    pg_power_W = static_ici_W * pg_config.ici_power_level_factors[-1]

    # No power-gating
    if not pg_config.ici_PG_enabled:
        op.stats.static_energy_ici_J = static_ici_W * op.stats.execution_time_ns / 1e9
        return op

    # assert (
    #     pg_config.ici_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW
    # ), "Only HW-managed power gating is supported for ICI"

    # calculate PG delay overhead and update op stats
    if op.stats.ici_time_ns > 0:
        pg_delay_ns = ceil(
            2 * cycle_to_ns(pg_config.ici_pg_delay_cycles, config.freq_GHz)
        )
        op.stats.ici_time_ns += pg_delay_ns
        # if op.stats.ici_time_ns > op.stats.execution_time_ns:
        #     op.stats.execution_time_ns = op.stats.ici_time_ns
        #     op.stats.bounded_by = "ICI/NVLink"

    ici_time_ns = op.stats.ici_time_ns
    exe_time_ns = max(op.stats.execution_time_ns, ici_time_ns)

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.LINK
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.COMPONENT
    ):
        pg_energy = pg_power_W * (exe_time_ns - ici_time_ns) / 1e9
        static_energy = static_ici_W * ici_time_ns / 1e9
        op.stats.static_energy_ici_J = pg_energy + static_energy
        return op

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.LINK
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.COMPONENT
    ):
        if ici_time_ns > 0:
            op.stats.static_energy_ici_J = static_ici_W * exe_time_ns / 1e9
        else:
            op.stats.static_energy_ici_J = pg_power_W * exe_time_ns / 1e9
        return op

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.LINK
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.COMPONENT
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid ICI power gating configuration")


def analyze_hbm_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for HBM.
    """
    static_hbm_W = config.static_power_hbm_W
    pg_power_W = static_hbm_W * pg_config.hbm_power_level_factors[-1]

    # No power-gating
    if not pg_config.hbm_PG_enabled:
        op.stats.static_energy_hbm_J = (
            config.static_power_hbm_W * op.stats.execution_time_ns / 1e9
        )
        return op

    # assume 4MB DMA size if memory traffic is larger than this
    if op.stats.memory_traffic_bytes < 4 * 1024 * 1024:
        active_length_ns = op.stats.memory_time_ns
    else:
        active_length_ns = (
            4 * 1024 * 1024 / (config.hbm_bw_GBps * 1024 * 1024 * 1024) * 1e9
        )
    hbm_util = op.stats.memory_time_ns / op.stats.execution_time_ns
    num_periods = ceil(
        op.stats.memory_time_ns / active_length_ns
    )
    idle_length_ns = ceil(
        op.stats.execution_time_ns / num_periods - active_length_ns
    )

    # break-even time
    BET_ns = config.hbm_latency_ns * 2 + ceil(cycle_to_ns(pg_config.hbm_pg_delay_cycles, config.freq_GHz)) * 4
    idle_detect_timeout_ns = BET_ns * 4

    if pg_config.hbm_PG_policy == PowerGatingConfig.PowerGatingPolicy.SW:
        if idle_length_ns >= BET_ns:
            # power gate HBM
            pg_energy = pg_power_W * (op.stats.execution_time_ns - op.stats.memory_time_ns) / 1e9
            static_energy = op.stats.memory_time_ns / 1e9 * static_hbm_W
            op.stats.static_energy_hbm_J = pg_energy + static_energy
        else:
            # do not power gate HBM
            op.stats.static_energy_hbm_J = static_hbm_W * op.stats.execution_time_ns / 1e9
    elif pg_config.hbm_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW:
        if idle_length_ns >= idle_detect_timeout_ns:
            # power gate HBM
            pg_energy = pg_power_W * (op.stats.execution_time_ns - op.stats.memory_time_ns) / 1e9
            static_energy = op.stats.memory_time_ns / 1e9 * static_hbm_W
            op.stats.static_energy_hbm_J = pg_energy + static_energy

            # calculate PG delay overhead and update op stats
            pg_delay_ns = ceil(
                cycle_to_ns(pg_config.hbm_pg_delay_cycles, config.freq_GHz) * num_periods
            )
            op.stats.memory_time_ns += pg_delay_ns
        else:
            # do not power gate HBM
            op.stats.static_energy_hbm_J = static_hbm_W * op.stats.execution_time_ns / 1e9
    else:
        raise NotImplementedError("Unknown HBM power gating policy")

    return op


def analyze_other_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for other.
    """
    assert pg_config.other_PG_enabled is False, "Other power gating is not supported"

    op.stats.static_energy_other_J = (
        config.static_power_other_W * op.stats.execution_time_ns / 1e9
    )
    return op


def analyze_operator_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: str | PowerGatingConfig
) -> Operator.Operator:
    if isinstance(pg_config, str):
        # get predefined power-gating configs from config name
        pg_config = get_power_gating_config(pg_config)

    # static power
    analyze_sa_static_energy(op, config, pg_config)
    analyze_vu_static_energy(op, config, pg_config)
    analyze_vmem_static_energy(op, config, pg_config)
    analyze_ici_static_energy(op, config, pg_config)
    analyze_hbm_static_energy(op, config, pg_config)
    analyze_other_static_energy(op, config, pg_config)

    # update execution time and bounded-by
    if op.stats.execution_time_ns < op.stats.sa_time_ns:
        op.stats.execution_time_ns = op.stats.sa_time_ns
        op.stats.bounded_by = "Compute"
    if op.stats.execution_time_ns < op.stats.vu_time_ns:
        op.stats.execution_time_ns = op.stats.vu_time_ns
        op.stats.bounded_by = "Compute"
    if op.stats.execution_time_ns < op.stats.vmem_time_ns:
        op.stats.execution_time_ns = op.stats.vmem_time_ns
        op.stats.bounded_by = "Compute"
    if op.stats.execution_time_ns < op.stats.ici_time_ns:
        op.stats.execution_time_ns = op.stats.ici_time_ns
        op.stats.bounded_by = "ICI/NVLink"

    # Dynamic power
    analyze_dynamic_energy(op, config)

    return op
