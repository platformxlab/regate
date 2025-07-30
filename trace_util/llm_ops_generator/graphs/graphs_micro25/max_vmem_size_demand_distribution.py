#!/usr/bin/env python3

from typing import Any, Sequence
from absl import app, flags, logging

import json
import numpy as np
import matplotlib as PlotLib
from matplotlib.axes import Axes
import matplotlib.pyplot as PyPlot
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages
import os
import csv

from trace_util.llm_ops_generator import query_results_helper_lib as results_lib
import trace_util.llm_ops_generator.Operator as Operator

PlotLib.use('Agg')

PlotLib.rcParams['pdf.fonttype'] = 42
PlotLib.rcParams['ps.fonttype'] = 42

PlotLib.rcParams.update({'font.size': 16})
PlotLib.rcParams.update({'font.family': 'serif'})
PlotLib.rcParams['xtick.major.pad'] = '8'
PlotLib.rcParams['ytick.major.pad'] = '8'
PlotLib.rcParams['hatch.linewidth'] = 0.4

# font sizes
fontsize_xticklabel = 14
fontsize_yticklabel = 14
fontsize_xlabel = 16  # unused
fontsize_ylabel = 16
fontsize_legend = 14
fontsize_barlabel = 10

# bar width and x tick scaling factor
BarWidth = 0.4
XTickFactor = 2


__PG_STRATEGY = flags.DEFINE_string(
    "pg_strategy",
    "NoPG",
    "Power gating strategy to use for energy analysis, \
    one of 'NoPG', 'Base', 'HW', 'Full', 'Ideal'",
)
__NPU_VERSION = flags.DEFINE_string(
    "npu_version",
    "5p",
    "NPU version to use for energy analysis",
)

LLM_MODELS = ["llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b"]
LLM_MODEL_NAMES = [
    "Llama3-8B", "Llama2-13B", "Llama3-70B", "Llama3.1-405B"
]
# LLM_MODELS = ["llama3-70b", "llama3_1-405b"]
# LLM_MODEL_NAMES = [
#     "Llama3\n70B", "Llama3.1\n405B"
# ]
LLM_TRAINING_BATCH_SIZE = 32
DLRM_MODELS = ["dlrm-s", "dlrm-m", "dlrm-l"]
DLRM_MODEL_NAMES = ["DLRM-S", "DLRM-M", "DLRM-L"]
# DLRM_MODELS = ["dlrm-m", "dlrm-l"]
# DLRM_MODEL_NAMES = ["DLRM-M", "DLRM-L"]
SD_MODELS = ["dit-xl", "gligen"]
SD_MODEL_NAMES = ["DiT-XL", "GLIGEN"]
# NPU_VERSIONS = ["2", "3", "4", "5p"]

RESULTS_DIR = os.environ["RESULTS_DIR"]
SLO_RESULTS_DIR = os.path.join(RESULTS_DIR, "slo")

MAX_SLO_SCALE = 6

# Configs that cannot satisfy SLO within 8K chips
SLO_VIOLATION_CONFIGS = [
    # (model, v, workload, batch_size, prefill_or_decode, slo_scale)
]


def get_energy_optimal_chip_config(
    model: str,
    v: str,
    workload: str,
    batch_size: int = 1,
    prefill_or_decode: str = "",
    slo_scale: int = 1,
) -> tuple[tuple[int, ...], int]:
    '''
    @return: (split the pstr "dp1-tp8-pp1-dpdcn1-tpdcn1-ppdcn1-bs1" into a tuple of integers, slo_scale)
    '''
    global SLO_VIOLATION_CONFIGS

    if results_lib.is_model_llm(model):
        if workload == "inference":
            fname = f"{workload}-{prefill_or_decode}.json"
        else:
            fname = f"{workload}.json"
    else:
        fname = f"{workload}.json"
    results_path = os.path.join(SLO_RESULTS_DIR, model, fname)
    with open(results_path, "r") as f:
        raw_results = json.load(f)

    results = raw_results[str(slo_scale)][v]
    while "optimal_energy_eff_stats_file" not in results:
        # If no results that satisfy SLO can be found, there is no SLO results file
        SLO_VIOLATION_CONFIGS.append(
            (model, v, workload, batch_size, prefill_or_decode, slo_scale)
        )
        slo_scale += 1
        if slo_scale > MAX_SLO_SCALE: #str(slo_scale) not in raw_results:
            # no configs can satisfy any SLO
            return (1, 1, 1, 1, 1, 1, batch_size), -1
        results = raw_results[str(slo_scale)][v]

    stats_file_path = results["optimal_energy_eff_stats_file"]
    pstr = stats_file_path.split("/")[-2].split("-")
    pstr = (
        pstr[0].removeprefix("dp"),
        pstr[1].removeprefix("tp"),
        pstr[2].removeprefix("pp"),
        pstr[3].removeprefix("dpdcn"),
        pstr[4].removeprefix("tpdcn"),
        pstr[5].removeprefix("ppdcn"),
        pstr[6].removeprefix("bs"),
    )
    return tuple(map(int, pstr)), slo_scale


def violate_slo(
    model: str,
    v: str,
    workload: str,
    batch_size: int = 1,
    prefill_or_decode: str = "",
    slo_scale: int = 1,
) -> bool:
    global SLO_VIOLATION_CONFIGS
    return (model, v, workload, batch_size, prefill_or_decode, slo_scale) in SLO_VIOLATION_CONFIGS


def get_optimal_energy_stat(
    model: str,
    v: str,
    workload: str,
    batch_size: int = 1,
    prefill_or_decode: str = "",
) -> tuple[dict[str, Any], int]:
    '''Returns (stats, slo_scale). slo_scale is -1 is not config can satisfy @MAX_SLO_SCALE.'''
    results_lib.set_results_path(os.path.join(RESULTS_DIR, f"carbon_{__PG_STRATEGY.value}/CI0.0624/UTIL0.6"))
    pconfig, slo_scale = get_energy_optimal_chip_config(model, v, workload, batch_size, prefill_or_decode)
    assert len(pconfig) == 7, f"Invalid pconfig: {pconfig}"
    return results_lib.get_stats(
        model, v, workload, *pconfig, prefill_or_decode=prefill_or_decode
    ), slo_scale


def get_vmem_size_demand_data(stats: dict[str, Any]) -> tuple[list[float | int], list[float | int]]:
    '''
    @return[0]: list of exe time ns for each op
    @return[1]: list of vmem size demand for each op
    '''
    out_file_path = stats["sim_config"]["output_file_path"]
    ops: list[Operator.Operator] = []
    with open(out_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = Operator.from_csv_dict(row)
            ops.append(op)

    # get the max vmem size demand for each op
    exe_time_ns = []
    vmem_size_demand = []
    for op in ops:
        exe_time_ns.append(op.stats.execution_time_ns * op.stats.count)
        if isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics)):
            vmem_size_demand.append(op.stats.max_vmem_demand_bytes / 1024 / 1024)
        else:
            vmem_size_demand.append(8)  # 8MB for elementwise ops by default

    return exe_time_ns, vmem_size_demand

def get_data_helper(
    models: Sequence[str],
    workload: str,
    v: str,
    batch_size: int = -1,
    prefill_or_decode: str = "",
) -> tuple[list[list[float | int]], ...]:
    '''
    @return: (exe_time_ns[model][op], vmem_demand_MB[model][op], slo_scale[model])
    '''
    exe_time = []
    vmem_demand = []
    slo_scales = []
    for im, model in enumerate(models):
        stats, slo_scale = get_optimal_energy_stat(
            model, v, workload, batch_size, prefill_or_decode
        )
        exe_time_ns, vmem_demand_MB = get_vmem_size_demand_data(stats)
        # num_layers = stats["sim_config"]["num_layers"]
        # if slo_scale == -1:
        #     exe_time.append(0)
        #     vmem_demand.append(0)
        # else:
        exe_time.append(exe_time_ns)
        vmem_demand.append(vmem_demand_MB)
        slo_scales.append(slo_scale)

    return exe_time, vmem_demand, slo_scales


def get_energy_eff_llm_training() -> tuple[list[list[float | int]], ...]:
    return get_data_helper(
        LLM_MODELS,
        "training",
        __NPU_VERSION.value,
        LLM_TRAINING_BATCH_SIZE,
        "",
    )


def get_energy_eff_llm_inference(prefill_or_decode: str) -> tuple[list[list[float | int]], ...]:
    return get_data_helper(
        LLM_MODELS,
        "inference",
        __NPU_VERSION.value,
        -1,
        prefill_or_decode,
    )


def get_energy_eff_dlrm_inference() -> tuple[list[list[float | int]], ...]:
    return get_data_helper(
        DLRM_MODELS,
        "inference",
        __NPU_VERSION.value,
        -1,
    )


def get_energy_eff_sd_inference() -> tuple[list[list[float | int]], ...]:
    return get_data_helper(
        SD_MODELS,
        "inference",
        __NPU_VERSION.value,
        -1,
    )


def plot_data(
    ax: Axes,
    x_data: Sequence[Sequence[float | int]],
    y_data: Sequence[Sequence[float | int]],
    x_names: Sequence[str],
    y_label: str | None,
    title: str,
    x_lim,
    log_scale: bool = False,
):
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if y_label:
        ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"], fontsize=fontsize_xticklabel)
    else:
        ax.set_yticklabels("")
    ax.yaxis.set_ticks_position( 'none' )

    ax.xaxis.set_ticks_position( 'none' )

    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=1)
    ax.tick_params(axis="y", which="major", labelsize=fontsize_yticklabel)

    ax.set_axisbelow(True)
    ax.yaxis.grid(which='major', color='lightgray', linestyle='solid')
    ax.yaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    ax.xaxis.grid(which='major', color='lightgray', linestyle='solid')
    ax.xaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    if y_label:
        ax.set_ylabel( y_label, fontsize=fontsize_ylabel )
    ax.set_xlabel( "SRAM Demand (MB)", fontsize=fontsize_xlabel )

    ax.set_title( title, fontsize=fontsize_ylabel )

    if log_scale:
        ax.set_yscale( 'log' )

    colors = ["#a6c938", "#ef7a35", "#75c2aa", "#b08fc2"]
    linestyles = [":", "-.", "--", "-"]
    legend_names = x_names

    for idx, (x, y) in enumerate(zip(x_data, y_data)):
        ax.ecdf(
            y,
            x,
            label=legend_names[idx],
            linewidth=3,
            color=colors[idx],
            linestyle=linestyles[idx],
        )

    edgecolor = 'white' # '#404040'
    linewidth = 0.


    ax.set_ylim( 0, 1 )
    default_xlim = ax.get_xlim()
    new_xlim = (
        default_xlim[0] if x_lim[0] is None else x_lim[0],
        default_xlim[1] if x_lim[1] is None else x_lim[1],
    )
    ax.set_xlim( new_xlim )

    if "llama" in x_names[0].lower():
        legend_location = "lower right"
    elif "dlrm" in x_names[0].lower():
        legend_location = "upper left"
    else: # stable diffusion
        legend_location = "lower right"

    if ("llama" in x_names[0].lower() and "decode" in title.lower()) or "llama" not in x_names[0].lower():
        lg=ax.legend(prop={'size': fontsize_legend}, ncol=1, loc=legend_location, borderaxespad=0.)
        lg.draw_frame(False)


def main(argv: list[str]):
    del argv  # Unused

    LLM_XNames = LLM_MODEL_NAMES
    DLRM_XNames = DLRM_MODEL_NAMES
    SD_XNames = SD_MODEL_NAMES

    all_data = [
        get_energy_eff_llm_training(),
        get_energy_eff_llm_inference("prefill"),
        get_energy_eff_llm_inference("decode"),
        get_energy_eff_dlrm_inference(),
        get_energy_eff_sd_inference(),
    ]
    all_exe_time = [
        x for x, y, z in all_data
    ]
    all_vmem_demand = [
        y for x, y, z in all_data
    ]
    # all_slo_scale = [
    #     z for x, y, z in all_data
    # ]
    # set 1 value to empty string for SLO scale
    # all_slo_scale = [
    #     [
    #         [
    #             "" if x == 1 else f"{x}x"
    #             for x in slo_scale
    #         ]
    #         for slo_scale in slo_scales
    #     ]
    #     for slo_scales in all_slo_scale
    # ]

    all_XNames = [
        LLM_XNames,
        LLM_XNames,
        LLM_XNames,
        DLRM_XNames,
        SD_XNames,
    ]

    all_YLabels = [
        "CDF (Operator Exe. Time)",
        # "SA Spatial Util.",
        None,
        None,
        None,
        None,
        # "SA Spatial Util.",
        # "SA Spatial Util.",
        # "SA Spatial Util.",
        # "SA Spatial Util.",
    ]

    all_Titles = [
        "LLM Training",
        "LLM Prefill",
        "LLM Decode",
        "DLRM Inference",
        "Stable Diffusion",
    ]

    # None means use default.
    # Otherwise, use the specified value.
    # all_ylims = [
    #     (1e2, 1.2e5),
    #     (1e-4, 1e-1),
    #     (1e-2, 1.2e-1),
    # ]
    all_xlims = [
        (0, 1800),
        (0, None),
        (0, 128),
        (0, None),
        (0, 128),
    ]
    all_logy = [
        False,
        False,
        False,
        False,
        False,
    ]

    NROWS = 1
    NCOLS = 5

    Figure = PyPlot.figure( figsize=(17, 4) )
    pdf_filename = f"outputs/max_vmem_size_demand_distribution.pdf"
    PDF = PdfPages( pdf_filename )

    graphs = Figure.subplots( nrows=NROWS, ncols=NCOLS )

    for idx, (xnames, xdata, ydata, ylabel, title, xlim, logy) in enumerate(zip(
        all_XNames, all_exe_time, all_vmem_demand, all_YLabels, all_Titles, all_xlims, all_logy
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[idx]
        plot_data(Graph, xdata, ydata, xnames, ylabel, title, xlim, logy)

    Figure.tight_layout()
    
    Figure.subplots_adjust(wspace=0.02)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
