#!/usr/bin/env python3

import itertools
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
BarWidth = 0.5
XTickFactor = 3


__PG_STRATEGIES = flags.DEFINE_list(
    "pg_strategy",
    ["NoPG", "Base", "HW", "Full"],
    "Power gating strategy to use for energy analysis, \
    choose from 'NoPG', 'Base', 'HW', 'Full', 'Ideal'",
)
__NPU_VERSION = flags.DEFINE_string(
    "npu_version",
    "5p",
    "NPU version to use for energy analysis",
)

LLM_MODELS = ["llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b"]
LLM_MODEL_NAMES = [
    "8B", "13B", "70B", "405B"
]
LLM_TRAINING_BATCH_SIZE = 32
DLRM_MODELS = ["dlrm-s", "dlrm-m", "dlrm-l"]
DLRM_MODEL_NAMES = ["DLRM-S", "DLRM-M", "DLRM-L"]
SD_MODELS = ["dit-xl", "gligen"]
SD_MODEL_NAMES = ["DiT-XL", "GLIGEN"]

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


def get_optimal_energy_stat(
    model: str,
    v: str,
    workload: str,
    pg_strategy: str,
    batch_size: int = 1,
    prefill_or_decode: str = "",
) -> tuple[dict[str, Any], int]:
    '''Returns (stats, slo_scale). slo_scale is -1 is not config can satisfy @MAX_SLO_SCALE.'''
    results_lib.set_results_path(os.path.join(RESULTS_DIR, f"carbon_{pg_strategy}/CI0.0624/UTIL0.6"))
    pconfig, slo_scale = get_energy_optimal_chip_config(model, v, workload, batch_size, prefill_or_decode)
    assert len(pconfig) == 7, f"Invalid pconfig: {pconfig}"
    return results_lib.get_stats(
        model, v, workload, *pconfig, prefill_or_decode=prefill_or_decode
    ), slo_scale


def get_stats_helper(
    models: Sequence[str],
    pg_strategies: Sequence[str],
    workload: str,
    prefill_or_decode: str = "",
    batch_size: int = -1,
) -> list[list[int | float]]:
    '''
    @return[pg_strategy][model]
    '''
    values = [[] for _ in pg_strategies]
    for im, model in enumerate(models):
        for iv, pg in enumerate(pg_strategies):
            stats, slo_scale = get_optimal_energy_stat(
                model, __NPU_VERSION.value, workload, pg, batch_size, prefill_or_decode
            )
            metric, min_max_metric = results_lib.get_energy_eff_metric_name_and_min_max(
                model, workload, prefill_or_decode
            )
            # value = 1 / stats["carbon_and_energy_stats"]["1"][metric]
            value = stats["energy_stats"][pg]["component_stats"]["total_exe_time_ns"]
            values[iv].append(value)
    return values


def get_data_llm_training() -> list[list[int | float]]:
    return get_stats_helper(
        LLM_MODELS,
        __PG_STRATEGIES.value,
        "training",
        batch_size=LLM_TRAINING_BATCH_SIZE,
    )


def get_data_llm_inference(prefill_or_decode: str) -> list[list[int | float]]:
    return get_stats_helper(
        LLM_MODELS,
        __PG_STRATEGIES.value,
        "inference",
        prefill_or_decode=prefill_or_decode,
    )


def get_data_dlrm_inference() -> list[list[int | float]]:
    return get_stats_helper(
        DLRM_MODELS,
        __PG_STRATEGIES.value,
        "inference",
    )


def get_data_sd_inference() -> list[list[int | float]]:
    return get_stats_helper(
        SD_MODELS,
        __PG_STRATEGIES.value,
        "inference",
    )


def plot_data(
    ax: Axes,
    data: Sequence[Sequence[float]],
    x_names: Sequence[str],
    y_label: str | None,
    title: str,
    y_lim,
    log_scale: bool = False,
    plot_legend: bool = False,
    yticks: Sequence[float] | None = None,
    yticklabels: Sequence[str] | None = None,
):
    # normalize data to NoPG
    data = np.array(data)  # type: ignore
    data = data / data[0]  # type: ignore
    data = data - 1  # type: ignore

    # do not plot NoPG
    data = data[1:]

    print(f"################### {title} ####################")
    print(data)

    if log_scale:
        ax.set_yscale( 'log' )

    XValues = np.arange( len( data[ 0 ] ) ) * XTickFactor
    XTicks = XValues
    ax.set_xticks( XTicks )
    ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )
    ax.xaxis.set_ticks_position( 'none' )

    ax.yaxis.set_ticks_position( 'none' )
    if yticks and yticklabels:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=fontsize_yticklabel)
    # else:
    #     if not y_label:
    #         ax.set_yticklabels(["", "", "", "", "", ""])

    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=1)
    ax.tick_params(axis="y", which="major", labelsize=fontsize_yticklabel)

    ax.set_axisbelow(True)
    ax.yaxis.grid(which='major', color='lightgray', linestyle='solid')
    ax.yaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    if y_label:
        ax.set_ylabel( y_label, fontsize=fontsize_ylabel )

    ax.set_title( title, fontsize=fontsize_ylabel )

    color1, color2, color3, color4 = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    color1_light, color2_light, color3_light, color4_light = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    legend_names = __PG_STRATEGIES.value[1:]
    colors = ["#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA", "#00AAAA"]
    hatches = ["", "--", "//", "\\\\", "x"]

    edgecolor = 'white' # '#404040'
    linewidth = 0.

    num_bars = len(data)
    for i in range(num_bars):
        xvals = XValues-BarWidth*(num_bars/2-i-0.5)
        ax.bar(
            xvals, data[i], BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=colors[i], hatch=hatches[i], label=legend_names[i]
        )

    ax.set_xlim( ( XValues[0]-7*BarWidth/2, XValues[-1]+7*BarWidth/2) )
    default_ylim = ax.get_ylim()
    new_ylim = (
        default_ylim[0] if y_lim[0] is None else y_lim[0],
        default_ylim[1] if y_lim[1] is None else y_lim[1],
    )
    ax.set_ylim( new_ylim )

    # lg=ax.legend(prop={'size': fontsize_legend}, ncol=5, loc=2, borderaxespad=0.)
    # lg.draw_frame(False)
    if plot_legend:
        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])
        handles, labels = ax.get_legend_handles_labels()
        lg=ax.legend(
            flip(handles, 7), flip(labels, 7),
            prop={'size': fontsize_legend}, ncol=7, loc="upper center", borderaxespad=0.,
            bbox_to_anchor=(2.9, 1.3), frameon=False,
        )
        lg.draw_frame(True)
        frame = lg.get_frame()
        frame.set_edgecolor("black")


def main(argv: list[str]):
    del argv  # Unused

    LLM_XNames = LLM_MODEL_NAMES

    DLRM_XNames = DLRM_MODEL_NAMES

    SD_XNames = SD_MODEL_NAMES

    all_data = [
        get_data_llm_training(),
        get_data_llm_inference("prefill"),
        get_data_llm_inference("decode"),
        get_data_dlrm_inference(),
        get_data_sd_inference(),
    ]

    all_XNames = [
        LLM_XNames,
        LLM_XNames,
        LLM_XNames,
        DLRM_XNames,
        SD_XNames,
    ]

    all_YLabels = [
        "Performance Overhead",
        None,
        None,
        None,
        None,
    ]

    all_Titles = [
        "LLM Training",
        "LLM Prefill",
        "LLM Decode",
        "DLRM Inference",
        "Stable Diffusion",
    ]

    all_ylims = [
        (0, 0.05),
        (0, 0.05),
        (0, 0.003),
        (0, 0.003),
        (0, 0.05),
    ]
    all_logy = [
        False,
        False,
        False,
        False,
        False,
    ]

    all_yticks = [
        [0, 0.01, 0.02, 0.03, 0.04, 0.05],
        [0, 0.01, 0.02, 0.03, 0.04, 0.05],
        [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        [0, 0.01, 0.02, 0.03, 0.04, 0.05],
    ]
    all_yticklabels = [
        ["0", "1%", "2%", "3%", "4%", "5%"],
        ["0", "1%", "2%", "3%", "4%", "5%"],
        ["0", "0.1%", "0.2%", "0.3%", "0.4%", "0.5%", "0.6%"],
        ["0", "0.1%", "0.2%", "0.3%", "0.4%", "0.5%", "0.6%"],
        ["0", "1%", "2%", "3%", "4%", "5%"],
    ]

    NROWS = 1
    NCOLS = 5

    Figure = PyPlot.figure( figsize=(22, 3) )
    pdf_filename = f"outputs/eval_perf_impact.pdf"
    PDF = PdfPages( pdf_filename )

    graphs = Figure.subplots( nrows=NROWS, ncols=NCOLS )

    for idx, (xnames, data, ylabel, title, ylim, logy, yticks, yticklabels) in enumerate(zip(
        all_XNames, all_data, all_YLabels, all_Titles, all_ylims, all_logy, all_yticks, all_yticklabels
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[idx]
        plot_legend = (idx == 0)
        plot_data(Graph, data, xnames, ylabel, title, ylim, logy, plot_legend, yticks, yticklabels)

    Figure.tight_layout()
    # Figure.subplots_adjust(wspace=0.04)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
