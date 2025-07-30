#!/usr/bin/env python3

import itertools
from math import ceil
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
fontsize_xticklabel = 12
fontsize_yticklabel = 14
fontsize_xlabel = 16  # unused
fontsize_ylabel = 15
fontsize_legend = 14
fontsize_barlabel = 10

# bar width and x tick scaling factor
BarWidth = 0.4
XTickFactor = 2.5


# __PG_STRATEGIES = flags.DEFINE_list(
#     "pg_strategy",
#     ["NoPG", "Base", "HW", "Full", "Ideal"],
#     "Power gating strategy to use for energy analysis, \
#     choose from 'NoPG', 'Base', 'HW', 'Full', 'Ideal'",
# )
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
DLRM_MODEL_NAMES = ["S", "M", "L"]
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
    batch_size: int = 1,
    prefill_or_decode: str = "",
) -> tuple[dict[str, Any], int]:
    '''Returns (stats, slo_scale). slo_scale is -1 is not config can satisfy @MAX_SLO_SCALE.'''
    results_lib.set_results_path(os.path.join(RESULTS_DIR, "raw_energy"))
    pconfig, slo_scale = get_energy_optimal_chip_config(model, v, workload, batch_size, prefill_or_decode)
    assert len(pconfig) == 7, f"Invalid pconfig: {pconfig}"
    return results_lib.get_stats(
        model, v, workload, *pconfig, prefill_or_decode=prefill_or_decode
    ), slo_scale


def get_stats_helper(
    models: Sequence[str],
    workload: str,
    prefill_or_decode: str = "",
    batch_size: int = -1,
) -> tuple[list[int | float], ...]:
    '''
    @return (vu_data[model], sram_data[model])
    '''
    vu_data = []
    sram_data = []
    for im, model in enumerate(models):
        stats, slo_scale = get_optimal_energy_stat(
            model, __NPU_VERSION.value, workload, batch_size, prefill_or_decode
        )
        total_time_ns = int(stats["energy_stats"]["Full"]["component_stats"]["total_exe_time_ns"])
        freq_GHz = float(stats["sim_config"]["freq_GHz"])
        total_time_cycles = total_time_ns * freq_GHz
        num_setpm_vu = int(stats["energy_stats"]["Full"]["component_stats"]["num_setpm_vu"])
        num_setpm_sram = int(stats["energy_stats"]["Full"]["component_stats"]["num_setpm_sram"])

        # get num of setpm instructions per 1K cycles
        vu_data.append(num_setpm_vu * 1000 / total_time_cycles)
        sram_data.append(num_setpm_sram * 1000 / total_time_cycles)
    return vu_data, sram_data


def get_data_llm_training() -> tuple[list[int | float], ...]:
    return get_stats_helper(
        LLM_MODELS,
        "training",
        batch_size=LLM_TRAINING_BATCH_SIZE,
    )


def get_data_llm_inference(prefill_or_decode: str) -> tuple[list[int | float], ...]:
    return get_stats_helper(
        LLM_MODELS,
        "inference",
        prefill_or_decode=prefill_or_decode,
    )


def get_data_dlrm_inference() -> tuple[list[int | float], ...]:
    return get_stats_helper(
        DLRM_MODELS,
        "inference",
    )


def get_data_sd_inference() -> tuple[list[int | float], ...]:
    return get_stats_helper(
        SD_MODELS,
        "inference",
    )


def plot_data(
    ax: Axes,
    all_data: Sequence[Sequence[Sequence[float]]],
    x_names: Sequence[str],
    y_label_left: str | None,
    y_label_right: str | None,
    title: str,
    y_lim_left: None | tuple[None | float, ...],
    y_lim_right: None | tuple[None | float, ...],
    log_scale: bool = False,
    plot_legend: bool = False,
):
    (
        vu_data, sram_data
    ) = all_data

    vu_data = np.array(vu_data)
    sram_data = np.array(sram_data)

    # if log_scale:
    #     ax.set_yscale( 'log' )

    ax2 = ax.twinx()

    # ax.set_yscale('log')
    # ax2.set_yscale('log')

    XValues = np.arange( len( vu_data ) ) * XTickFactor
    XTicks = XValues
    ax.set_xticks( XTicks )
    ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )
    ax.xaxis.set_ticks_position( 'none' )

    # ax.yaxis.set_ticks_position( 'none' )
    # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # if y_label:
    #     ax.set_yticklabels(
    #         ["0", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
    #         fontsize=fontsize_yticklabel,
    #     )
    # else:
    #     ax.set_yticklabels("")

    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=1, labelsize=fontsize_yticklabel)
    ax2.tick_params(axis="y", which="major", pad=1, labelsize=fontsize_yticklabel)

    ax.set_axisbelow(True)
    ax.yaxis.grid(which='major', color='lightgray', linestyle='solid')
    ax.yaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    # ax2.yaxis.grid(which='major', color='lightgray', linestyle='solid')
    # ax2.yaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    if y_label_left:
        ax.set_ylabel( y_label_left, fontsize=fontsize_ylabel )
    if y_label_right:
        ax2.set_ylabel( y_label_right, fontsize=fontsize_ylabel )

    if y_label_left:
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_yticklabels(["0", "10", "20", "30", "40"], fontsize=fontsize_yticklabel)
    else:
        ax.yaxis.set_ticks_position( 'none' )
        ax.set_yticklabels("")
    if y_label_right:
        ax2.set_yticks([0, 0.025, 0.050, 0.075, 0.1])
        ax2.set_yticklabels(["0", "0.025", "0.050", "0.075", "0.1"], fontsize=fontsize_yticklabel)
    else:
        ax2.yaxis.set_ticks_position( 'none' )
        ax2.set_yticklabels("")


    ax.set_title( title, fontsize=fontsize_ylabel )

    color1, color2, color3, color4 = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    color1_light, color2_light, color3_light, color4_light = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    legend_names = ["VU", "SRAM"]
    colors = ["#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA", "#00AAAA"]
    hatches = ["", "--", "//", "\\\\", "x"]

    edgecolor = 'white' # '#404040'
    linewidth = 0.1

    # colors = [
    #     "#E8422A", "#E8842A", "#E8632A", "#EB4663", "#E8A22A", "#F0B59B",
    #     "lightgrey", "#2A8FE8", "#2ACCE8", "#2AE885", "#2A53E8", "#B9E3EA",
    # ]
    hatches = [
        "", "//", "\\\\", "-", "//", "//",
        "",
    ]

    # plot VU data on the left Y axis, SRAM data on the right Y axis
    # use lines and dots
    # X axis is the xnames
    ax.plot(
        XValues, vu_data, label="VU", color=color1, linestyle='-', marker='o', markersize=6,
    )
    ax2.plot(
        XValues, sram_data, label="SRAM", color=color2, linestyle='--', marker='s', markersize=6,
    )

    # if plot_legend:
    #     ax.legend(
    #         frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.15),
    #         ncol=2, fontsize=fontsize_legend, edgecolor="black",
    #     )

    # def plot_bar(xvals, yvals_stack, BarWidth, plot_label, **kwargs):
    #     bar_containers = []
    #     for i, yvals in enumerate(yvals_stack):
    #         if plot_label:
    #             bar_containers.append(
    #                 ax.bar(
    #                     xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
    #                     **kwargs,
    #                 )
    #             )
    #         else:
    #             bar_containers.append(
    #                 ax.bar(
    #                     xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
    #                     **kwargs,
    #                 )
    #             )
    #     return bar_containers

    # num_bars = len(operational)
    # for i in range(num_bars):
    #     xvals = XValues-BarWidth*(num_bars/2-i-0.5)
    #     yvals_stack = [
    #         operational[i]
    #     ]
    #     plot_label = True if i == 0 else False
    #     plot_bar(
    #         xvals, yvals_stack, BarWidth, plot_label,
    #         edgecolor=edgecolor, linewidth=linewidth, label=legend_names[i],
    #         color=colors[i], hatch=hatches[i],
    #     )

    ax.set_xlim( ( XValues[0]-6*BarWidth/2, XValues[-1]+6*BarWidth/2) )
    def set_ylim(ax: Axes, y_lim: tuple[None | float, ...]):
        default_ylim = ax.get_ylim()
        new_ylim = (
            default_ylim[0] if y_lim[0] is None else y_lim[0],
            default_ylim[1] if y_lim[1] is None else y_lim[1],
        )
        ax.set_ylim( new_ylim )
    set_ylim(ax, y_lim_left or (None, None))
    set_ylim(ax2, y_lim_right or (None, None))

    if plot_legend:
        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        lg=ax.legend(
            flip(handles, 5), flip(labels, 5),
            prop={'size': fontsize_legend}, ncol=5, loc="upper center", borderaxespad=0.,
            bbox_to_anchor=(2.6, 1.42), frameon=True,
        )
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

    all_YLabels_left = [
        "# of setpm per\n 1K Cycles (VU)",
        None,
        None,
        None,
        None,
    ]
    all_YLabels_right = [
        None,
        None,
        None,
        None,
        "# of setpm per\n 1K Cycles (SRAM)",
    ]

    all_Titles = [
        "LLM Training",
        "LLM Prefill",
        "LLM Decode",
        "DLRM Inference",
        "Stable Diffusion",
    ]

    all_ylims_left = [
        (0, 40),
        (0, 40),
        (0, 40),
        (0, 40),
        (0, 40),
    ]
    all_ylims_right = [
        (0, 0.1),
        (0, 0.1),
        (0, 0.1),
        (0, 0.1),
        (0, 0.1),
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

    Figure = PyPlot.figure( figsize=(12, 2.2) )
    pdf_filename = f"outputs/eval_num_setpm.pdf"
    PDF = PdfPages( pdf_filename )

    graphs = Figure.subplots( nrows=NROWS, ncols=NCOLS )

    for idx, (xnames, data, ylabel_left, ylabel_right, title, ylim_left, ylim_right, logy) in enumerate(zip(
        all_XNames, all_data, all_YLabels_left, all_YLabels_right, all_Titles, all_ylims_left, all_ylims_right, all_logy
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[idx]
        plot_legend = (idx == 0)
        plot_data(Graph, data, xnames, ylabel_left, ylabel_right, title, ylim_left, ylim_right, logy, plot_legend)

    Figure.tight_layout()
    Figure.subplots_adjust(wspace=0.04)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
