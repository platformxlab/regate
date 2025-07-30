#!/usr/bin/env python3

import itertools
from turtle import st
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
PlotLib.rcParams['hatch.color'] = 'black'

# font sizes
fontsize_xticklabel = 14
fontsize_yticklabel = 14
fontsize_xlabel = 16  # unused
fontsize_ylabel = 16
fontsize_legend = 14
fontsize_barlabel = 10

# bar width and x tick scaling factor
BarWidth = 0.4
XTickFactor = 2.2

__OUT_FILENAME = flags.DEFINE_string(
    "out_filename",
    "eval_energy_breakdown.pdf",
    "Output filename for the energy breakdown evaluation plot",
)

__PG_STRATEGIES = flags.DEFINE_list(
    "pg_strategy",
    ["NoPG", "Base", "HW", "Full", "Ideal"],
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

MAX_SLO_SCALE = 5

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
    print(f"####################### {model} {v} {workload} {prefill_or_decode}: {pstr}")
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
) -> tuple[list[list[int | float]], ...]:

    '''
    @return ({static_component}[pg_strategy][model], dynamic[pg_strategy][model])
    '''
    print(f"PG Strategies: {pg_strategies}")
    static_sa = [[] for _ in pg_strategies]
    static_vu = [[] for _ in pg_strategies]
    static_sram = [[] for _ in pg_strategies]
    static_ici = [[] for _ in pg_strategies]
    static_hbm = [[] for _ in pg_strategies]
    static_other = [[] for _ in pg_strategies]
    dynamic = [[] for _ in pg_strategies]
    for im, model in enumerate(models):
        for iv, pg in enumerate(pg_strategies):
            stats, slo_scale = get_optimal_energy_stat(
                model, __NPU_VERSION.value, workload, pg, batch_size, prefill_or_decode
            )
            # metric, min_max_metric = results_lib.get_energy_eff_metric_name_and_min_max(
            #     model, workload, prefill_or_decode
            # )
            # value = 1 / stats["carbon_and_energy_stats"]["1"][metric]
            # value = stats["energy_stats"][pg]["total_energy_J"]
            # values[iv].append(value)
            static_sa[iv].append(stats["energy_stats"][pg]["static_sa_energy_J"])
            static_vu[iv].append(stats["energy_stats"][pg]["static_vu_energy_J"])
            static_sram[iv].append(stats["energy_stats"][pg]["static_sram_energy_J"])
            static_ici[iv].append(stats["energy_stats"][pg]["static_ici_energy_J"])
            static_hbm[iv].append(stats["energy_stats"][pg]["static_hbm_energy_J"])
            static_other[iv].append(stats["energy_stats"][pg]["static_other_energy_J"])
            dynamic[iv].append(stats["energy_stats"][pg]["total_dynamic_energy_J"])
    return (
        static_sa, static_vu, static_sram, static_ici, static_hbm, static_other,
        dynamic,
    )


def get_data_llm_training() -> tuple[list[list[int | float]], ...]:
    return get_stats_helper(
        LLM_MODELS,
        __PG_STRATEGIES.value,
        "training",
        batch_size=LLM_TRAINING_BATCH_SIZE,
    )


def get_data_llm_inference(prefill_or_decode: str) -> tuple[list[list[int | float]], ...]:
    return get_stats_helper(
        LLM_MODELS,
        __PG_STRATEGIES.value,
        "inference",
        prefill_or_decode=prefill_or_decode,
    )


def get_data_dlrm_inference() -> tuple[list[list[int | float]], ...]:
    return get_stats_helper(
        DLRM_MODELS,
        __PG_STRATEGIES.value,
        "inference",
    )


def get_data_sd_inference() -> tuple[list[list[int | float]], ...]:
    return get_stats_helper(
        SD_MODELS,
        __PG_STRATEGIES.value,
        "inference",
    )


def plot_data(
    ax: Axes,
    all_data: Sequence[Sequence[Sequence[float]]],
    x_names: Sequence[str],
    y_label: str | None,
    title: str,
    y_lim,
    log_scale: bool = False,
    plot_legend: bool = False,
    yticks: Sequence[float] | None = None,
    yticklabels: Sequence[str] | None = None,
):
    (
        static_sa,
        static_vu,
        static_sram,
        static_ici,
        static_hbm,
        static_other,
        dynamic,
    ) = all_data

    # normalize to NoPG
    static_sa = np.array(static_sa)
    static_vu = np.array(static_vu)
    static_sram = np.array(static_sram)
    static_ici = np.array(static_ici)
    static_hbm = np.array(static_hbm)
    static_other = np.array(static_other)
    dynamic = np.array(dynamic)

    total_energy = static_sa + static_vu + static_sram + static_ici + static_hbm + static_other + dynamic
    total_energy_ratio = total_energy / total_energy[0]
    static_sa = static_sa / total_energy * total_energy_ratio
    static_vu = static_vu / total_energy * total_energy_ratio
    static_sram = static_sram / total_energy * total_energy_ratio
    static_ici = static_ici / total_energy * total_energy_ratio
    static_hbm = static_hbm / total_energy * total_energy_ratio
    static_other = static_other / total_energy * total_energy_ratio
    dynamic = dynamic / total_energy * total_energy_ratio
    total_energy = total_energy / total_energy * total_energy_ratio

    static_sa = static_sa[0] - static_sa
    static_vu = static_vu[0] - static_vu
    static_sram = static_sram[0] - static_sram
    static_ici = static_ici[0] - static_ici
    static_hbm = static_hbm[0] - static_hbm
    static_other = static_other[0] - static_other
    dynamic = dynamic[0] - dynamic
    total_energy = total_energy[0] - total_energy

    # do not plot NoPG. 0: nopg, 1: base, 2: hw, 3: full, 4: ideal
    static_sa = static_sa[1:]
    static_vu = static_vu[1:]
    static_sram = static_sram[1:]
    static_ici = static_ici[1:]
    static_hbm = static_hbm[1:]
    static_other = static_other[1:]
    dynamic = dynamic[1:]
    total_energy = total_energy[1:]

    print(f"################### {title} (static SA) ####################")
    print(static_sa)
    print(f"################### {title} (static VU) ####################")
    print(static_vu)
    print(f"################### {title} (static SRAM) ####################")
    print(static_sram)
    print(f"################### {title} (static ICI) ####################")
    print(static_ici)
    print(f"################### {title} (static HBM) ####################")
    print(static_hbm)
    print(f"################### {title} (static Other) ####################")
    print(static_other)

    print(f"################### {title} (total (sum all static)) ####################")
    print(total_energy)

    # print(f"################### {title} (dynamic Full) ####################")
    # print(dynamic)
    # print(f"################### {title} (total) ####################")
    # print(total_energy)
    # print(f"################### {title} (total: Ideal - Full) ####################")
    # print(total_energy[-1] - total_energy[-2])

    # print(f"################### {title} (vu) ####################")
    # print(static_vu)
    # print(f"################### {title} (sram) ####################")
    # print(static_sram)

    if log_scale:
        ax.set_yscale( 'log' )

    XValues = np.arange( len( static_sa[ 0 ] ) ) * XTickFactor
    XTicks = XValues
    ax.set_xticks( XTicks )
    ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )
    ax.xaxis.set_ticks_position( 'none' )

    ax.yaxis.set_ticks_position( 'none' )
    if yticks and yticklabels:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=fontsize_yticklabel)
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # if y_label:
    #     ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"], fontsize=fontsize_yticklabel)
    # else:
    #     ax.set_yticklabels(["", "", "", "", "", ""])

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
    legend_names = __PG_STRATEGIES.value
    # colors = ["#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA", "#00AAAA"]
    # hatches = ["", "--", "//", "\\\\", "x"]

    edgecolor = 'black' # '#404040'
    linewidth = 0.1

    colors = [
        "#F8523A", "#F8943A", "#F8733A", "#FB5673", "#F8B23A", "#FF95AB",
        "lightgrey", "#3A9FF8", "#3ADCF8", "#3AF895", "#3A63F8", "#C9F3FA",
    ]
    hatches = [
        "//", "", "\\\\", "x", "", "//", ""
    ]

    def plot_bar(xvals, yvals_stack, BarWidth, plot_label, **kwargs):
        bar_containers = []
        legend_names = ["SA", "VU", "SRAM", "ICI", "HBM", "Other", "Dynamic"]
        for i, yvals in enumerate(yvals_stack):
            if plot_label:
                bar_containers.append(
                    ax.bar(
                        xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
                        color=colors[i], hatch=hatches[i], label=legend_names[i],
                        **kwargs,
                    )
                )
            else:
                bar_containers.append(
                    ax.bar(
                        xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
                        color=colors[i], hatch=hatches[i],
                        **kwargs,
                    )
                )
        return bar_containers

    barlabel_offsets = [0, -0.3, -0.8, -1.2]
    num_bars = len(static_sa)
    for i in range(num_bars):
        xvals = XValues-BarWidth*(num_bars/2-i-0.5)
        yvals_stack = [
            static_sa[i], static_vu[i], static_sram[i], static_ici[i], static_hbm[i],
        ]
        plot_label = True if i == 0 else False
        plot_bar(
            xvals, yvals_stack, BarWidth, plot_label, edgecolor=edgecolor, linewidth=linewidth
        )
        if title == "LLM Decode":
            ax.text(
                xvals[i] + barlabel_offsets[i], total_energy[-2][i] * 1.03,
                f"{np.round(total_energy[-2][i] * 100, 1)}%",
                fontsize=fontsize_xticklabel,
            )
    bar_label_position = (0, 2)
    if title == "LLM Decode":
        pass
    else:
        ax.bar_label(
            ax.containers[-len(yvals_stack)-1],  # type: ignore
            labels=[f"{x:.1f}%" for x in np.round(total_energy[-2] * 100, 1)],
            label_type="edge",
            fontsize=fontsize_xticklabel,
            position=bar_label_position,
        )

    ax.set_xlim( ( XValues[0]-7.5*BarWidth/2, XValues[-1]+8*BarWidth/2) )
    default_ylim = ax.get_ylim()
    new_ylim = (
        default_ylim[0] if y_lim[0] is None else y_lim[0],
        default_ylim[1] * 1.06 if y_lim[1] is None else y_lim[1],
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
            bbox_to_anchor=(2.9, 1.23), frameon=False,
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
        "Energy Savings Breakdown",
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
        (0, None),
        (0, 0.15),
        (0, 0.24),
        (0, None),
        (0, None),
    ]
    all_logy = [
        False,
        False,
        False,
        False,
        False,
    ]

    all_yticks = [
        [0, 0.05, 0.1, 0.15, 0.2, 0.25],
        [0, 0.05, 0.1, 0.15, 0.2, 0.25],
        [0, 0.05, 0.1, 0.15, 0.2, 0.25],
        [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    ]
    all_yticklabels = [
        ["0", "5%", "10%", "15%", "20%", "25%"],
        ["0", "5%", "10%", "15%", "20%", "25%"],
        ["0", "5%", "10%", "15%", "20%", "25%"],
        ["0", "5%", "10%", "15%", "20%", "25%", "30%", "35%"],
        ["0", "5%", "10%", "15%", "20%", "25%", "30%", "35%"],
    ]
    # all_bar_label_offset_xy = [
    #     (0, 2),
    #     (0, -2),
    #     (0, 5),
    #     (0, 2),
    #     (0, 2),
    # ]

    NROWS = 1
    NCOLS = 5

    Figure = PyPlot.figure( figsize=(24, 4) )
    pdf_filename = f"outputs/{__OUT_FILENAME.value}"
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
