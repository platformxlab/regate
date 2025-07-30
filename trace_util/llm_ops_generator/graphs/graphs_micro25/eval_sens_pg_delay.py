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
fontsize_xlabel = 15  # unused
fontsize_ylabel = 15
fontsize_legend = 14
fontsize_barlabel = 10

# bar width and x tick scaling factor
BarWidth = 0.3
XTickFactor = 2


__PG_STRATEGIES = flags.DEFINE_list(
    "pg_strategy",
    ["NoPG", "Base", "HW", "Full"],
    "Power gating strategy to use for energy analysis, \
    choose from 'NoPG', 'Base', 'HW', 'Full', 'Ideal'",
)
__PG_VARY_VERSIONS = flags.DEFINE_list(
    "pg_vary_versions",
    ["", "_vary_PG_delay_1.5", "_vary_PG_delay_2", "_vary_PG_delay_3", "_vary_PG_delay_4"],
    "Power gating variations (suffix to pg_strategy) to use for energy analysis \
    leave blank for the base PG config",
)
__NPU_VERSION = flags.DEFINE_string(
    "npu_versions",
    "5p",
    "NPU version to use for energy analysis",
)

LLM_MODELS = ["llama3_1-405b"]
LLM_MODEL_NAMES = [
    "Llama3.1-405B"
]
LLM_TRAINING_BATCH_SIZE = 32
DLRM_MODELS = ["dlrm-l"]
DLRM_MODEL_NAMES = ["DLRM-L"]
SD_MODELS = ["dit-xl"]
SD_MODEL_NAMES = ["DiT-XL"]

RESULTS_DIR = os.environ["RESULTS_DIR"]
SLO_RESULTS_DIR = os.path.join(RESULTS_DIR, "slo")

MAX_SLO_SCALE = 5

# Configs that cannot satisfy SLO within 8K chips
SLO_VIOLATION_CONFIGS = [
    # (model, v, workload, batch_size, prefill_or_decode, slo_scale)
]

XNAMES = [
    "1x",
    "1.5x",
    "2x",
    "3x",
    "4x",
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
    pg_vary_versions: Sequence[str],
    pg_strategies: Sequence[str],
    workload: str,
    prefill_or_decode: str = "",
    batch_size: int = -1,
) -> tuple[list[list[list[int | float]]], ...]:
    '''
    @return[pg_strategy][pg_vary_versions][model]
    '''
    perf_data = []
    energy_data = []
    for ipg, pg in enumerate(pg_strategies):
        perf_data.append([])
        energy_data.append([])
        for iv, pg_v in enumerate(pg_vary_versions):
            perf_data[ipg].append([])
            energy_data[ipg].append([])
            for im, model in enumerate(models):
                if pg == "NoPG":
                    pg_config = pg
                else:
                    pg_config = pg + pg_v
                stats, slo_scale = get_optimal_energy_stat(
                    model, __NPU_VERSION.value, workload, pg_config, batch_size, prefill_or_decode
                )
                perf_value = stats["energy_stats"][pg_config]["component_stats"]["total_exe_time_ns"] # ["total_energy_J"]
                perf_data[ipg][iv].append(perf_value)
                energy_value = stats["energy_stats"][pg_config]["total_energy_J"]
                energy_data[ipg][iv].append(energy_value)
    return energy_data, perf_data


def get_data_llm_training() -> tuple[list[list[list[int | float]]], ...]:
    return get_stats_helper(
        LLM_MODELS,
        __PG_VARY_VERSIONS.value,
        __PG_STRATEGIES.value,
        "training",
        batch_size=LLM_TRAINING_BATCH_SIZE,
    )


def get_data_llm_inference(prefill_or_decode: str) -> tuple[list[list[list[int | float]]], ...]:
    return get_stats_helper(
        LLM_MODELS,
        __PG_VARY_VERSIONS.value,
        __PG_STRATEGIES.value,
        "inference",
        prefill_or_decode=prefill_or_decode,
    )


def get_data_dlrm_inference() -> tuple[list[list[list[int | float]]], ...]:
    return get_stats_helper(
        DLRM_MODELS,
        __PG_VARY_VERSIONS.value,
        __PG_STRATEGIES.value,
        "inference",
    )


def get_data_sd_inference() -> tuple[list[list[list[int | float]]], ...]:
    return get_stats_helper(
        SD_MODELS,
        __PG_VARY_VERSIONS.value,
        __PG_STRATEGIES.value,
        "inference",
    )


def plot_data(
    ax: Axes,
    data: Sequence[Sequence[Sequence[float]]],
    x_names: Sequence[str],
    y_label: str | None,
    title: str | None,
    y_lim,
    log_scale: bool = False,
    plot_legend: bool = False,
    yticks: Sequence[float] | None = None,
    yticklabels: Sequence[str] | None = None,
):
    # all_data[pg_strategy][pg_vary_version][model]

    # print(f"################### {title} ####################")
    # print(data)

    if log_scale:
        ax.set_yscale( 'log' )

    XValues = np.arange( len( data[0][0] ) ) * XTickFactor
    # XTicks = XValues
    # ax.set_xticks( XTicks )
    # ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )
    # ax.xaxis.set_ticks_position( 'none' )

    ax.yaxis.set_ticks_position( 'none' )
    if yticks:
        ax.set_yticks(yticks)
    if yticklabels:
        ax.set_yticklabels(yticklabels, fontsize=fontsize_yticklabel)
    else:
        ax.set_yticklabels("")

    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=1)
    ax.tick_params(axis="y", which="major", labelsize=fontsize_yticklabel)

    ax.set_axisbelow(True)
    ax.yaxis.grid(which='major', color='lightgray', linestyle='solid')
    ax.yaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    if y_label:
        ax.set_ylabel( y_label, fontsize=fontsize_ylabel )

    if title:
        ax.set_title( title, fontsize=fontsize_ylabel )

    color1, color2, color3, color4 = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    color1_light, color2_light, color3_light, color4_light = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    legend_names = __PG_STRATEGIES.value[1:]  # skip NoPG
    colors = ["#A0B1BA", "#3C93C2", "#0000AA", "#9EC9C2", "#00AAAA"]
    hatches = ["", "--", "//", "\\\\", "x"]
    markers = ["o", "s", "*", "v", "^"]
    marker_sizes = [6, 6, 10, 6, 6]
    zorders = [1, 2, 4, 3, 5]
    alphas = [1, 1, 1, 1]

    edgecolor = 'white' # '#404040'
    linewidth = 0.1


    def plot_series(xvals, yvals_stack, plot_label, **kwargs):
        for i, yvals in enumerate(yvals_stack):
            if plot_label:
                ax.plot(
                    xvals, yvals,
                    label=legend_names[i],
                    color=colors[i],
                    marker=markers[i],
                    markersize=marker_sizes[i],
                    alpha=alphas[i],
                    zorder=zorders[i],
                    **kwargs,
                )
            else:
                ax.plot(
                    xvals, yvals,
                    label=legend_names[i],
                    color=colors[i],
                    marker=markers[i],
                    markersize=marker_sizes[i],
                    alpha=alphas[i],
                    zorder=zorders[i],
                    **kwargs,
                )

    def bar_series(xvals, yvals_stack, plot_label, **kwargs):
        for i, yvals in enumerate(yvals_stack):
            if plot_label:
                ax.bar(
                    xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
                    **kwargs,
                )
            else:
                ax.bar(
                    xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
                    **kwargs,
                )

    num_bars = len(data[0])
    num_series = len(data[0][0])
    XValues_all = []
    for i in range(num_bars):
        xvals = XValues-BarWidth*(num_bars/2-i-0.5)
        XValues_all.append(xvals)
    XValues_all = np.array(XValues_all).T

    for i in range(num_series):
        # all_data[pg_strategy][version][model]
        yvals_stack = data[:, :, i][1:]   # skip NoPG  # type: ignore
        plot_label = True if i == 0 else False
        plot_series(
            XValues_all[i], yvals_stack, plot_label,
            linewidth=linewidth,
        )
        # yvals_stack = energy_data[:, :, i][1:]  # skip NoPG
        # bar_label = True if i == 0 else False
        # bar_series(
        #     XValues_all[i], yvals_stack, bar_label,
        #     linewidth=linewidth,
        #     edgecolor=edgecolor,
        # )

    # set xticklabels
    if x_names[0] is not None:
        XTicks = np.concatenate(XValues_all)
        ax.set_xticks(XTicks)
        ax.set_xticklabels(
            XNAMES * num_series,
            fontsize=fontsize_xticklabel,
            position=(0, 0.03),
            # ha='right',
            # rotation=90,
        )
    else:
        ax.set_xticklabels("")
    ax.xaxis.set_ticks_position( 'none' )

    ax.set_xlim( ( XValues[0]-5*BarWidth/2, XValues[-1]+5*BarWidth/2) )
    default_ylim = ax.get_ylim()
    new_ylim = (
        default_ylim[0] if y_lim[0] is None else y_lim[0],
        default_ylim[1] * 1.1 if y_lim[1] is None else y_lim[1],
    )
    ax.set_ylim( new_ylim )

    # lg=ax.legend(prop={'size': fontsize_legend}, ncol=5, loc=2, borderaxespad=0.)
    # lg.draw_frame(False)
    if plot_legend:
        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])
        handles, labels = ax.get_legend_handles_labels()
        lg=ax.legend(
            flip(handles, 5), flip(labels, 5),
            prop={'size': fontsize_legend}, ncol=5, loc="upper center", borderaxespad=0.,
            bbox_to_anchor=(2.5, 1.6), frameon=True,
        )
        frame = lg.get_frame()
        frame.set_edgecolor("black")


def main(argv: list[str]):
    del argv  # Unused

    LLM_XNames = [
        f"{x}" for x in
        LLM_MODEL_NAMES
    ]

    DLRM_XNames = [
        f"{x}" for x in
        DLRM_MODEL_NAMES
    ]

    SD_XNames = [
        f"{x}" for x in
        SD_MODEL_NAMES
    ]

    all_data = [
        get_data_llm_training(),
        get_data_llm_inference("prefill"),
        get_data_llm_inference("decode"),
        get_data_dlrm_inference(),
        get_data_sd_inference(),
    ]
    all_energy_data = [
        x for x, _ in all_data
    ]
    all_perf_data = [
        x for _, x in all_data
    ]

    all_XNames = [
        LLM_XNames,
        LLM_XNames,
        LLM_XNames,
        DLRM_XNames,
        SD_XNames,
    ]

    all_YLabels = [
        "Norm. Energy\nSavings",
        None,
        None,
        None,
        None,
    ]

    all_Titles = [
        "Llama3.1-405B\nTraining",
        "Llama3.1-405B\nPrefill",
        "Llama3.1-405B\nDecode",
        "DLRM\nInference",
        "DiT-XL\nInference",
    ]

    all_ylims = [
        (0, None), # (0, 0.25),
        (0, None), # (0, None),
        (0, None), # (0, None),
        (0, None), # (0, 0.4),
        (0, None), # (0, None),
    ]
    all_logy = [
        False,
        False,
        False,
        False,
        False,
    ]

    all_yticks = [
        None,
        None,
        None,
        None,
        None,
    ]
    all_yticklabels = [
        None,
        None,
        None,
        None,
        None,
    ]
    # all_yticks = [
    #     [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    #     [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    #     [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    # ]
    # all_yticklabels = [
    #     ["0", "5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%"],
    #     ["0", "5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%"],
    #     ["0", "10%", "20%", "30%", "40%", "50%"],
    #     ["0", "10%", "20%", "30%", "40%", "50%"],
    #     ["0", "5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%"],
    # ]

    NROWS = 2
    NCOLS = 5

    Figure = PyPlot.figure( figsize=(12, 4) )
    pdf_filename = f"outputs/eval_sens_pg_delay.pdf"
    PDF = PdfPages( pdf_filename )

    graphs = Figure.subplots( nrows=NROWS, ncols=NCOLS )

    all_ylims = [
        (0, 0.4), # (0, 0.25),
        (0, 0.4), # (0, None),
        (0, 0.4), # (0, None),
        (0, 0.4), # (0, 0.4),
        (0, 0.4), # (0, None),
    ]
    all_yticks = [
        [0, 0.1, 0.2, 0.3, 0.4]
    ]  * 5
    all_yticklabels = [
        ["0", "10%", "20%", "30%", "40%"]
    ] + [None] * 4
    for idx, (xnames, data, ylabel, title, ylim, logy, yticks, yticklabels) in enumerate(zip(
        all_XNames, all_energy_data, all_YLabels, all_Titles, all_ylims, all_logy, all_yticks, all_yticklabels
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[0][idx]
        plot_legend = (idx == 0)
        data = np.array(data)
        data = data / data[0]  # normalize to NoPG
        data = 1 - data
        plot_data(Graph, data, [None], ylabel, title, ylim, logy, plot_legend, yticks, yticklabels)  # type: ignore

    all_YLabels[0] = "Performance\nOverhead"
    all_ylims = [
        (1e-5, 0.3), # (0, 0.25),
        (1e-5, 0.3), # (0, None),
        (1e-5, 0.3), # (0, None),
        (1e-5, 0.3), # (0, 0.4),
        (1e-5, 0.3), # (0, None),
    ]
    all_yticks = [
        # [0.01, 0.02, 0.05, 0.1, 0.2]
        [0.0001, 0.001, 0.01, 0.05, 0.2]
    ] * 5
    all_yticklabels = [
        # ["1%", "2%", "5%", "10%", "20%"]
        ["0.01%", "0.1%", "1%", "5%", "20%"]
    ] + [None] * 4
    all_logy = [
        True,
        True,
        True,
        True,
        True,
    ]
    for idx, (xnames, data, ylabel, title, ylim, logy, yticks, yticklabels) in enumerate(zip(
        all_XNames, all_perf_data, all_YLabels, all_Titles, all_ylims, all_logy, all_yticks, all_yticklabels
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[1][idx]
        plot_legend = False
        data = np.array(data)
        data = data / data[0]  # normalize to NoPG
        data = data - 1
        plot_data(Graph, data, xnames, ylabel, None, ylim, logy, plot_legend, yticks, yticklabels)

    Figure.text(
        0.5, -0.01, "Power-Gate & Wake-up Delay", ha="center", fontsize=fontsize_xlabel
    )
    Figure.tight_layout()
    Figure.subplots_adjust(wspace=0.04, hspace=0.06)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
