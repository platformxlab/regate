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
fontsize_ylabel = 15
fontsize_legend = 14
fontsize_barlabel = 10

# bar width and x tick scaling factor
BarWidth = 0.4
XTickFactor = 2.5


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
) -> tuple[list[list[int | float]], ...]:
    '''
    @return (embodied[pg_strategy][model], operational[pg_strategy][model])
    '''
    embodied_carbon = [[] for _ in pg_strategies]
    operational_carbon = [[] for _ in pg_strategies]
    for im, model in enumerate(models):
        for iv, pg in enumerate(pg_strategies):
            stats, slo_scale = get_optimal_energy_stat(
                model, __NPU_VERSION.value, workload, pg, batch_size, prefill_or_decode
            )
            metric, min_max_metric = results_lib.get_energy_eff_metric_name_and_min_max(
                model, workload, prefill_or_decode
            )
            slowdown = stats["energy_stats"][pg]["component_stats"]["slowdown"]
            total_carbon = 1 / stats["carbon_and_energy_stats"]["1"][metric] * slowdown
            # value = stats["energy_stats"][pg]["total_energy_J"]
            # values[iv].append(value)
            embodied_percentage = stats["carbon_and_energy_stats"]["1"]["avg_embodied_carbon_percentage"]
            embodied = total_carbon * embodied_percentage
            operational = total_carbon * (1 - embodied_percentage)
            embodied_carbon[iv].append(embodied)
            operational_carbon[iv].append(operational)
    return embodied_carbon, operational_carbon


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
):
    (
        embodied, operational
    ) = all_data

    embodied = np.array(embodied)
    operational = np.array(operational)

    operational = operational / operational[0]
    operational = 1 - operational
    operational = operational[1:]  # do not plot NoPG

    # # normalize to NoPG
    # static_sa = np.array(static_sa)
    # static_vu = np.array(static_vu)
    # static_sram = np.array(static_sram)
    # static_ici = np.array(static_ici)
    # static_hbm = np.array(static_hbm)
    # static_other = np.array(static_other)
    # dynamic = np.array(dynamic)

    # total_energy = static_sa + static_vu + static_sram + static_ici + static_hbm + static_other + dynamic
    # total_energy_ratio = total_energy / total_energy[0]
    # static_sa = static_sa / total_energy * total_energy_ratio
    # static_vu = static_vu / total_energy * total_energy_ratio
    # static_sram = static_sram / total_energy * total_energy_ratio
    # static_ici = static_ici / total_energy * total_energy_ratio
    # static_hbm = static_hbm / total_energy * total_energy_ratio
    # static_other = static_other / total_energy * total_energy_ratio
    # dynamic = dynamic / total_energy * total_energy_ratio


    print(f"################### {title} ####################")
    print(operational)

    if log_scale:
        ax.set_yscale( 'log' )

    XValues = np.arange( len( operational[ 0 ] ) ) * XTickFactor
    XTicks = XValues
    ax.set_xticks( XTicks )
    ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )
    ax.xaxis.set_ticks_position( 'none' )

    ax.yaxis.set_ticks_position( 'none' )
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if y_label:
        ax.set_yticklabels(
            ["0", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
            fontsize=fontsize_yticklabel,
        )
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

    ax.set_title( title, fontsize=fontsize_ylabel )

    color1, color2, color3, color4 = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    color1_light, color2_light, color3_light, color4_light = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    legend_names = __PG_STRATEGIES.value[1:]  # skip NoPG
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

    def plot_bar(xvals, yvals_stack, BarWidth, plot_label, **kwargs):
        bar_containers = []
        for i, yvals in enumerate(yvals_stack):
            if plot_label:
                bar_containers.append(
                    ax.bar(
                        xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
                        **kwargs,
                    )
                )
            else:
                bar_containers.append(
                    ax.bar(
                        xvals, yvals, BarWidth, bottom=sum(yvals_stack[:i]),
                        **kwargs,
                    )
                )
        return bar_containers

    num_bars = len(operational)
    for i in range(num_bars):
        xvals = XValues-BarWidth*(num_bars/2-i-0.5)
        yvals_stack = [
            operational[i]
        ]
        plot_label = True if i == 0 else False
        plot_bar(
            xvals, yvals_stack, BarWidth, plot_label,
            edgecolor=edgecolor, linewidth=linewidth, label=legend_names[i],
            color=colors[i], hatch=hatches[i],
        )

    ax.set_xlim( ( XValues[0]-6*BarWidth/2, XValues[-1]+6*BarWidth/2) )
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

    all_YLabels = [
        "Operational\nCarbon Reduction",
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
        (0, 0.7),
        (0, 0.7),
        (0, 0.7),
        (0, 0.7),
        (0, 0.7),
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

    Figure = PyPlot.figure( figsize=(12, 2) )
    pdf_filename = f"outputs/eval_carbon.pdf"
    PDF = PdfPages( pdf_filename )

    graphs = Figure.subplots( nrows=NROWS, ncols=NCOLS )

    for idx, (xnames, data, ylabel, title, ylim, logy) in enumerate(zip(
        all_XNames, all_data, all_YLabels, all_Titles, all_ylims, all_logy
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[idx]
        plot_legend = (idx == 0)
        plot_data(Graph, data, xnames, ylabel, title, ylim, logy, plot_legend)

    Figure.tight_layout()
    Figure.subplots_adjust(wspace=0.04)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
