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

# LLM_MODELS = ["llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b"]
# LLM_MODEL_NAMES = [
#     "Llama3\n8B", "Llama2\n13B", "Llama3\n70B", "Llama3.1\n405B"
# ]
LLM_MODELS = ["llama3-70b", "llama3_1-405b"]
LLM_MODEL_NAMES = [
    "Llama3\n70B", "Llama3.1\n405B"
]
LLM_TRAINING_BATCH_SIZE = 32
# DLRM_MODELS = ["dlrm-s", "dlrm-m", "dlrm-l"]
# DLRM_MODEL_NAMES = ["DLRM-S", "DLRM-M", "DLRM-L"]
DLRM_MODELS = ["dlrm-m", "dlrm-l"]
DLRM_MODEL_NAMES = ["DLRM-M", "DLRM-L"]
SD_MODELS = ["dit-xl", "gligen"]
SD_MODEL_NAMES = ["DiT-XL", "GLIGEN"]
NPU_VERSIONS = ["2", "3", "4", "5p"]

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
    logging.info("Optimal chip config for %s %s %s %d %s: (dp-tp-pp-dpdcn-tpdcn-ppdcn-bs) %s, SLO scale: %d",
                 model, v, workload, batch_size, prefill_or_decode, pconfig, slo_scale)
    return results_lib.get_stats(
        model, v, workload, *pconfig, prefill_or_decode=prefill_or_decode
    ), slo_scale


def get_energy_eff_llm_training() -> tuple[list[list[float]], ...]:
    '''
    @return[0]: list of joule per iteration per layer for each model for each NPU version
    @return[1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    energy_eff = [[] for _ in NPU_VERSIONS]
    slo_scales = [[] for _ in NPU_VERSIONS]
    for im, model in enumerate(LLM_MODELS):
        for iv, v in enumerate(NPU_VERSIONS):
            stats, slo_scale = get_optimal_energy_stat(
                model, v, "training", LLM_TRAINING_BATCH_SIZE
            )
            value = stats["energy_stats"][__PG_STRATEGY.value]["component_stats"]["sa_temp_util"]
            # num_layers = stats["sim_config"]["num_layers"]
            if slo_scale == -1:
                energy_eff[iv].append(0)
            else:
                energy_eff[iv].append(value)
            slo_scales[iv].append(slo_scale)

    return energy_eff, slo_scales


def get_energy_eff_llm_inference(prefill_or_decode: str) -> tuple[list[list[float]], ...]:
    '''
    @return[0]: list of joule per token per layer for each model for each NPU version
    @return[1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    energy_eff = [[] for _ in NPU_VERSIONS]
    slo_scales = [[] for _ in NPU_VERSIONS]
    for im, model in enumerate(LLM_MODELS):
        for iv, v in enumerate(NPU_VERSIONS):
            stats, slo_scale = get_optimal_energy_stat(
                model, v, "inference", -1, prefill_or_decode
            )
            value = stats["energy_stats"][__PG_STRATEGY.value]["component_stats"]["sa_temp_util"]
            # num_layers = stats["sim_config"]["num_layers"]
            if slo_scale == -1:
                energy_eff[iv].append(0)
            else:
                energy_eff[iv].append(value)
            slo_scales[iv].append(slo_scale)

    return energy_eff, slo_scales


def get_energy_eff_dlrm_inference() -> tuple[list[list[float]], ...]:
    '''
    @return[0]: list of joule per request for each model for each NPU version
    @return[1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    energy_eff = [[] for _ in NPU_VERSIONS]
    slo_scales = [[] for _ in NPU_VERSIONS]
    for im, model in enumerate(DLRM_MODELS):
        for iv, v in enumerate(NPU_VERSIONS):
            stats, slo_scale = get_optimal_energy_stat(
                model, v, "inference", -1
            )
            value = stats["energy_stats"][__PG_STRATEGY.value]["component_stats"]["sa_temp_util"]
            # num_layers = stats["sim_config"]["num_layers"]
            if slo_scale == -1:
                energy_eff[iv].append(0)
            else:
                energy_eff[iv].append(value)
            slo_scales[iv].append(slo_scale)

    return energy_eff, slo_scales


def get_energy_eff_sd_inference() -> tuple[list[list[float]], ...]:
    '''
    @return[0]: list of joule per image for each model for each NPU version
    @return[1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    energy_eff = [[] for _ in NPU_VERSIONS]
    slo_scales = [[] for _ in NPU_VERSIONS]
    for im, model in enumerate(SD_MODELS):
        for iv, v in enumerate(NPU_VERSIONS):
            stats, slo_scale = get_optimal_energy_stat(
                model, v, "inference", -1
            )
            value = stats["energy_stats"][__PG_STRATEGY.value]["component_stats"]["sa_temp_util"]
            # num_layers = stats["sim_config"]["num_layers"]
            if slo_scale == -1:
                energy_eff[iv].append(0)
            else:
                energy_eff[iv].append(value)
            slo_scales[iv].append(slo_scale)

    return energy_eff, slo_scales


def plot_data(
    ax: Axes,
    data: Sequence[Sequence[float]],
    slo_data: Sequence[Sequence[float | int | str]],
    x_names: Sequence[str],
    y_label: str | None,
    title: str,
    y_lim,
    log_scale: bool = False,
):
    XValues = np.arange( len( data[ 0 ] ) ) * XTickFactor
    XTicks = XValues
    ax.set_xticks( XTicks )
    ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )
    ax.xaxis.set_ticks_position( 'none' )

    ax.yaxis.set_ticks_position( 'none' )
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if y_label:
        ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"], fontsize=fontsize_yticklabel)
    else:
        ax.set_yticklabels(["", "", "", "", "", ""])

    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=1)
    ax.tick_params(axis="y", which="major", labelsize=fontsize_yticklabel)

    ax.set_axisbelow(True)
    ax.yaxis.grid(which='major', color='lightgray', linestyle='solid')
    ax.yaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    if y_label:
        ax.set_ylabel( y_label, fontsize=fontsize_ylabel )

    ax.set_title( title, fontsize=fontsize_ylabel )

    if log_scale:
        ax.set_yscale( 'log' )

    color1, color2, color3, color4 = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    color1_light, color2_light, color3_light, color4_light = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    legend_names = ["NPU-A", "NPU-B", "NPU-C", "NPU-D"]

    edgecolor = 'white' # '#404040'
    linewidth = 0.

    YValues1 = data[0]
    ax.bar( XValues-BarWidth/2*3, YValues1, BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color1, hatch="")
    ax.bar_label(ax.containers[0], labels=[f"{round(x*100):.0f}" for x in data[0]], label_type="edge", fontsize=fontsize_barlabel)  # type: ignore

    YValues2 = data[1]
    ax.bar( XValues-BarWidth/2, YValues2, BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color2, hatch="--")
    ax.bar_label(ax.containers[1], labels=[f"{round(x*100):.0f}" for x in data[1]], label_type="edge", fontsize=fontsize_barlabel)  # type: ignore

    YValues3 = data[2]
    ax.bar( XValues+BarWidth/2, YValues3, BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color3, hatch="///")
    ax.bar_label(ax.containers[2], labels=[f"{round(x*100):.0f}" for x in data[2]], label_type="edge", fontsize=fontsize_barlabel)  # type: ignore

    YValues4 = data[3]
    ax.bar( XValues+BarWidth/2*3, YValues4, BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color4, hatch="\\\\")
    ax.bar_label(ax.containers[3], labels=[f"{round(x*100):.0f}" for x in data[3]], label_type="edge", fontsize=fontsize_barlabel)  # type: ignore

    ax.set_xlim( ( XValues[0]-7*BarWidth/2, XValues[-1]+7*BarWidth/2) )
    default_ylim = ax.get_ylim()
    new_ylim = (
        default_ylim[0] if y_lim[0] is None else y_lim[0],
        default_ylim[1] if y_lim[1] is None else y_lim[1],
    )
    ax.set_ylim( new_ylim )

    lg=ax.legend(prop={'size': fontsize_legend}, ncol=1, loc=2, borderaxespad=0.)
    lg.draw_frame(False)


def main(argv: list[str]):
    del argv  # Unused

    LLM_XNames = [
        "A B C D\n" + m
        for m in LLM_MODEL_NAMES
    ]
    DLRM_XNames = [
        "A B C D\n" + m
        for m in DLRM_MODEL_NAMES
    ]
    SD_XNames = [
        "A B C D\n" + m
        for m in SD_MODEL_NAMES
    ]

    all_data = [
        # get_energy_eff_llm_training(),
        get_energy_eff_llm_inference("prefill"),
        get_energy_eff_llm_inference("decode"),
        get_energy_eff_dlrm_inference(),
        get_energy_eff_sd_inference(),
    ]
    all_energy = [
        x for x, _ in all_data
    ]
    all_slo_scale = [
        y for _, y in all_data
    ]
    # set 1 value to empty string for SLO scale
    all_slo_scale = [
        [
            [
                "" if x == 1 else f"{x}x"
                for x in slo_scale
            ]
            for slo_scale in slo_scales
        ]
        for slo_scales in all_slo_scale
    ]

    all_XNames = [
        # LLM_XNames,
        LLM_XNames,
        LLM_XNames,
        DLRM_XNames,
        SD_XNames,
    ]

    all_YLabels = [
        # "SA Temporal Util.",
        "SA Temporal Util.",
        None,
        None,
        None,
        # "SA Temporal Util.",
        # "SA Temporal Util.",
        # "SA Temporal Util.",
        # "SA Temporal Util.",
    ]

    all_Titles = [
        # "LLM Training",
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
    all_ylims = [
        # (None, 1.1),
        (None, 1.1),
        (None, 1.1),
        (None, 1.1),
        (None, 1.1),
    ]
    all_logy = [
        # False,
        False,
        False,
        False,
        False,
    ]

    NROWS = 1
    NCOLS = 4

    Figure = PyPlot.figure( figsize=(12, 3.2) )
    pdf_filename = f"outputs/sa_temporal_utilization.pdf"
    PDF = PdfPages( pdf_filename )

    graphs = Figure.subplots( nrows=NROWS, ncols=NCOLS )

    for idx, (xnames, energy_data, slo_scale_data, ylabel, title, ylim, logy) in enumerate(zip(
        all_XNames, all_energy, all_slo_scale, all_YLabels, all_Titles, all_ylims, all_logy
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[idx]
        plot_data(Graph, energy_data, slo_scale_data, xnames, ylabel, title, ylim, logy)

    Figure.tight_layout()
    Figure.subplots_adjust(wspace=0.04)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
