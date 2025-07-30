#!/usr/bin/env python3

from functools import lru_cache
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
XTickFactor = 5


__PG_STRATEGY = flags.DEFINE_list(
    "pg_strategy",
    ["NoPG", "Full"],
    "Power gating strategy to use for energy analysis, \
    choose from 'NoPG', 'Base', 'HW', 'Full', 'Ideal'",
)

# LLM_MODELS = ["llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b"]
# LLM_MODEL_NAMES = [
#     "Llama3\n8B", "Llama2\n13B", "Llama3\n70B", "Llama3.1\n405B"
# ]
LLM_MODELS = ["llama3_1-405b"]
LLM_MODEL_NAMES = [
    "Llama3.1-405B"
]
LLM_TRAINING_BATCH_SIZE = 32
# DLRM_MODELS = ["dlrm-s", "dlrm-m", "dlrm-l"]
# DLRM_MODEL_NAMES = ["DLRM-S", "DLRM-M", "DLRM-L"]
DLRM_MODELS = ["dlrm-l"]
DLRM_MODEL_NAMES = ["DLRM-L"]
SD_MODELS = ["dit-xl"]
SD_MODEL_NAMES = ["DiT-XL"]

__NPU_VERSION = flags.DEFINE_list(
    "npu_version",
    ["4", "5p"],
    "Two NPU versions to use for deriving energy efficiency improvement ratio."
)
__LIFETIME = flags.DEFINE_list(
    "lifetime",
    [str(x) for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
    "Lifetime of the chip in years to analyze.",
)
__PERIOD = flags.DEFINE_integer(
    "period",
    10,
    "Tiem period in years to analyze.",
)

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


@lru_cache(maxsize=None)
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


def get_data_helper(
    models: Sequence[str],
    workload: str,
    versions: Sequence[str],
    batch_size: int = -1,
    prefill_or_decode: str = "",
) -> tuple[list[list[float | int]], ...]:
    '''
    @return: (embodied_carbon[model][year], operational_carbon_disabled[model][year], operational_carbon_ideal[model][year])
    '''
    embodied_carbon = [[] for _ in range(len(models))]
    operational_carbon_disabled = [[] for _ in range(len(models))]
    operational_carbon_ideal = [[] for _ in range(len(models))]

    for im, model in enumerate(models):
        for year in __LIFETIME.value:
            def get_embodied_carbon() -> float:
                '''embodied carbon: kgCO2e per yield'''
                stats, _ = get_optimal_energy_stat(
                    model, __NPU_VERSION.value[1], workload, __PG_STRATEGY.value[0], batch_size, prefill_or_decode
                )
                metric, min_max_metric = results_lib.get_carbon_eff_metric_name_and_min_max(
                    model, workload, prefill_or_decode
                )
                embodied_percentage = stats["carbon_and_energy_stats"]["1"]["avg_embodied_carbon_percentage"]
                total_carbon = 1 / stats["carbon_and_energy_stats"]["1"][metric]
                em_carbon_year_1 = total_carbon * embodied_percentage
                total_em_carbon = em_carbon_year_1 * (__PERIOD.value / int(year))
                return total_em_carbon

            def get_operational_carbon(pg_strategy: str) -> float:
                stats1, _ = get_optimal_energy_stat(
                    model, versions[0], workload, pg_strategy, batch_size, prefill_or_decode
                )
                stats2, _ = get_optimal_energy_stat(
                    model, versions[1], workload, pg_strategy, batch_size, prefill_or_decode
                )
                metric, min_max_metric = results_lib.get_carbon_eff_metric_name_and_min_max(
                    model, workload, prefill_or_decode
                )
                embodied_percentage1 = stats1["carbon_and_energy_stats"]["1"]["avg_embodied_carbon_percentage"]
                total_carbon1 = 1 / stats1["carbon_and_energy_stats"]["1"][metric]
                embodied_percentage2 = stats2["carbon_and_energy_stats"]["1"]["avg_embodied_carbon_percentage"]
                total_carbon2 = 1 / stats2["carbon_and_energy_stats"]["1"][metric]
                operation_carbon_per_year1 = total_carbon1 * (1 - embodied_percentage1)
                operation_carbon_per_year2 = total_carbon2 * (1 - embodied_percentage2)
                operation_carbon_reduction_ratio = operation_carbon_per_year2 / operation_carbon_per_year1
                # print(f"operational carbon ratio {model} {workload} {prefill_or_decode} {pg_strategy}: {operation_carbon_reduction_ratio}")
                # print(f"operation_carbon_per_year {versions[0]} {model} {workload} {prefill_or_decode} {pg_strategy}: {operation_carbon_per_year1}")
                # print(f"operation_carbon_per_year {versions[1]} {model} {workload} {prefill_or_decode} {pg_strategy}: {operation_carbon_per_year2}")
                # print()
                assert 0 < operation_carbon_reduction_ratio < 1, f"op_carbon_ratio={operation_carbon_reduction_ratio}"

                total_op_carbon = 0
                op_carbon_per_year = operation_carbon_per_year1
                for i in range(1, __PERIOD.value + 1):
                    total_op_carbon += op_carbon_per_year
                    if i % int(year) == 0:
                        op_carbon_per_year *= (operation_carbon_reduction_ratio ** int(year))
                return total_op_carbon

            embodied_carbon[im].append(
                get_embodied_carbon()
            )
            operational_carbon_disabled[im].append(
                get_operational_carbon(__PG_STRATEGY.value[0])
            )
            operational_carbon_ideal[im].append(
                get_operational_carbon(__PG_STRATEGY.value[1])
            )

    return embodied_carbon, operational_carbon_disabled, operational_carbon_ideal


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
    embodied_data: Sequence[Sequence[float | int]] | np.ndarray,
    op_disabled_data: Sequence[Sequence[float | int]] | np.ndarray,
    op_ideal_data: Sequence[Sequence[float | int]] | np.ndarray,
    x_names: Sequence[str],
    y_label: str | None,
    title: str,
    y_lim,
    log_scale: bool = False,
    plot_legend: bool = False,
    yticks: Sequence[float] | None = None,
    yticklabels: Sequence[str] | None = None,
):
    if not isinstance(embodied_data, np.ndarray):
        embodied_data = np.array(embodied_data)
    if not isinstance(op_disabled_data, np.ndarray):
        op_disabled_data = np.array(op_disabled_data)
    if not isinstance(op_ideal_data, np.ndarray):
        op_ideal_data = np.array(op_ideal_data)

    # if log_scale:
    #     ax.set_yscale( 'log' )

    XValues = np.arange( len( embodied_data[ 0 ] ) ) * XTickFactor
    XTicks = XValues
    ax.set_xticks( XTicks )
    ax.xaxis.set_ticks_position( 'none' )
    ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )

    ax.yaxis.set_ticks_position( 'none' )
    if yticks:
        ax.set_yticks(yticks)
    if yticklabels:
        ax.set_yticklabels(yticklabels, fontsize=fontsize_yticklabel)
    ax.minorticks_off()


    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="y", which="major", pad=1)
    ax.tick_params(axis="y", which="major", labelsize=fontsize_yticklabel)

    ax.set_axisbelow(True)
    ax.yaxis.grid(which='major', color='lightgray', linestyle='solid')
    ax.yaxis.grid(which='minor', color='#EEEEEE', linestyle='solid')
    if y_label:
        ax.set_ylabel( y_label, fontsize=fontsize_ylabel )
    ax.set_xlabel("Device Lifespan (Years)", fontsize=fontsize_xlabel)

    ax.set_title( title, fontsize=fontsize_ylabel )

    color1, color2, color3, color4 = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    color1_light, color2_light, color3_light, color4_light = "#A0B1BA", "#3C93C2", "#9EC9C2", "#0000AA"
    legend_names = ["NPU-A", "NPU-B", "NPU-C", "NPU-D"]

    edgecolor = 'black' # '#404040'
    linewidth = 0.1

    # ax.bar( XValues-BarWidth/2*3, embodied_data[0], BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color1, hatch="--")
    # ax.bar( XValues-BarWidth/2*3, op_disabled_data[0], BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color1_light, hatch="", bottom=embodied_data[0])

    # YValues2 = embodied_data[1]
    # ax.bar( XValues-BarWidth/2, YValues2, BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color2, hatch="--")

    # YValues3 = embodied_data[2]
    # ax.bar( XValues+BarWidth/2, YValues3, BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color3, hatch="///")

    # YValues4 = embodied_data[3]
    # ax.bar( XValues+BarWidth/2*3, YValues4, BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color4, hatch="\\\\")

    # for debugging only
    # op_disabled_data *= 1.1
    # op_ideal_data *= 0.9
    # op_ideal_data = op_disabled_data * 0.5
    # embodied_data *= 0.8

    # find index of the smallest value
    disabled_min_idx = np.argmin(embodied_data + op_disabled_data)  # type: ignore
    idel_min_idx = np.argmin(embodied_data + op_ideal_data)  # type: ignore

    num_bars = len(__LIFETIME.value)
    for i in range(num_bars):
        xvals = XValues-BarWidth*(num_bars/2-i-0.5)-0.03
        disabled_color= "firebrick" if i == disabled_min_idx else color1
        ideal_color= "firebrick" if i == idel_min_idx else "white"
        disabled_edgecolor= "black" if i == disabled_min_idx else "black"
        ideal_marker = "*" if i == idel_min_idx else "^"
        ideal_s = 150 if i == idel_min_idx else 100
        if i == 0:
            ax.bar(
                xvals, embodied_data[i], BarWidth, edgecolor=disabled_edgecolor, linewidth=linewidth, color=disabled_color, hatch="//", label="Embodied Carbon"  # type: ignore
            )
            ax.bar(
                xvals, op_disabled_data[i], BarWidth, edgecolor=disabled_edgecolor, linewidth=linewidth, color=disabled_color, hatch="", bottom=embodied_data[i], label="Operational Carbon (NoPG)" # type: ignore
            )
            ax.scatter(
                xvals, op_ideal_data[i] + embodied_data[i], marker=ideal_marker, s=ideal_s, edgecolor="black", facecolor=ideal_color, linewidth=linewidth, label="Embodied+Operational Carbon (ReGate-Full)"  # type: ignore
            )
        else:
            ax.bar(
                xvals, embodied_data[i], BarWidth, edgecolor=disabled_edgecolor, linewidth=linewidth, color=disabled_color, hatch="//"  # type: ignore
            )
            ax.bar(
                xvals, op_disabled_data[i], BarWidth, edgecolor=disabled_edgecolor, linewidth=linewidth, color=disabled_color, hatch="", bottom=embodied_data[i] # type: ignore
            )
            ax.scatter(
                xvals, op_ideal_data[i] + embodied_data[i], marker=ideal_marker, s=ideal_s, edgecolor="black", facecolor=ideal_color, linewidth=linewidth  # type: ignore
            )
        # ax.bar(
        #     xvals, op_disabled_data[i], BarWidth, edgecolor=edgecolor, linewidth=linewidth, color=color2, hatch="", bottom=embodied_data[i]
        # )
        # ax.scatter(
        #     xvals, embodied_data[i] + op_disabled_data[i], color=color1, marker="v", edgecolor="black", linewidth=linewidth  # type: ignore
        # )

    ax.set_xlim( (XValues[0]-num_bars*BarWidth/1.7, XValues[-1]+num_bars*BarWidth/1.7) )
    default_ylim = ax.get_ylim()
    new_ylim = (
        default_ylim[0] if y_lim[0] is None else y_lim[0],
        default_ylim[1] if y_lim[1] is None else y_lim[1],
    )
    ax.set_ylim( new_ylim )

    # lg=ax.legend(prop={'size': fontsize_legend}, ncol=1, loc="upper right", borderaxespad=0.)
    # lg.draw_frame(False)
    if plot_legend:
        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])
        handles, labels = ax.get_legend_handles_labels()
        lg=ax.legend(
            flip(handles, 3), flip(labels, 3),
            prop={'size': fontsize_legend}, ncol=3, loc="upper center", borderaxespad=0.,
            bbox_to_anchor=(3, 1.29), frameon=False,
        )
        lg.draw_frame(True)
        frame = lg.get_frame()
        frame.set_edgecolor("black")


def main(argv: list[str]):
    del argv  # Unused

    LLM_XNames = [
        "  ".join(__LIFETIME.value)
    ]
    DLRM_XNames = [
        "  ".join(__LIFETIME.value)
    ]
    SD_XNames = [
        "  ".join(__LIFETIME.value)
    ]

    all_data = [
        get_energy_eff_llm_training(),
        get_energy_eff_llm_inference("prefill"),
        get_energy_eff_llm_inference("decode"),
        get_energy_eff_dlrm_inference(),
        get_energy_eff_sd_inference(),
    ]
    all_embodied = [
        np.array(x).T * 1e3 for x, _, _ in all_data
    ]
    all_op_disabled = [
        np.array(y).T * 1e3 for _, y, _ in all_data
    ]
    all_op_ideal = [
        np.array(z).T * 1e3 for _, _, z in all_data
    ]
    all_embodied[1] = all_embodied[1] * 1e3
    all_embodied[2] = all_embodied[2] * 1e3
    all_embodied[3] = all_embodied[3] * 1e3
    all_op_disabled[1] = all_op_disabled[1] * 1e3
    all_op_disabled[2] = all_op_disabled[2] * 1e3
    all_op_disabled[3] = all_op_disabled[3] * 1e3
    all_op_ideal[1] = all_op_ideal[1] * 1e3
    all_op_ideal[2] = all_op_ideal[2] * 1e3
    all_op_ideal[3] = all_op_ideal[3] * 1e3

    all_XNames = [
        LLM_XNames,
        LLM_XNames,
        LLM_XNames,
        DLRM_XNames,
        SD_XNames,
    ]

    all_YLabels = [
        "Carbon Footprint\n(gCO2e/Iter)",
        "(kgCO2e/Million Token)",
        "(kgCO2e/Million Token)",
        "(kgCO2e/Million Request)",
        "(gCO2e/Image)",
    ]

    all_Titles = [
        "Llama3.1-405B Training",
        "Llama3.1-405B Prefill",
        "Llama3.1-405B Decode",
        "DLRM-L Inference",
        "DiT-XL Inference",
    ]

    # None means use default.
    # Otherwise, use the specified value.
    # all_ylims = [
    #     (1e2, 1.2e5),
    #     (1e-4, 1e-1),
    #     (1e-2, 1.2e-1),
    # ]
    all_ylims = [
        (None, None), #(5, 3e1),
        (None, None), #(2e-1, 1),
        (None, None), #(2, 7),
        (None, None), #(8e-3, 4e-2),
        (None, None), #(1e-1, 6e-1),
    ]
    all_logy = [
        True,
        True,
        True,
        True,
        True,
    ]

    all_yticks = [
        None, # [5, 10, 15, 20, 25, 30],
        [0, 0.2, 0.4, 0.6, 0.8, 1], # [2e-1, 4e-1, 6e-1, 8e-1, 1],
        None, # [2, 3, 4, 5, 6, 7],
        [0, 0.01, 0.02, 0.03], # [1e-2, 2e-2, 3e-2, 4e-2],
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], # [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1],
    ]

    all_yticklabels = [
        None, # ["5", "10", "15", "20", "25", "30"],
        ["0", "0.2", "0.4", "0.6", "0.8", "1"], # [f"{x:.1f}" for x in [2e-1, 4e-1, 6e-1, 8e-1]] + ["1"],
        None, # [str(x) for x in [2, 3, 4, 5, 6, 7]],
        ["0", "0.01", "0.02", "0.03"], # [f"{x:.2f}" for x in [1e-2, 2e-2, 3e-2, 4e-2]],
        ["0"] + [f"{x:.1f}" for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], # [f"{x:.1f}" for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    ]
    # all_yticks = [ None ] * 5
    # all_yticklabels = [ None ] * 5

    NROWS = 1
    NCOLS = 5

    Figure = PyPlot.figure( figsize=(23, 3))
    pdf_filename = f"outputs/carbon_efficiency_lca.pdf"
    PDF = PdfPages( pdf_filename )

    graphs = Figure.subplots( nrows=NROWS, ncols=NCOLS )

    for idx, (xnames, embodied_data, op_disabled_data, op_ideal_data, ylabel, title, ylim, logy, yticks, yticklabels) in enumerate(zip(
        all_XNames, all_embodied, all_op_disabled, all_op_ideal, all_YLabels, all_Titles, all_ylims, all_logy, all_yticks, all_yticklabels
    )):
        # Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        Graph = graphs[idx]
        plot_legend = (idx == 0)
        plot_data(Graph, embodied_data, op_disabled_data, op_ideal_data, xnames, ylabel, title, ylim, logy, plot_legend, yticks, yticklabels)

    Figure.tight_layout()
    Figure.subplots_adjust(wspace=0.28)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
