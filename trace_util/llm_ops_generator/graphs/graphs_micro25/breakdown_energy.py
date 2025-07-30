#!/usr/bin/env python3

import itertools
from typing import Any, Sequence
from absl import app, flags, logging

import json
import numpy as np
import matplotlib as PlotLib
from matplotlib.axes import Axes
import matplotlib.pyplot as PyPlot
from matplotlib.backends.backend_pdf import PdfPages
import os
import csv
from scipy.stats import gmean

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

# bar width and x tick scaling factor
BarWidth = 0.4
XTickFactor = 2


__PG_STRATEGY = flags.DEFINE_string(
    "pg_strategy",
    "NoPG",
    "Power gating strategy to use for energy analysis, \
    one of 'disabled', 'ideal_inst_component', 'ideal_op_component', 'ideal_inst_PE_ALU'",
)


LLM_MODELS = ["llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b"]
LLM_MODEL_NAMES = [
    "Llama3\n8B", "Llama2\n13B", "Llama3\n70B", "Llama3.1\n405B"
]
LLM_TRAINING_BATCH_SIZE = 32
DLRM_MODELS = ["dlrm-s", "dlrm-m", "dlrm-l"]
DLRM_MODEL_NAMES = ["DLRM-S", "DLRM-M", "DLRM-L"]
SD_MODELS = ["dit-xl", "gligen"]
SD_MODEL_NAMES = ["DiT-XL", "GLIGEN"]

__NPU_VERSIONS = flags.DEFINE_list(
    "npu_versions",
    ["2", "3", "4", "5p"],
    "NPU versions to use for energy analysis, one of '2', '3', '4', '5p'",
)

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
    logging.info("Optimal chip config for %s %s %s %d %s: %s, SLO scale: %d",
                 model, v, workload, batch_size, prefill_or_decode, pconfig, slo_scale)
    return results_lib.get_stats(
        model, v, workload, *pconfig, prefill_or_decode=prefill_or_decode
    ), slo_scale


def get_energy_eff_helper(
    models: Sequence[str],
    batch_size: int,
    workload: str,
    prefill_or_decode: str,
    energy_key: str,
) -> tuple[list[list[float]], ...]:
    '''
    @return[0:-1]: list of joule per iteration per layer breakdown for each model for each NPU version
    @return[-1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    idle_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    static_sa_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    static_vu_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    static_sram_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    static_ici_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    static_hbm_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    static_other_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    dyn_sa_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    dyn_vu_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    dyn_sram_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    dyn_ici_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    dyn_hbm_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    dyn_other_energy_eff = [[] for _ in __NPU_VERSIONS.value]
    slo_scales = [[] for _ in __NPU_VERSIONS.value]
    for im, model in enumerate(models):
        for iv, v in enumerate(__NPU_VERSIONS.value):
            stats, slo_scale = get_optimal_energy_stat(
                model, v, workload, batch_size, prefill_or_decode
            )
            value = stats["carbon_and_energy_stats"]["1"][energy_key]
            total_energy = stats["carbon_and_energy_stats"]["1"]["avg_total_energy_consumption_kWh"]
            idle_energy = stats["carbon_and_energy_stats"]["1"]["avg_idle_energy_consumption_kWh"]
            static_sa_energy = stats["carbon_and_energy_stats"]["1"]["avg_static_sa_energy_consumption_kWh"]
            static_vu_energy = stats["carbon_and_energy_stats"]["1"]["avg_static_vu_energy_consumption_kWh"]
            static_sram_energy = stats["carbon_and_energy_stats"]["1"]["avg_static_sram_energy_consumption_kWh"]
            static_ici_energy = stats["carbon_and_energy_stats"]["1"]["avg_static_ici_energy_consumption_kWh"]
            static_hbm_energy = stats["carbon_and_energy_stats"]["1"]["avg_static_hbm_energy_consumption_kWh"]
            static_other_energy = stats["carbon_and_energy_stats"]["1"]["avg_static_other_energy_consumption_kWh"]
            dyn_sa_energy = stats["carbon_and_energy_stats"]["1"]["avg_dynamic_sa_energy_consumption_kWh"]
            dyn_vu_energy = stats["carbon_and_energy_stats"]["1"]["avg_dynamic_vu_energy_consumption_kWh"]
            dyn_sram_energy = stats["carbon_and_energy_stats"]["1"]["avg_dynamic_sram_energy_consumption_kWh"]
            dyn_ici_energy = stats["carbon_and_energy_stats"]["1"]["avg_dynamic_ici_energy_consumption_kWh"]
            dyn_hbm_energy = stats["carbon_and_energy_stats"]["1"]["avg_dynamic_hbm_energy_consumption_kWh"]
            dyn_other_energy = stats["carbon_and_energy_stats"]["1"]["avg_dynamic_other_energy_consumption_kWh"]
            idle_ratio = idle_energy / total_energy
            static_sa_ratio = static_sa_energy / total_energy
            static_vu_ratio = static_vu_energy / total_energy
            static_sram_ratio = static_sram_energy / total_energy
            static_ici_ratio = static_ici_energy / total_energy
            static_hbm_ratio = static_hbm_energy / total_energy
            static_other_ratio = static_other_energy / total_energy
            dyn_sa_ratio = dyn_sa_energy / total_energy
            dyn_vu_ratio = dyn_vu_energy / total_energy
            dyn_sram_ratio = dyn_sram_energy / total_energy
            dyn_ici_ratio = dyn_ici_energy / total_energy
            dyn_hbm_ratio = dyn_hbm_energy / total_energy
            dyn_other_ratio = dyn_other_energy / total_energy

            # print("########################", model, v, workload, prefill_or_decode, ":")
            # print(stats["carbon_and_energy_stats"]["1"])
            # per_dp_batch_size = int(stats["sim_config"]["global_batch_size"]) / int(stats["sim_config"]["data_parallelism_degree"])
            # print(f"per_dp_batch_size: {per_dp_batch_size}")


            if slo_scale == -1:
                idle_energy_eff[iv].append(0)
                static_sa_energy_eff[iv].append(0)
                static_vu_energy_eff[iv].append(0)
                static_sram_energy_eff[iv].append(0)
                static_ici_energy_eff[iv].append(0)
                static_hbm_energy_eff[iv].append(0)
                static_other_energy_eff[iv].append(0)
                dyn_sa_energy_eff[iv].append(0)
                dyn_vu_energy_eff[iv].append(0)
                dyn_sram_energy_eff[iv].append(0)
                dyn_ici_energy_eff[iv].append(0)
                dyn_hbm_energy_eff[iv].append(0)
                dyn_other_energy_eff[iv].append(0)
            else:
                idle_energy_eff[iv].append(idle_ratio / value)
                static_sa_energy_eff[iv].append(static_sa_ratio / value)
                static_vu_energy_eff[iv].append(static_vu_ratio / value)
                static_sram_energy_eff[iv].append(static_sram_ratio / value)
                static_ici_energy_eff[iv].append(static_ici_ratio / value)
                static_hbm_energy_eff[iv].append(static_hbm_ratio / value)
                static_other_energy_eff[iv].append(static_other_ratio / value)
                dyn_sa_energy_eff[iv].append(dyn_sa_ratio / value)
                dyn_vu_energy_eff[iv].append(dyn_vu_ratio / value)
                dyn_sram_energy_eff[iv].append(dyn_sram_ratio / value)
                dyn_ici_energy_eff[iv].append(dyn_ici_ratio / value)
                dyn_hbm_energy_eff[iv].append(dyn_hbm_ratio / value)
                dyn_other_energy_eff[iv].append(dyn_other_ratio / value)
            slo_scales[iv].append(slo_scale)


    return (
        idle_energy_eff,
        static_sa_energy_eff,
        static_vu_energy_eff,
        static_sram_energy_eff,
        static_ici_energy_eff,
        static_hbm_energy_eff,
        static_other_energy_eff,
        dyn_sa_energy_eff,
        dyn_vu_energy_eff,
        dyn_sram_energy_eff,
        dyn_ici_energy_eff,
        dyn_hbm_energy_eff,
        dyn_other_energy_eff,
        slo_scales,
    )


def get_energy_eff_llm_training() -> tuple[list[list[float]], ...]:
    '''
    @return[0:-1]: list of joule per iteration per layer breakdown for each model for each NPU version
    @return[-1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    return get_energy_eff_helper(
        LLM_MODELS, LLM_TRAINING_BATCH_SIZE, "training", "", "avg_power_efficiency_iteration_per_joule"
    )


def get_energy_eff_llm_inference(prefill_or_decode: str) -> tuple[list[list[float]], ...]:
    '''
    @return[0:-1]: list of joule per token per layer breakdown for each model for each NPU version
    @return[-1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    return get_energy_eff_helper(
        LLM_MODELS, -1, "inference", prefill_or_decode, "avg_power_efficiency_tkn_per_joule"
    )


def get_energy_eff_dlrm_inference() -> tuple[list[list[float]], ...]:
    '''
    @return[0:-1]: list of joule per request breakdown for each model for each NPU version
    @return[-1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    return get_energy_eff_helper(
        DLRM_MODELS, -1, "inference", "", "avg_power_efficiency_req_per_joule"
    )


def get_energy_eff_sd_inference() -> tuple[list[list[float]], ...]:
    '''
    @return[0:-1]: list of joule per image breakdown for each model for each NPU version
    @return[-1]: list of SLO scales for each model for each NPU version
        @return[x][iv][im] is the data for NPU version iv and model im
    '''
    return get_energy_eff_helper(
        SD_MODELS, -1, "inference", "", "avg_power_efficiency_req_per_joule"
    )


def plot_data(
    ax: Axes,
    _idle_energy: Sequence[Sequence[float]],
    _static_sa_energy: Sequence[Sequence[float]],
    _static_vu_energy: Sequence[Sequence[float]],
    _static_sram_energy: Sequence[Sequence[float]],
    _static_ici_energy: Sequence[Sequence[float]],
    _static_hbm_energy: Sequence[Sequence[float]],
    _static_other_energy: Sequence[Sequence[float]],
    _dyn_sa_energy: Sequence[Sequence[float]],
    _dyn_vu_energy: Sequence[Sequence[float]],
    _dyn_sram_energy: Sequence[Sequence[float]],
    _dyn_ici_energy: Sequence[Sequence[float]],
    _dyn_hbm_energy: Sequence[Sequence[float]],
    _dyn_other_energy: Sequence[Sequence[float]],
    slo_data: Sequence[Sequence[float | int | str]],
    x_names: Sequence[str],
    y_label: str | None,
    title: str,
    y_lim,
    log_scale: bool = False,
    plot_legend: bool = False,
):
    XValues = np.arange( len( _idle_energy[ 0 ] ) ) * XTickFactor
    XTicks = XValues
    ax.set_xticks( XTicks )
    ax.set_xticklabels( x_names, fontsize=fontsize_xticklabel, position=(0,0.02), ha='center' )
    ax.xaxis.set_ticks_position( 'none' )

    YTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_yticks( YTicks )
    if plot_legend:
        ax.set_yticklabels( [f"{int(x*100)}%" for x in YTicks], fontsize=fontsize_yticklabel )
    else:
        ax.set_yticklabels( [] )
    ax.yaxis.set_ticks_position( 'none' )
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

    edgecolor = 'black' # '#404040'
    linewidth = 0.1

    # sum up all dynamic energy
    idle_energy = np.array(_idle_energy)
    static_sa_energy = np.array(_static_sa_energy)
    static_vu_energy = np.array(_static_vu_energy)
    static_sram_energy = np.array(_static_sram_energy)
    static_ici_energy = np.array(_static_ici_energy)
    static_hbm_energy = np.array(_static_hbm_energy)
    static_other_energy = np.array(_static_other_energy)
    dyn_sa_energy = np.array(_dyn_sa_energy)
    dyn_vu_energy = np.array(_dyn_vu_energy)
    dyn_sram_energy = np.array(_dyn_sram_energy)
    dyn_ici_energy = np.array(_dyn_ici_energy)
    dyn_hbm_energy = np.array(_dyn_hbm_energy)
    dyn_other_energy = np.array(_dyn_other_energy)

    tot_energy = (
        idle_energy +
        static_sa_energy + static_vu_energy + static_sram_energy + static_ici_energy + static_hbm_energy + static_other_energy +
        dyn_sa_energy + dyn_vu_energy + dyn_sram_energy + dyn_ici_energy + dyn_hbm_energy + dyn_other_energy
    )

    # normalize to 100%
    tot_energy_ratio = 1 # tot_energy / tot_energy[0]
    idle_energy = idle_energy / tot_energy * tot_energy_ratio
    static_sa_energy = static_sa_energy / tot_energy * tot_energy_ratio
    static_vu_energy = static_vu_energy / tot_energy * tot_energy_ratio
    static_sram_energy = static_sram_energy / tot_energy * tot_energy_ratio
    static_ici_energy = static_ici_energy / tot_energy * tot_energy_ratio
    static_hbm_energy = static_hbm_energy / tot_energy * tot_energy_ratio
    static_other_energy = static_other_energy / tot_energy * tot_energy_ratio
    dyn_sa_energy = dyn_sa_energy / tot_energy * tot_energy_ratio
    dyn_vu_energy = dyn_vu_energy / tot_energy * tot_energy_ratio
    dyn_sram_energy = dyn_sram_energy / tot_energy * tot_energy_ratio
    dyn_ici_energy = dyn_ici_energy / tot_energy * tot_energy_ratio
    dyn_hbm_energy = dyn_hbm_energy / tot_energy * tot_energy_ratio
    dyn_other_energy = dyn_other_energy / tot_energy * tot_energy_ratio

    static_energy = (
        static_sa_energy + static_vu_energy + static_sram_energy + static_ici_energy + static_hbm_energy + static_other_energy
    )
    print("######################", title, "######################")
    print("Row = NPU version, Col = Model")
    print("Idle energy percentage:")
    print(idle_energy)
    print("gmean:", gmean(idle_energy, axis=None))  # type: ignore

    print("static energy percentage:")
    print(static_energy)
    print("gmean:", gmean(static_energy, axis=None))  # type: ignore

    print("static energy / active energy:")
    print(static_energy / (1 - idle_energy))

    print("static sa / static:")
    static_sa_percentage = static_sa_energy / static_energy
    print(static_sa_percentage)
    print("gmean:", gmean(static_sa_percentage, axis=None))  # type: ignore
    print("Min:", np.min(static_sa_percentage, axis=None))  # type: ignore
    print("Max:", np.max(static_sa_percentage, axis=None))  # type: ignore

    print("static sa / active energy:")
    print(static_sa_percentage / (1 - idle_energy))
    print("gmean:", gmean(static_sa_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Min:", np.min(static_sa_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Max:", np.max(static_sa_percentage / (1 - idle_energy), axis=None))  # type: ignore

    print("static vu / static:")
    static_vu_percentage = static_vu_energy / static_energy
    print(static_vu_percentage)
    print("gmean:", gmean(static_vu_percentage, axis=None))  # type: ignore
    print("Min:", np.min(static_vu_percentage, axis=None))  # type: ignore
    print("Max:", np.max(static_vu_percentage, axis=None))  # type: ignore

    print("static vu / active energy:")
    print(static_vu_percentage / (1 - idle_energy))
    print("gmean:", gmean(static_vu_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Min:", np.min(static_vu_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Max:", np.max(static_vu_percentage / (1 - idle_energy), axis=None))  # type: ignore

    print("static sram / static:")
    static_sram_percentage = static_sram_energy / static_energy
    print(static_sram_percentage)
    print("gmean:", gmean(static_sram_percentage, axis=None))  # type: ignore
    print("Min:", np.min(static_sram_percentage, axis=None))  # type: ignore
    print("Max:", np.max(static_sram_percentage, axis=None))  # type: ignore

    print("static sram / active energy:")
    print(static_sram_percentage / (1 - idle_energy))
    print("gmean:", gmean(static_sram_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Min:", np.min(static_sram_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Max:", np.max(static_sram_percentage / (1 - idle_energy), axis=None))  # type: ignore

    print("static ici / static:")
    static_ici_percentage = static_ici_energy / static_energy
    print(static_ici_percentage)
    print("gmean:", gmean(static_ici_percentage, axis=None))  # type: ignore
    print("Min:", np.min(static_ici_percentage, axis=None))  # type: ignore
    print("Max:", np.max(static_ici_percentage, axis=None))  # type: ignore

    print("static ici / active energy:")
    print(static_ici_percentage / (1 - idle_energy))
    print("gmean:", gmean(static_ici_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Min:", np.min(static_ici_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Max:", np.max(static_ici_percentage / (1 - idle_energy), axis=None))  # type: ignore

    print("static hbm / static:")
    static_hbm_percentage = static_hbm_energy / static_energy
    print(static_hbm_percentage)
    print("gmean:", gmean(static_hbm_percentage, axis=None))  # type: ignore
    print("Min:", np.min(static_hbm_percentage, axis=None))  # type: ignore
    print("Max:", np.max(static_hbm_percentage, axis=None))  # type: ignore

    print("static hbm / active energy:")
    print(static_hbm_percentage / (1 - idle_energy))
    print("gmean:", gmean(static_hbm_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Min:", np.min(static_hbm_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Max:", np.max(static_hbm_percentage / (1 - idle_energy), axis=None))  # type: ignore

    print("static other / static:")
    static_other_percentage = static_other_energy / static_energy
    print(static_other_percentage)
    print("gmean:", gmean(static_other_percentage, axis=None))  # type: ignore
    print("Min:", np.min(static_other_percentage, axis=None))  # type: ignore
    print("Max:", np.max(static_other_percentage, axis=None))  # type: ignore

    print("static other / active energy:")
    print(static_other_percentage / (1 - idle_energy))
    print("gmean:", gmean(static_other_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Min:", np.min(static_other_percentage / (1 - idle_energy), axis=None))  # type: ignore
    print("Max:", np.max(static_other_percentage / (1 - idle_energy), axis=None))  # type: ignore

    # print(idle_energy)
    # print(static_energy)
    # print(dyn_energy)

    # print(f"{title}:\n\tmin idle, min static, min idle+static:\n\t", np.min(idle_energy), np.min(static_energy), np.min(idle_energy + static_energy))

    def plot_bar(xvals, yvals_stack, colors, hatches, labels):
        bar_containers = []
        for i, yvals in enumerate(yvals_stack):
            if labels:
                bar_containers.append(
                    ax.bar(
                        xvals, yvals, BarWidth, edgecolor=edgecolor, linewidth=linewidth,
                        color=colors[i], hatch=hatches[i], bottom=sum(yvals_stack[:i]),
                        label=labels[i],
                    )
                )
            else:
                bar_containers.append(
                    ax.bar(
                        xvals, yvals, BarWidth, edgecolor=edgecolor, linewidth=linewidth,
                        color=colors[i], hatch=hatches[i], bottom=sum(yvals_stack[:i]),
                    )
                )
        return bar_containers

    # 13 random colors
    colors = [
        "#A2B3BC",
        "#EA442C", "#EA862C", "#EA652C", "#ED4865", "#EAA42C", "#F2B79B",
        "#2CEAC8", "#2C91EA", "#2CCEEA", "#2CEA87", "#2C55EA", "#BBE5EC",
    ]
    hatches = [
        "",
        "//", "//", "//", "//", "//", "//",
        "", "", "", "", "", "",
    ]
    labels = [
        "Idle",
        "Static SA", "Static VU", "Static SRAM", "Static ICI", "Static HBM", "Static Other",
        "Dynamic SA", "Dynamic VU", "Dynamic SRAM", "Dynamic ICI", "Dynamic HBM", "Dynamic Other",
    ]

    yvals_stack = [
        idle_energy[0],
        static_sa_energy[0], static_vu_energy[0], static_sram_energy[0], static_ici_energy[0], static_hbm_energy[0], static_other_energy[0],
        dyn_sa_energy[0], dyn_vu_energy[0], dyn_sram_energy[0], dyn_ici_energy[0], dyn_hbm_energy[0], dyn_other_energy[0],
    ]
    bar_containers = plot_bar(XValues-BarWidth/2*3, yvals_stack, colors, hatches, labels)

    yvals_stack = [
        idle_energy[1],
        static_sa_energy[1], static_vu_energy[1], static_sram_energy[1], static_ici_energy[1], static_hbm_energy[1], static_other_energy[1],
        dyn_sa_energy[1], dyn_vu_energy[1], dyn_sram_energy[1], dyn_ici_energy[1], dyn_hbm_energy[1], dyn_other_energy[1],
    ]
    plot_bar(XValues-BarWidth/2, yvals_stack, colors, hatches, None)

    yvals_stack = [
        idle_energy[2],
        static_sa_energy[2], static_vu_energy[2], static_sram_energy[2], static_ici_energy[2], static_hbm_energy[2], static_other_energy[2],
        dyn_sa_energy[2], dyn_vu_energy[2], dyn_sram_energy[2], dyn_ici_energy[2], dyn_hbm_energy[2], dyn_other_energy[2],
    ]
    plot_bar(XValues+BarWidth/2, yvals_stack, colors, hatches, None)

    yvals_stack = [
        idle_energy[3],
        static_sa_energy[3], static_vu_energy[3], static_sram_energy[3], static_ici_energy[3], static_hbm_energy[3], static_other_energy[3],
        dyn_sa_energy[3], dyn_vu_energy[3], dyn_sram_energy[3], dyn_ici_energy[3], dyn_hbm_energy[3], dyn_other_energy[3],
    ]
    plot_bar(XValues+BarWidth/2*3, yvals_stack, colors, hatches, None)

    ax.set_xlim( ( XValues[0]-7*BarWidth/2, XValues[-1]+7*BarWidth/2) )
    default_ylim = (0, 1.1) #ax.get_ylim()
    new_ylim = (
        default_ylim[0] if y_lim[0] is None else y_lim[0],
        default_ylim[1] if y_lim[1] is None else y_lim[1],
    )
    ax.set_ylim( new_ylim )

    if plot_legend:
        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])
        handles, labels = ax.get_legend_handles_labels()
        lg=ax.legend(
            flip(handles, 7), flip(labels, 7),
            prop={'size': fontsize_legend}, ncol=7, loc="upper center", borderaxespad=0.,
            bbox_to_anchor=(2.5, 1.23), frameon=False,
        )
        lg.draw_frame(True)
        frame = lg.get_frame()
        frame.set_edgecolor("black")


def main(argv: list[str]):
    del argv  # Unused

    LLM_XNames = [
        "A B C D\n" + m
        for m in LLM_MODEL_NAMES
    ]
    DLRM_XNames = [
        "A  B  C  D\n" + m
        for m in DLRM_MODEL_NAMES
    ]
    SD_XNames = [
        "A    B    C    D\n" + m
        for m in SD_MODEL_NAMES
    ]

    # (
    #    idle_energy_eff,
    #    static_energy_eff,
    #    dyn_compute_energy_eff,
    #    dyn_memory_energy_eff,
    #    dyn_ici_energy_eff,
    #    slo_scales,
    # )
    all_data = [
        get_energy_eff_llm_training(),
        get_energy_eff_llm_inference("prefill"),
        get_energy_eff_llm_inference("decode"),
        get_energy_eff_dlrm_inference(),
        get_energy_eff_sd_inference(),
    ]

    all_idle_energy = [
        x[0] for x in all_data
    ]

    all_static_sa_energy = [
        x[1] for x in all_data
    ]
    all_static_vu_energy = [
        x[2] for x in all_data
    ]
    all_static_sram_energy = [
        x[3] for x in all_data
    ]
    all_static_ici_energy = [
        x[4] for x in all_data
    ]
    all_static_hbm_energy = [
        x[5] for x in all_data
    ]
    all_static_other_energy = [
        x[6] for x in all_data
    ]

    all_dyn_sa_energy = [
        x[7] for x in all_data
    ]
    all_dyn_vu_energy = [
        x[8] for x in all_data
    ]
    all_dyn_sram_energy = [
        x[9] for x in all_data
    ]
    all_dyn_ici_energy = [
        x[10] for x in all_data
    ]
    all_dyn_hbm_energy = [
        x[11] for x in all_data
    ]
    all_dyn_other_energy = [
        x[12] for x in all_data
    ]

    all_slo_scale = [
        x[13] for x in all_data
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
        LLM_XNames,
        LLM_XNames,
        LLM_XNames,
        DLRM_XNames,
        SD_XNames,
    ]

    all_YLabels = [
        "Norm. Energy Consumption", #\n(Joule/Iter)",
        None, #"Norm. Energy Consumption", #\n(Joule/Token)",
        None, #"Norm. Energy Consumption", #\n(Joule/Token)",
        None, #"Norm. Energy Consumption", #\n(Joule/Request)",
        None, #"Norm. Energy Consumption", #\n(Joule/Image)",
    ]

    all_Titles = [
        "LLM Training",
        "LLM Inference (Prefill)",
        "LLM Inference (Decode)",
        "DLRM Inference",
        "Stable Diffusion Inference",
    ]

    # None means use default.
    # Otherwise, use the specified value.
    # all_ylims = [
    #     (1e2, 1.2e5),
    #     (1e-4, 1e-1),
    #     (1e-2, 1.2e-1),
    # ]
    all_ylims = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
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

    Figure = PyPlot.figure( figsize=(25*1.2, 5*1.2) )
    pdf_filename = f"outputs/breakdown_energy_{__PG_STRATEGY.value}.pdf"
    PDF = PdfPages( pdf_filename )

    for idx, (
        xnames,
        idle,
        static_sa,
        static_vu,
        static_sram,
        static_ici,
        static_hbm,
        static_other,
        dyn_sa,
        dyn_vu,
        dyn_sram,
        dyn_ici,
        dyn_hbm,
        dyn_other,
        slo_scale_data,
        ylabel,
        title,
        ylim,
        logy
    ) in enumerate(zip(
        all_XNames,
        all_idle_energy,
        all_static_sa_energy,
        all_static_vu_energy,
        all_static_sram_energy,
        all_static_ici_energy,
        all_static_hbm_energy,
        all_static_other_energy,
        all_dyn_sa_energy,
        all_dyn_vu_energy,
        all_dyn_sram_energy,
        all_dyn_ici_energy,
        all_dyn_hbm_energy,
        all_dyn_other_energy,
        all_slo_scale,
        all_YLabels,
        all_Titles,
        all_ylims,
        all_logy,
    )):
        Graph = Figure.add_subplot(NROWS, NCOLS, idx + 1)
        plot_data(
            Graph,

            idle,
            static_sa,
            static_vu,
            static_sram,
            static_ici,
            static_hbm,
            static_other,
            dyn_sa,
            dyn_vu,
            dyn_sram,
            dyn_ici,
            dyn_hbm,
            dyn_other,

            slo_scale_data,

            xnames,
            ylabel,
            title,
            ylim,
            logy,
            plot_legend=(idx == 0),
        )

    Figure.tight_layout()
    Figure.subplots_adjust(wspace=0.02)

    PDF.savefig( Figure, bbox_inches='tight' )
    PDF.close()


if __name__ == "__main__":
    app.run(main)
