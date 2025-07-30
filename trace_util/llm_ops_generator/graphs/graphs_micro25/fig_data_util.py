import json
import csv
import os
from typing import Sequence, Any


def get_stats_filepath(
    model: str,
    version: str,
    workload: str,
    dp: int,
    tp: int,
    pp: int,
    dp_dcn: int,
    tp_dcn: int,
    pp_dcn: int,
    batch_size: int,
    prefill_or_decode: str = "",
) -> str:
    '''
    Returns the path to the json stats file for the given experiment.
    '''
    pstr = f"dp{dp}-tp{tp}-pp{pp}-dpdcn{dp_dcn}-tpdcn{tp_dcn}-ppdcn{pp_dcn}-bs{batch_size}"
    if workload == "inference":
        prefill_or_decode = f"_{prefill_or_decode}"
    elif workload == "training":
        prefill_or_decode = ""
    else:
        raise ValueError(f"Invalid workload: {workload}")
    return f"../results/raw/{model}/{pstr}/{workload}-v{version}{prefill_or_decode}.json"


def get_stats(
    model: str,
    version: str,
    workload: str,
    dp: int,
    tp: int,
    pp: int,
    dp_dcn: int,
    tp_dcn: int,
    pp_dcn: int,
    batch_size: int,
    prefill_or_decode: str = "",
) -> dict[str, Any]:
    '''
    Returns the stats dictionary for the given experiment.
    '''
    stats_file = get_stats_filepath(
        model, version, workload, dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, batch_size, prefill_or_decode
    )
    stats = json.load(open(stats_file, "r"))
    return stats


def get_all_stats(
    model: str,
    version: str,
    workload: str,
    prefill_or_decode: str = "",
    max_num_chips: int = -1,
    batch_size: int = 128,
) -> dict[tuple[int, int, int, int, int, int], dict[str, Any]]:
    '''
    Returns all stats dictionaries for all experiments of the given model and version.
    Index the returned dict by @return[(dp, tp, pp, dp_dcn, pp_dcn, batch_size)].
    '''
    model_results_dir = f"../results/raw/{model}/"
    config_dirs = os.listdir(model_results_dir)
    all_stats = {}
    for config_dir in config_dirs:
        dpstr, tpstr, ppstr, dpdcnstr, tpdcnstr, ppdcnstr, bsstr = config_dir.split("-")
        dp = int(dpstr.removeprefix("dp"))
        tp = int(tpstr.removeprefix("tp"))
        pp = int(ppstr.removeprefix("pp"))
        dp_dcn = int(dpdcnstr.removeprefix("dpdcn"))
        tp_dcn = int(tpdcnstr.removeprefix("tpdcn"))  # unused for now
        pp_dcn = int(ppdcnstr.removeprefix("ppdcn"))
        bs = int(bsstr.removeprefix("bs"))

        if batch_size != bs:
            continue

        num_chips = dp * tp * pp * dp_dcn * tp_dcn * pp_dcn
        if max_num_chips > 0 and num_chips > max_num_chips:
            continue

        # skip if the file does not exist
        stats_file = get_stats_filepath(
            model, version, workload, dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, batch_size, prefill_or_decode
        )
        if not os.path.exists(stats_file):
            continue

        stats = get_stats(
            model, version, workload, dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, batch_size, prefill_or_decode
        )
        all_stats[(dp, tp, pp, dp_dcn, pp_dcn, batch_size)] = stats
    return all_stats


def get_optimal_stats_for_max_num_chips(
    model: str,
    version: str,
    max_num_chips: int = 1024,
    workload: str = "inference",
    prefill_or_decode: str = "prefill",
    perf_metric: str = "total_execution_time_ns",
    min_or_max_metric: str = "min",
    batch_size: int = 128,
) -> dict[str, Any]:
    '''
    Returns the stats dictionary for the optimal configuration
    (e.g., parallelism config and batch size) for the given model and version.
    @perf_metric: one of [
            "total_execution_time_ns",
            "overlapped_compute_time_ns",
            "compute_only_time_ns",
            "memory_only_time_ns",
            "ici_bound_time_ns",
            "throughput_tokens_per_sec",
            "throughput_tokens_per_sec_request",
            "TPOT_ms_request",
    ].
    @min_or_max_metric: one of ["min", "max"]; whether to use the min or max value of @perf_metric as the optimal.
    '''
    all_stats = get_all_stats(model, version, workload, prefill_or_decode, max_num_chips, batch_size)
    if min_or_max_metric == "min":
        optimal_stats = min(all_stats.values(), key=lambda x: x[perf_metric])
    elif min_or_max_metric == "max":
        optimal_stats = max(all_stats.values(), key=lambda x: x[perf_metric])
    else:
        raise ValueError(f"Invalid min_or_max_metric: {min_or_max_metric}")

    return optimal_stats


def get_component_data_from_file(results_file: str, component: str) -> int:
    '''
    Returns the bounded-by time (exposed time) specified by @component.
    @component is one of ["Execution time", "Compute time", "Memory time", "ICI/NVLink time"].
    If @component is "Execution time", returns the overlapped time between compute and *memory or network*.
    If @component is "Compute time" or "Memory time", returns the compute-only or memory-only time.
    If @component is "ICI/NVLink time", returns the *sum of the overlapped network+memory (no compute) time and the network-only time*
    (i.e., the network time that is not overlapped with compute).
    The total execution time is the sum of get_component_data() for calling all types of components.
    '''

    with open(results_file, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        bounded_by_key = component.removesuffix(" time")
        if bounded_by_key == "Execution":
            data_point = sum([
                min(
                    int(row["Compute time"]),
                    max(
                        int(row["Memory time"]),
                        int(row["ICI/NVLink time"])
                    )
                ) * int(row["Count"])
                for row in reader
            ])
        elif bounded_by_key == "Compute":
            data_point = sum([
                (int(row["Compute time"]) - max(int(row["Memory time"]), int(row["ICI/NVLink time"])))
                * int(row["Count"])
                for row in reader
                if row["Bounded-by"] == bounded_by_key
            ])
        elif bounded_by_key == "Memory":
            data_point = sum([
                (int(row["Memory time"]) - max(int(row["Compute time"]), int(row["ICI/NVLink time"])))
                * int(row["Count"])
                for row in reader
                if row["Bounded-by"] == bounded_by_key
            ])
        elif bounded_by_key == "ICI/NVLink":
            data_point = sum([
                abs(int(row["ICI/NVLink time"]) - int(row["Compute time"]))
                * int(row["Count"])
                for row in reader
                if row["Bounded-by"] == bounded_by_key
            ])
        else:
            raise ValueError(f"Invalid bounded_by_key: {bounded_by_key}")
    return data_point


def get_total_execution_time_from_file(results_file: str) -> int:
    '''
    Returns the total execution time (sum of all components) from the results file.
    '''
    with open(results_file, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        data_point = sum([
            int(row["Execution time"]) * int(row["Count"])
            for row in reader
        ])
    return data_point


def get_optimal_parallelism_config(
    model: str,
    version: str,
    max_num_chips: int = 1024,
    workload: str = "inference",
    prefill_or_decode: str = "prefill",
) -> tuple[int, int, int, int]:
    '''
    Returns the optimal (DP, TP, PP, exe time ns) configuration for the model and TPU version.
    '''
    model_results_dir = f"../results/raw/{model}/"
    pconfig_dirs = os.listdir(model_results_dir)

    if workload == "training":
        prefill_or_decode = ""
    elif workload == "inference":
        prefill_or_decode = f"_{prefill_or_decode}"
    else:
        raise ValueError(f"Invalid workload: {workload}")
    
    all_config_times = []
    for pconfig_dir in pconfig_dirs:
        dpstr, tpstr, ppstr = pconfig_dir.split("-")
        dp = int(dpstr.removeprefix("dp"))
        tp = int(tpstr.removeprefix("tp"))
        pp = int(ppstr.removeprefix("pp"))
        if dp * tp * pp > max_num_chips:
            continue
        results_file = os.path.join(
            model_results_dir, pconfig_dir, f"{workload}-v{version}{prefill_or_decode}.csv"
        )
        if not os.path.exists(results_file):
            continue
        data_point = get_total_execution_time_from_file(results_file)
        all_config_times.append((dp, tp, pp, data_point))
    
    optimal_config = min(all_config_times, key=lambda x: x[3])
    
    return optimal_config


def get_all_optimal_parallelism_configs(
    models: Sequence[str],
    versions: Sequence[str],
    max_num_chips: int = 1024,
    dump_configs: bool = False,
    workload: str = "inference",
    prefill_or_decode: str = "prefill",
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    '''
    Returns the optimal (DP, TP, PP, exe time ns) configuration for each model and TPU version.
    @return[m][v] is the optimal (DP, TP, PP, exe time ns) configuration for model m and TPU version v.
    '''
    optimal_configs = {}
    for model in models:
        optimal_configs[model] = {}
        for v in versions:
            model_results_dir = f"../results/raw/{model}/"
            optimal_config = get_optimal_parallelism_config(
                model, v, max_num_chips, workload, prefill_or_decode
            )
            optimal_configs[model][v] = optimal_config

    if dump_configs:
        # dump as json
        with open(f"optimal_parallelism_configs_{workload}_{max_num_chips}.json", "w") as f:
            json.dump(optimal_configs, f, indent=4)

    return optimal_configs


def get_carbon_energy_stats_filepath(
    model: str,
    version: str,
    workload: str,
    num_chips: int,
    batch_size: int = 128,
    prefill_or_decode: str = "",
    carbon_intensity: float = 0.0624,
) -> str:
    prefix = f"../results/carbon_old/{model}/CI{carbon_intensity}/{workload}-"
    prefill_or_decode_str = "" if workload == "training" else f"{prefill_or_decode}-"
    prefix += prefill_or_decode_str
    bs_str = "" if workload == "inference" else f"-bs{batch_size}"
    fpath = prefix + f"v{version}-chip{num_chips}{bs_str}.json"
    return fpath


def get_carbon_energy_stats(
    model: str,
    version: str,
    workload: str,
    num_chips: int,
    batch_size: int = 128,
    prefill_or_decode: str = "",
    carbon_intensity: float = 0.0624,
) -> dict[str, Any]:
    filepath = get_carbon_energy_stats_filepath(
        model, version, workload, num_chips, batch_size, prefill_or_decode, carbon_intensity
    )
    stats = json.load(open(filepath, "r"))
    return stats


def get_slo_stats(
    model: str,
    workload: str,
    versions: Sequence[str],
    slo_scales: Sequence[int],
    batch_size: int = 128,
    prefill_or_decode: str = "",
) -> list[list[int]]:
    '''
    Returns the SLO stats for the given model and version.

    @return[slo_scale][version] = min # of chips to satisfy the given SLO scale for the given version.
    '''
    SLO_STATS_PATH = "/home/yuqixue2/neucloud/code/tpusim/trace_util/llm_ops_generator/results/slo"
    if workload == "inference":
        if "llama" in model:
            fname = f"{workload}-{prefill_or_decode}.json"
        else:
            fname = f"{workload}.json"
    elif workload == "training":
        fname = f"{workload}.json"
    else:
        raise ValueError(f"Invalid workload: {workload}")
    stats_path = os.path.join(SLO_STATS_PATH, model, fname)
    with open(stats_path, "r") as f:
        slo_stats_dict = json.load(f)
    
    slo_stats = []
    for slo_scale in slo_scales:
        num_chip_stats = []
        for v in versions:
            num_chip_stats.append(slo_stats_dict[str(slo_scale)][v]["min_num_chips"])
        slo_stats.append(num_chip_stats)
    return slo_stats
