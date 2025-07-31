#!/bin/bash

### Run all stuff for ReGate sensitivity analysis


# assert $NPUSIM_HOME is set
if [ -z "$NPUSIM_HOME" ]; then
    echo "Error: NPUSIM_HOME is not set. Please set it to the root directory of the simulator."
    exit 1
fi

# stop on error
set -xe

export RESULTS_DIR="$NPUSIM_HOME/trace_util/llm_ops_generator/results"
export CONFIGS_PATH="$NPUSIM_HOME/trace_util/llm_ops_generator/configs"
export PYTHONPATH="$NPUSIM_HOME:$PYTHONPATH"

export RAY_ADDRESS="http://127.0.0.1:8265"

function RAY_PREFIX() {
    # if $1 is set, use it as job_id, else use default
    if [ -n "$1" ]; then
        ray_prefix="ray job submit --working-dir ./ --runtime-env ./runtime_env.json --submission-id $1-$(date +'%F-%T') --"
    else
        ray_prefix="ray job submit --working-dir ./ --runtime-env ./runtime_env.json --"
    fi
    echo "$ray_prefix"
}


NPU_VERSIONS="5p"  # "2,3,4,5p,6p"

pg_strategies=("NoPG" "Base" "HW" "Full" "Ideal")

# Vary the power-gating delays. The values are number of times of the default delay value.
pg_strategies+=("Full_vary_PG_delay_1.5" "Full_vary_PG_delay_2" "Full_vary_PG_delay_3" "Full_vary_PG_delay_4")
pg_strategies+=("HW_vary_PG_delay_1.5" "HW_vary_PG_delay_2" "HW_vary_PG_delay_3" "HW_vary_PG_delay_4")
pg_strategies+=("Base_vary_PG_delay_1.5" "Base_vary_PG_delay_2" "Base_vary_PG_delay_3" "Base_vary_PG_delay_4")

# Vary the power-gating threshold voltages. The values are the percentage of the power-gated power consumption
# over the power consumption at the default fully powered-on voltage.
pg_strategies+=("Full_vary_Vth_0.1_0.01" "Full_vary_Vth_0.2_0.1" "Full_vary_Vth_0.4_0.25" "Full_vary_Vth_0.6_0.4")
pg_strategies+=("HW_vary_Vth_0.1_0.3" "HW_vary_Vth_0.2_0.4" "HW_vary_Vth_0.4_0.5" "HW_vary_Vth_0.6_0.8" "HW_vary_Vth_0.03_0.1")
pg_strategies+=("Base_vary_Vth_0.1_0.3" "Base_vary_Vth_0.2_0.4" "Base_vary_Vth_0.4_0.5" "Base_vary_Vth_0.6_0.8" "Base_vary_Vth_0.03_0.1")

# launch per op energy analysis runs
pg_str=""
for i in "${pg_strategies[@]}"; do
    pg_str="$pg_str,$i"
done
pg_str=${pg_str:1}  # remove leading comma
echo "Power gating strategies: $pg_str"
# LLMs
$(RAY_PREFIX energy_operator_llm_inference) python energy_operator_analysis_main.py --power_gating_strategy=$pg_str --models="llama3-8b,llama2-13b,llama3-70b,llama3_1-405b" --npu_versions=$NPU_VERSIONS --workload=inference --results_path="$RESULTS_DIR/raw" &
$(RAY_PREFIX energy_operator_llm_training) python energy_operator_analysis_main.py --power_gating_strategy=$pg_str --models="llama3-8b,llama2-13b,llama3-70b,llama3_1-405b" --npu_versions=$NPU_VERSIONS --workload=training --results_path="$RESULTS_DIR/raw" &
# DLRM
$(RAY_PREFIX energy_operator_dlrm_inference) python energy_operator_analysis_main.py --power_gating_strategy=$pg_str --models="dlrm-s,dlrm-m,dlrm-l" --npu_versions=$NPU_VERSIONS --workload=inference --results_path="$RESULTS_DIR/raw" &
# DiT
$(RAY_PREFIX energy_operator_dit_inference) python energy_operator_analysis_main.py --power_gating_strategy=$pg_str --models="dit-xl" --npu_versions=$NPU_VERSIONS --workload=inference --results_path="$RESULTS_DIR/raw" &
# # GLIGEN
$(RAY_PREFIX energy_operator_gligen_inference) python energy_operator_analysis_main.py --power_gating_strategy=$pg_str --models="gligen" --npu_versions=$NPU_VERSIONS --workload=inference --results_path="$RESULTS_DIR/raw" &
wait

# For carbon analysis, we do not need to include the default power-gating paramters again ("NoPG" "Base" "HW" "Full" "Ideal").
pg_strategies=("Full_vary_PG_delay_1.5" "Full_vary_PG_delay_2" "Full_vary_PG_delay_3" "Full_vary_PG_delay_4")
pg_strategies+=("HW_vary_PG_delay_1.5" "HW_vary_PG_delay_2" "HW_vary_PG_delay_3" "HW_vary_PG_delay_4")
pg_strategies+=("Base_vary_PG_delay_1.5" "Base_vary_PG_delay_2" "Base_vary_PG_delay_3" "Base_vary_PG_delay_4")
pg_strategies+=("Full_vary_Vth_0.1_0.01" "Full_vary_Vth_0.2_0.1" "Full_vary_Vth_0.4_0.25" "Full_vary_Vth_0.6_0.4")
pg_strategies+=("HW_vary_Vth_0.1_0.3" "HW_vary_Vth_0.2_0.4" "HW_vary_Vth_0.4_0.5" "HW_vary_Vth_0.6_0.8")
pg_strategies+=("Base_vary_Vth_0.1_0.3" "Base_vary_Vth_0.2_0.4" "Base_vary_Vth_0.4_0.5" "Base_vary_Vth_0.6_0.8")

# launch carbon analysis runs
for PG_STRATEGY in "${pg_strategies[@]}"; do
    UTIL_FACTOR=0.6
    CI="0.0624" # ,0.0717,0.1012,0.1155,0.1352"
    # LLMs
    $(RAY_PREFIX carbon_analysis_llm_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="llama3-8b,llama2-13b,llama3-70b,llama3_1-405b" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --npu_versions=$NPU_VERSIONS --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_energy" &
    $(RAY_PREFIX carbon_analysis_llm_training_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="llama3-8b,llama2-13b,llama3-70b,llama3_1-405b" --workload=training --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --npu_versions=$NPU_VERSIONS --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_energy" &
    # DLRM
    $(RAY_PREFIX carbon_analysis_dlrm_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="dlrm-s,dlrm-m,dlrm-l" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --npu_versions=$NPU_VERSIONS --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_energy" &
    # DiT
    $(RAY_PREFIX carbon_analysis_dit_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="dit-xl" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --npu_versions=$NPU_VERSIONS --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_energy" &
    # GLIGEN
    $(RAY_PREFIX carbon_analysis_gligen_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="gligen" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --npu_versions=$NPU_VERSIONS --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_energy" &
    wait
done

wait