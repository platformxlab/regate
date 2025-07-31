## Overview

This directory contains the ops generator classes that acts as the frontend of the NPU simulator.
Each ops generator represents the DNN graph of a specific model architecture, such as LLM, DLRM, and stable diffusion models.
This README provides detailed explanations of the ops generator classes and their functionalities.


## Quick Start (Launching a Single Simulation Run)

Each ops generator class can be invoked directly in a Python script by feeding it a simulation configuration. It can generate the operators of a fordward or backward pass of a DNN model and dump the operator-level performance stats into a CSV file. Here, we provide a minimum working example, which launches the simulation for Llama3-8B inference.

First, we need to create a simulation configuration. This can be either a Python dictionary or an `LLMConfig` object (see [`LLMConfig.py`](configs/models/LLMConfig.py) for all configurable parameters).
It is recommended to put all configuration parameters into JSON files (see [`configs/`](configs/) for examples; most of the parameters have self-explanatory names), load and merging them into a Python dictionary, and then create the `LLMConfig` object from the dictionary.
```python
import json
from trace_util.llm_ops_generator.configs.models.LLMConfig import LLMConfig

# Load model configuration from JSON file
with open("configs/models/llama3_8b.json") as f:
    model_cfg = json.load(f)

# Load NPU configuration from JSON file
with open("configs/chips/tpuv5p.json") as f:
    npu_cfg = json.load(f)

# load system configuration from JSON file
with open("configs/systems/system_config.json") as f:
    system_cfg = json.load(f)

# Merge all configurations into a single dictionary
config_dict = { **system_cfg, **npu_cfg, **model_cfg }

# If you want to override some parameters, you can do so by directly modifying the dictionary.
# For example, to specify the output file path:
config_dict["output_file_path"] = "./llama3-8b-inference-v5p.csv"

# Create an LLMConfig object from the dictionary
# This step is optional as our ops generator class can accept a Python dictionary directly
# and automatically convert it to an `LLMConfig` object internally.
config: LLMConfig = LLMConfig.model_validate(config_dict)
```

Then, we create the ops generator instance and invoke the `generate()` function to trigger the simulation.
```python
from trace_util.llm_ops_generator.LLMOpsGenerator import LLMOpsGenerator

# Create an instance of the ops generator
ops_generator = LLMOpsGenerator(config)
# Alternatively, this can be done using a dictionary directly:
#   ops_generator = LLMOpsGenerator(config_dict)
# For the LLMOpsGenerator, we support both training and inference modes.
# The class LLMOpsGeneratorInference is just an alias of LLMOpsGenerator.
# It will gnerate the forward pass for both prefill and decode.
# The class LLMOpsGeneratorTraining is used for training.
# It will generate the forward and backward passes.

# Simulate the prefill and decode for LLM inference.
ops, prefill_ops, decode_ops = ops_generator.generate(dump_to_file=True, separate_prefill_decode=True)
```
The simulation results will be dumped as CSV files in the specified output path.
As we specified `separate_prefill_decode=True`, the ops generator will generate three CSV files:
- `llama3-8b-inference-v5p.csv`: Contains the operators for both prefill and decode phases.
- `llama3-8b-inference-v5p_prefill.csv`: Contains only the operators for the prefill phase.
- `llama3-8b-inference-v5p_decode.csv`: Contains only the operators for the decode phase.

The generated ops and their statistics are also returned as a list of `Operator` objects (see [Operator.py](Operator.py)). They can be programmatically accessed for further analysis or visualization in your own script.

### Power Simulation
The `power_analysis_lib.py` module implements the per-operator energy consumption analysis.
The power simulation can be invoked as follows:
```python
from trace_util.llm_ops_generator import power_analysis_lib as power_lib

# This can be either a power_lib.PowerGatingConfig or a string representing a pre-defined power gating strategy. See the get_power_gating_config() function in power_analysis_lib.py for more details.
power_gating_strategy = "NoPG"  # NoPG stands for no power gating.

# This can also be done separately for prefill_ops and decode_ops.
for op in ops:
    power_lib.analyze_operator_energy(
        op, config, pg_config=power_gating_strategy
    )

# Dump the operators into a CSV file.
with open(config.output_file_path, "w") as f:
    writer = csv.DictWriter(f, fieldnames=ops[0].keys())
    writer.writeheader()
    writer.writerows(ops)
```

## Supported Ops Generators

We currently provide the following ops generator modules for different model architectures:
- [`llm_ops_generator.py`](llm_ops_generator.py): Contains `LLMOpsGeneratorInference` and `LLMOpsGeneratorTraining` for Llama architecture, and `DeepSeekOpsGenerator` for DeepSeek inference.
- [`dlrm_ops_generator.py`](dlrm_ops_generator.py): Contains `DLRMOpsGenerator` for DLRM inference.
- [`dit_ops_generator.py`](dit_ops_generator.py): Contains `DiTOpsGenerator` for DiT inference. This is a wrapper of LLMOpsGenerator since DiT inference is similar to LLM prefill.
- [`gligen_ops_generator.py`](gligen_ops_generator.py): Contains `GLIGENOpsGenerator` for GLIGEN inference.

## Pre-defined Configurations

See [`configs/`](configs/) for the pre-defined configurations for different model architectures and NPU platforms.
- [`configs/chips/`](configs/chips/): Contains the NPU configurations for different NPU platforms. The configurable parameters are defined in [`ChipConfig.py`](configs/chips/ChipConfig.py).
- [`configs/models/`](configs/models/): Contains the model configurations for different model architectures. We create different pydantic classes for different model architectures as they require customized configurations. Parameters that are common to all model architectures are defined in [`ModelConfig.py`](configs/models/ModelConfig.py). Model-specific parameters are defined in their respective classes, such as [`LLMConfig.py`](configs/models/LLMConfig.py) for LLMs, [`DLRMConfig.py`](configs/models/DLRMConfig.py) for DLRM, [`DiTConfig.py`](configs/models/DiTConfig.py) for DiT, and [`GLIGENConfig.py`](configs/models/GLIGENConfig.py) for GLIGEN.
- [`configs/systems/`](configs/systems/): Contains the system configuration. Currently, we only support datacenter power usage efficiency (`PUE`) and carbon intensity `carbon_intensity_kgCO2_per_kWh`. These are used for fleet-wide energy and carbon analysis.