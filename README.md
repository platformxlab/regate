# ReGate: Enabling Power Gating in Neural Processing Units (Artifact)

This repository contains the artifact for the paper "ReGate: Enabling Power Gating in Neural Processing Units" presented at MICRO 2025.
We provide the NPU simulator source code and the scripts to reproduce the key figures in the paper.

***Note for MICRO'25 artifact reviewers:*** To reproduce the key figures in the paper, it is sufficient to go through section 1 to 3. Please see [micro25_ae_figures.md](trace_util/llm_ops_generator/graphs/graphs_micro25/micro25_ae_figures.md) for the list of all figures to be reproduced. Reproducing all figures take approximately 160 core-hours. We recommend a machine with at least 48 cores, 128 GB memory, and 600 GB disk space for a smooth experience.

## 0. Hardware and Software Dependencies 

The artifact can be executed on any x86 machine with at least 128 GB of main memory (for a smaller memory, disk swapping space may need to be enabled) and at least 600 GB of disk space. We strongly recommend running the artifact on a machine with at least 48 cores and 128 GB memory. Optionally, the artifact can also run on a Ray cluster with multiple machines to speedup the simulation experiments. The artifact needs a Linux distro (preferably Ubuntu) and a conda environment.

## 1. Installation & Setup

1. (Skip this step if you already have conda installed) Install Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer

2. Clone the repository:
   ```bash
   git clone https://github.com/platformxlab/regate.git
   ```

3. Change to the repository directory:
   ```bash
   cd regate
   ```

4. Create conda environment and install the required Python packages:
   ```bash
   conda create --name regate python=3.12.2
   conda activate regate
   pip install -r requirements.txt
   ```

5. Under the git repo directory, export the python path and the simulator root directory:
    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    export NPUSIM_HOME=$(pwd)
    ```

6. Configure the environment variables in `trace_util/llm_ops_generator/run_scripts/runtime_env.json`:
   <!-- - Set the `RESULTS_DIR` to the directory where you want to store the simulation results. ***Must use absolute path.*** -->
   - Set `PYTHONPATH` to the root directory of the cloned repository. ***Must use absolute path.*** When running on a single machine, this should be the same as the `$PYTHONPATH` set in step 5.
   - Set `RESULTS_DIR` to `${NPUSIM_HOME}/trace_util/llm_ops_generator/results`. ***Must use absolute path.*** For example, if in step 5, `NPUSIM_HOME` is set to `/home/user/regate`, then `RESULTS_DIR` should be set to `/home/user/regate/trace_util/llm_ops_generator/results`. Please refer to [Output Directory](#output-directory) for more details on how to set a different output directory.

   Using absolute paths is necessary since the simulation scripts may be run from different directories when running on a Ray cluster.

7. Start ray server:
    ```bash
    ray start --head --port=6379
    ```
    7.1. (Optional) To speed up the simulation, we can run the experiments on a Ray cluster with multiple machines. Please refer to [Running on a Ray Cluster](#running-on-a-ray-cluster).

8. Run the test script to verify the installation:
   ```bash
   cd trace_util/llm_ops_generator/run_scripts
   ./test.sh
   ```

   You may view the progress of the test runs in the Ray dashboard (at `http://127.0.0.1:8265/` by default; may require port forwarding if you are ssh'ing onto a remote machine; if you are ssh'ing using vscode, the port forwarding should be configured automatically once ray runtime starts). 

   After the script finishes with no errors, under the "Jobs" tab in the Ray dashboard, all jobs should have the "Status" column to be "SUCCEEDED".
   An output directory `trace_util/llm_ops_generator/results` should be created and contain the following folders:
   - `raw`: contains the performance simulation results. This is the output of the script `run_sim.py`.
   - `raw_energy`: contains the power simulation results. This is the output of the script `energy_operator_analysis_main.py`.
   - `carbon_NoPG/CI0.0624/UTIL0.6`: contains the results of the carbon emission analysis without power , with carbon intensity 0.0624 kgCO2e/kWh and NPU chip duty cycle 60%. This is the output of the script `carbon_analysis_main.py`.
   - `slo`: contains the SLO analysis results. This is the output of the script `slo_analysis_main.py`.


## 2. Experiment Workflow
We have provided two automated scripts to reproduce the key figures in the paper.
Under the `trace_util/llm_ops_generator/run_scripts` directory, please run:
```bash
./run_all_ReGate.sh
```
and then
```bash
./run_ReGate_sens.sh
```

You may go to the Ray dashboard (at `http://127.0.0.1:8265/` by default; may require port forwarding if you are ssh'ing onto a remote machine; if you are ssh'ing using vscode, the port forwarding should be configured automatically once ray runtime starts) to monitor the progress of the experiments.

The first script (`run_all_ReGate.sh`) will take approximately 120 core-hours. It will sweep through all possible NPU pod configurations (NPU version, number of chips, data/tensor/pipeline parallelisms, batch size) for all models, and run the performance, power, and carbon simulation experiments for each configuration. Then, it will run the SLO analysis to pick the best configuration for each model and each NPU version. The output results are used to generate the figures in the power-gating opportunity analysis and evaluation sections of the paper.

The second script (`run_ReGate_sens.sh`) will take approximately 40 core-hours. It will run the experiments for the sensitivity analysis figures in the evaluation section of the paper.

Please refer to the comments in the scripts for a detailed explanation of each step.
We strongly recommend running the scripts inside a persistent terminal session like `tmux`, as the experiments may run for several hours depending on the machine configuration.

After finishing all experiments, the output results will be stored in the `trace_util/llm_ops_generator/results` directory by default. Please make sure all experiment jobs finish successfully in the Ray dashboard. If any job fails, please check the logs in the Ray dashboard.


## 3. Evaluation and Expected Results

To reproduce the key figures in the paper, please go to the `trace_util/llm_ops_generator/graphs/graphs_micro25` directory.
Before plotting the graphs, please make sure to set up the environment variables in step 5 of [1. Installation & Setup](#1-installation--setup).

Then, export the environment variable for the results directory:
```bash
export RESULTS_DIR=$NPUSIM_HOME/trace_util/llm_ops_generator/results/
```

To generate all figures, simply run:
```bash
make
```
The generated figures will be under `trace_util/llm_ops_generator/graphs/graphs_micro25/outputs`.

Alternatively, to separately generate the figures for the NPU power-gating opportunity analysis section, run:
```bash
make motivation
```

To separately generate the figures for the evaluation section, run:
```bash
make evaluation
```

Please refer to [micro25_ae_figures.md](trace_util/llm_ops_generator/graphs/graphs_micro25/micro25_ae_figures.md) for a checklist of all key figures to be reproduced, along with their corresponding file names. To verify the results, one can compare the generated figures directly with those in the paper.


## 4. Experiment Customization


### Customizing Simulation Parameters

#### Output Directory
To customize the output directory for the simulation results:
1. Modify the `RESULTS_DIR` variable in `trace_util/llm_ops_generator/run_scripts/runtime_env.json`.
2. In `trace_util/llm_ops_generator/run_scripts`, change the `export` statement in all the bash scripts accordingly. For example, in `test.sh`, change:
   ```bash
   export RESULTS_DIR="$NPUSIM_HOME/trace_util/llm_ops_generator/results"
   ```
   to your desired output directory.
   The `RESUTS_DIR` environment variable also needs to be updated for plotting the figures in [3. Evaluation and Expected Results](#3-evaluation-and-expected-results).

#### Performance Simulation Parameters
The user can change the NPU hardware configuration and the model architecture of the simulation by creating new configuration files under `trace_util/llm_ops_generator/configs`.
We provide a set of pre-defined configurations in the `trace_util/llm_ops_generator/configs` directory:
- `trace_util/llm_ops_generator/configs/chips`: contains the NPU chip parameters, such as the number of SAs, VUs, core frequency, HBM bandwidth, on-chips SRAM size, etc.
- `trace_util/llm_ops_generator/configs/models`: contains the model architecture parameters as well as the parallelism configurations. We currently support LLMs (Llama and DeepSeek), DLRM, DiT-XL, and GLIGEN. See [Adding New DNN Models](#adding-new-dnn-models) for more details on how to add support for new models.
- `trace_util/llm_ops_generator/configs/systems`: contains the system-level parameters, including the datacenter power usage efficiency (PUE) and carbon intensity used for carbon emission analysis.

See `trace_util/llm_ops_generator/run_scripts/run_sim.py` for more details on how to launch the simulation with the new configuration file.

#### Power Simulation Parameters
The power gating parameters are define in `trace_util/llm_ops_generator/power_analysis_lib.py`. The user can modify the `get_power_gating_config()` function to add new power gating configurations, including power gating wake-up/delay cycles and power gating policies for each component.

See `trace_util/llm_ops_generator/run_scripts/energy_operator_analysis_main.py` for more details on how to launch the power simulation with the new configuration file.
For carbon emission analysis, see `trace_util/llm_ops_generator/run_scripts/carbon_analysis_main.py` for more details on how to launch the carbon emission analysis with different parameters, including carbon intensity, NPU duty cycle (utilization), and device lifetime.

### Running a Single Tensor Operator
Please see `trace_util/llm_ops_generator/run_scripts/run_single_op_main.py` for an example of how to run a single tensor operator simulation. This tool could be helpful for analyzing a specific operator of interest rather than simulating the entire DNN model.

### Running a Single Experiment
To run a single experiment, you can directly use the provided ops generator classes.
See the [README](trace_util/llm_ops_generator/README.md) under the `llm_ops_generator` directory.

### Adding New DNN Models
We currently support LLMs (see `trace_util/llm_ops_generator/llm_ops_generator.py`), DLRM (see `trace_util/llm_ops_generator/dlrm_ops_generator.py`), DiT-XL (see `trace_util/llm_ops_generator/dit_ops_generator.py`), and GLIGEN (see `trace_util/llm_ops_generator/gligen_ops_generator.py`). Variants of these models (such as changing the number of layers or hidden dimensions) can be created by adding new configuration files in the `trace_util/llm_ops_generator/configs/models` directory. To add support for new model architectures, the user needs to implement a new model generator class in the `trace_util/llm_ops_generator` directory to reflect the model's dataflow graph. Many commonly used operators such as GEMM, Conv, and LayerNorm are implemented in `trace_util/npusim_backend/npusim_lib.py`. Please refer to the existing model generator classes for examples on how to call these operators and implement new model generators.

### Running on a Ray Cluster
To scale out the simulator on multiple machines, we need to set up a shared storage directory and configure the Ray cluster. The instructions below shows an example of setting up a shared NFS directory and configuring a Ray cluster.

1. The NFS server can be any node in the cluster (preferably the head node).
    To set up NFS directory, run:
    ```bash
    sudo apt install nfs-kernel-server
    sudo mkdir -p /mnt/npusim_nfs_share
    sudo chown nobody:nogroup /mnt/npusim_nfs_share
    sudo chmod 777 /mnt/npusim_nfs_share
    echo "/mnt/npusim_nfs_share *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
    sudo exportfs -a
    sudo systemctl restart nfs-kernel-server
    ```
2. On each worker node, mount the NFS directory:
    ```bash
    sudo apt install nfs-common
    sudo mkdir -p /mnt/npusim_nfs_share
    sudo mount -t nfs [head_node_ip]:/mnt/npusim_nfs_share /mnt/npusim_nfs_share
    ```

3. The GitHub repository should be cloned inside the shared NFS directory to ensure all nodes have access to the codebase.

4. The conda environment `regate` with the pip packages must be installed on all nodes. See the above section for more details.

5. Launch ray runtime on the head node with the `regate` conda environment:
    ```bash
    conda activate regate
    ray start --head --port=6379
    ```
6. Finally, start the ray runtime on each worker node with the `regate` conda environment:
    ```bash
    conda activate regate
    ray start --address='[head_node_ip]:6379'
    ```

    You may verify all nodes are connected to the Ray cluster by running:
    ```bash
    ray status
    ```

    Alternatively, the "Cluster" tab in the Ray dashboard also shows the status of all nodes in the cluster.

7. The provided scripts under `trace_util/llm_ops_generator/run_scripts` can be launched on any node. Assuming we launch the scripts from the head node, we need to export the same environment variables as in the single machine setup:
   ```bash
   export PYTHONPATH=/mnt/npusim_nfs_share/[git_repo_dir]:$PYTHONPATH
   export NPUSIM_HOME=/mnt/npusim_nfs_share/[git_repo_dir]
   ```

   Make sure the path uses the NFS shared directory, not the local path, as this path will be used by other nodes in the cluster.

8. Change the `trace_util/llm_ops_generator/run_scripts/runtime_env.json` file to use the path with the NFS shared directory.

9. Run the test script to verify the setup:
   ```bash
   cd /mnt/npusim_nfs_share/[git_repo_dir]/trace_util/llm_ops_generator/run_scripts
   ./test.sh
   ```

   The test script will run the same tests as in the single machine setup, but Ray will automatically distribute the jobs across all nodes.

10. Other experiment scripts can be launched in the same way as in the single machine setup, but make sure to use the NFS shared directory for `NPUSIM_HOME`, `RESULTS_DIR`, and `PYTHONPATH`.


## Citation

(To be updated after the camera ready DOI is released.)
If you use this codebase in your research, please cite our [paper](link):
```bibtex
@inproceedings{xue2025regate,
  title={ReGate: Enabling Power Gating in Neural Processing Units},
  author={Xue, Yuqi and Huang, Jian},
  booktitle={Proceedings of the 58th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2025},
}
```
