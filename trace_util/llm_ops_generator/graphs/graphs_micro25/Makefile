PYTHON_EXEC=/usr/bin/env python

all: motivation evaluation


motivation: motiv_energy motiv_energy_breakdown motiv_utilization

motiv_energy: create_dir
	$(PYTHON_EXEC) e2e_energy_efficiency.py --pg_strategy=NoPG

motiv_energy_breakdown: create_dir
	$(PYTHON_EXEC) breakdown_energy.py --pg_strategy=NoPG

motiv_utilization: create_dir
	$(PYTHON_EXEC) sa_temporal_utilization.py
	$(PYTHON_EXEC) sa_spatial_utilization.py
	$(PYTHON_EXEC) vu_temporal_utilization.py
	$(PYTHON_EXEC) ici_temporal_utilization.py
	$(PYTHON_EXEC) hbm_temporal_utilization.py
	$(PYTHON_EXEC) max_vmem_size_demand_distribution.py


evaluation: eval_breakdown eval_perf_impact eval_num_setpm eval_sens eval_carbon eval_lca

eval_breakdown: create_dir
	$(PYTHON_EXEC) eval_energy_breakdown.py
	$(PYTHON_EXEC) eval_power_breakdown.py

eval_perf_impact: create_dir
	$(PYTHON_EXEC) eval_perf_impact.py

eval_sens: create_dir
	$(PYTHON_EXEC) eval_sens_npu_gen.py
	$(PYTHON_EXEC) eval_sens_Vth.py
	$(PYTHON_EXEC) eval_sens_pg_delay.py

eval_carbon: create_dir
	$(PYTHON_EXEC) eval_carbon.py

eval_lca: create_dir
	$(PYTHON_EXEC) carbon_efficiency_lca.py

eval_num_setpm: create_dir
	$(PYTHON_EXEC) eval_num_setpm.py

simulator_validation: create_dir
	$(PYTHON_EXEC) simulator_validation_e2e_time.py


create_dir:
	mkdir -p outputs

clean:
	rm -rf outputs


.PHONY: clean all create_dir motivation motiv_energy motiv_energy_breakdown motiv_utilization eval_carbon eval_breakdown eval_perf_impact eval_sens eval_lca eval_num_setpm simulator_validation
