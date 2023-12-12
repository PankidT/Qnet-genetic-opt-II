import json
import pandas as pd
import numpy as np

# %%
loss_list =  np.array([0.03])
p_dep_list = np.array([0.025])
gate_error_list = np.array([0, 0.0025])
mem_error_list = np.array([0.01])
measurement_error_list =  np.array([0, 0.0025])
elitism_list = np.array([0.2])
population_size_list = np.array([10])
mutation_rate_list = np.array([0.3])

mutation_sigma_list = np.array([0.3])
weight_f_list = np.array([0.5])
weight_tp_list = np.array([0.5])
weight_c_list = np.array([0.5])
num_hops_list = np.array([2, 4])

objective_fidelity = 0.7
objective_throughput = 4000

parameters_set = []; index = 0
for loss in loss_list:
    for p_dep in p_dep_list:
        for gate_error in gate_error_list:
            for mem_error in mem_error_list:
                for measurement_error in measurement_error_list:
                    for elitism in elitism_list:                        
                        for population_size in population_size_list:
                            for mutation_rate in mutation_rate_list:
                                for mutation_sigma in mutation_sigma_list:
                                    for weight_f in weight_f_list:
                                        for weight_tp in weight_tp_list:
                                            for weight_c in weight_c_list:
                                                for num_hops in num_hops_list:
                                                    simulation_parameters = {
                                                        "experiment_name": f"experiment_{index}",
                                                        "elitism": float(elitism),
                                                        "population_size": float(population_size),
                                                        "mutation_rate": float(mutation_rate),
                                                        "mutation_sigma": float(mutation_sigma),
                                                        "amount_optimization_steps": 5,
                                                        "weight_f": float(weight_f),
                                                        "weight_tp": float(weight_tp),
                                                        "weight_c": float(weight_c),
                                                        "objective_fidelity": float(objective_fidelity),
                                                        "objective_throughput": float(objective_throughput),
                                                        "num_hops": float(num_hops),
                                                        "loss_max": float(loss),
                                                        "coherence_max": float(p_dep),
                                                        "gateErr_max": float(gate_error),
                                                        "meaErr_max": float(measurement_error),
                                                        "excel_file": "exper_id3_selectedStats_2hops.xlsx",
                                                        "QnetGeneration": "0G",
                                                        "use_custom_node": "false"
                                                    }
                                                    
                                                    parameters_set.append(simulation_parameters)
                                                    json_object = json.dumps(parameters_set, indent = 4)
                                                    index += 1

                                                    # print(simulation_parameters)

                                                    with open(f'sim_set_{index}.json', "w") as f:
                                                        f.write(json_object)

DataFrame = pd.DataFrame(parameters_set)
DataFrame.to_excel(f'simulation_parameters.xlsx', index=False)

