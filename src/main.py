import numpy as np
from Genetic_opt.Genetic_algorithm import GeneticAlgorithm
from simulator import QwantaSimulation
from all_function import *
from tqdm import tqdm
import multiprocessing
import os

@dataclass
class ParameterHistory:
    Loss: list
    GateError: list
    MeasurementError: list
    MemoryTime: list

def main_process(
        experiment_name,
        elitism,
        population_size,
        mutation_rate,
        mutation_sigma,
        amount_optimization_steps,
        weight,
        objective_fidelity,
        num_hops,
        excel_file = "exper_id3_selectedStats_2hops.xlsx",
        QnetGeneration = "0G",        
    ):    

    # Define ga object
    ga = GeneticAlgorithm(dna_size = 4, elitism = elitism, population_size = population_size, mutation_rate = mutation_rate, mutation_sigma = mutation_sigma, objective_F=objective_fidelity, weight=weight)    

    # Define result object for save data in simulation
    result = ExperimentResult(ga_object=ga, experiment_name=experiment_name, qnet_generation=QnetGeneration, num_hops=num_hops)

    decorated_prompt = decorate_prompt(
        prompt = "This is your Genetic simulation hyperparameter...",
        experiment_name = experiment_name,
        weight = weight,        
        mutationRate = mutation_rate,
        numIndividual = population_size,
        parent_size = int(population_size*elitism),
        numGeneration = amount_optimization_steps,  
        QnetGeneration = QnetGeneration,
        num_hops = num_hops,
    )
    print(decorated_prompt)

    for step in tqdm(range(amount_optimization_steps)):
        
        fidelity_history = []      
        parameter = ParameterHistory(Loss=[], GateError=[], MeasurementError=[], MemoryTime=[])  

        # this loop is feed all ind to simulator
        for ind in ga.population:
        # define Qwanta simulation class
            sim_ind = paramsTransform(ind)
            # print(f'simulation with parameter set: {sim_ind}')            
            qwan_sim = QwantaSimulation(
                parameter_set = sim_ind, 
                num_hops = num_hops, 
                network_strategy=excel_file,
                network_generation=QnetGeneration,
            )            
            simulate = qwan_sim.execute()
            simulate_fidelity = simulate[QnetGeneration]['fidelity']            
            fidelity_history.append(simulate_fidelity)

            # collect parameter history in each step
            parameter.Loss.append(sim_ind.genotype[0])
            parameter.GateError.append(sim_ind.genotype[1])
            parameter.MeasurementError.append(sim_ind.genotype[2])
            parameter.MemoryTime.append(sim_ind.genotype[3])

        # save parameter history in each generation
        result.save_parameter_history(parameter_set=parameter)
        
        qwan_sim.collect_fidelity_history(fidelity_history)

        # collect cost and replace old population with new population
        cost = ga.evole(simulate_F=simulate_fidelity)
        # print(f'cost: {cost}')

        # save data in each step
        result.save_data(cost=cost, fidelity=fidelity_history)

    result.save_experiment(file_name=experiment_name, ga_object=ga)

def process_config(config_file_name):
    config = read_config(config_file_name)
    
    experiment_name = config['experiment_name']
    elitism = config['elitism']
    population_size = config['population_size']
    mutation_rate = config['mutation_rate']
    mutation_sigma = config['mutation_sigma']
    amount_optimization_steps = config['amount_optimization_steps']
    weight = config['weight']
    objective_fidelity = config['objective_fidelity']
    num_hops = config['num_hops']
    excel_file = config['excel_file']
    QnetGeneration = config['QnetGeneration']

    main_process(
        experiment_name=experiment_name,
        elitism = elitism,
        population_size = population_size,
        mutation_rate = mutation_rate,
        mutation_sigma = mutation_sigma,        
        amount_optimization_steps = amount_optimization_steps,                
        weight = weight,    
        objective_fidelity = objective_fidelity,
        num_hops = num_hops,
        excel_file = excel_file,
        QnetGeneration = QnetGeneration
    )

if __name__ == "__main__":

    user_choice = input("Run all config file? (Y/n): ").strip().lower()

    if user_choice == 'y':

        config_directory = "configs/"  # Update this to the directory where your config files are located
    
        # List all the files in the config directory
        config_files = os.listdir(config_directory)

        # Create a multiprocessing pool with the number of desired parallel processes
        num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Use pool.map to execute process_config in parallel for each config file
        pool.map(process_config, [filename for filename in config_files if filename.endswith(".json")])
        
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        

    elif user_choice == 'n':
        # Ask the user for the specific config file name to run
        config_filename = input("Enter the config file name to run: ").strip()
        process_config(config_filename)
        
    else:
        print("Invalid choice. Please enter 'all' or 'one'.")
