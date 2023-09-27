import numpy as np
from dataclasses import dataclass
from all_function import Individual

def cost_C(baseline_value, chromosome: Individual) -> float:   
    """
    This function calculate the cost of the chromosome from loss parameter (gene in chromosome)
    """    

    loss_cost = 0
    for i in range(len(baseline_value)):

        gene = chromosome.genotype[i]        

        if gene == 0 or gene == 1:
            gene = np.random.rand()

        k = np.log(baseline_value[i])/np.log(gene)
        loss_cost += k

    return loss_cost

def cost_F(objective_F: float, simulate_F: float):
    """
    This function calculate the cost of the chromosome from fidelity (gene in chromosome)
    """
    def Heaviside(x):
        if x >= 0:
            return 1
        else:
            return 0
        
    fidelity_cost = Heaviside(objective_F - simulate_F)
    return fidelity_cost

def total_cost_fn(baseline_value, chromosome: Individual, w: float, simulate_F: float, objective_F: float =0.7):
    """
    This function calculate the total cost of the chromosome from cost_c and cost_f
    """

    cost_c = cost_C(baseline_value, chromosome)
    cost_f = cost_F(objective_F, simulate_F)

    total_cost = w*cost_f + cost_c
    return total_cost