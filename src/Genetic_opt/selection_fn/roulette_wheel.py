from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from all_function import Individual

def roulette_wheel_selection(population: List[Individual]) -> int:

    cost_values = [individual.cost for individual in population]

    # Invert the cost values (higher costs become lower, and vice versa)
    inverted_costs = [1.0 / cost for cost in cost_values]

    # Calculate the total inverted cost
    total_inverted_cost = sum(inverted_costs)

    # Generate a random number between 0 and the total inverted cost
    random_value = np.random.uniform(0, total_inverted_cost)

    # Initialize variables for tracking cumulative inverted cost and the selected index
    cumulative_inverted_cost = 0
    selected_index = 0

    # Iterate through the inverted cost values to find the selected individual
    for i, inverted_cost in enumerate(inverted_costs):
        cumulative_inverted_cost += inverted_cost
        if cumulative_inverted_cost >= random_value:
            selected_index = i
            break

    return selected_index

def rw_genetic(population: List[Individual]) -> Tuple[Individual, Individual]:

    while True:
        index_p1 = roulette_wheel_selection(population)
        index_p2 = roulette_wheel_selection(population)        
        while index_p1 == index_p2:
            index_p2 = roulette_wheel_selection(population)                        
        else:
            break

    return population[index_p1], population[index_p2]

dna1 = Individual(genotype=np.array([0,0,0,0,0,0]), cost=100)
dna2 = Individual(genotype=np.array([1,1,1,1,1,1]), cost=1)
dna3 = Individual(genotype=np.array([1,1,1,1,1,1]), cost=10)

population = [dna1, dna2, dna3]
selected_index = roulette_wheel_selection(population)
print(selected_index)

rw_genetic(population)