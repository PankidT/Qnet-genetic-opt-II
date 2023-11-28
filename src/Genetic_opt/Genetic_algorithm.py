import numpy as np
import random
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from .selection_fn import roulette_wheel_selection, rw_genetic
from .cost_fn.multi_object_cost import total_cost_fn
from all_function import Individual

class GeneticAlgorithm:
    def __init__(self,                 
                 dna_size: int,
                 dna_bound: List[float]=[0, 1],
                 dna_start_position: float=0.5,
                 elitism:float=0.01,
                 population_size:int=200,
                 mutation_rate:float=0.01,
                 mutation_sigma:float=0.1,
                 objective_F:float=0.7,
                 objective_TP:float=4000,
                 weight_f:float=0.5,
                 weight_tp:float=0.5,
                 weight_c:float=1,                
                 mutate_fn=None,
                 crossover_fn=None,                              
                 ):

        self.population = self.__create_random_population(dna_size, mutation_sigma, dna_start_position, population_size, dna_bound)
        
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.dna_bounds = dna_bound

        self.population_size = population_size
        self.dna_start_position = dna_start_position
        self.elitism = elitism

        self.amount_new = int(population_size * elitism)
        self.amount_old = population_size - self.amount_new

        # random baseline value shape equal dna_size and value between 0 and 1
        self.baseline_value = np.array([np.random.rand() for i in range(dna_size)])
        self.w_f = weight_f
        self.w_tp = weight_tp
        self.w_c = weight_c
        self.objective_F = objective_F
        self.objective_TP = objective_TP

        self.min_cost_history = []
        self.avg_cost_history = []

        self.mutate_fn = mutate_fn
        self.crossover_fn = crossover_fn

    def __create_random_population(self, dna_size: int, mutation_sigma: float, dna_start_position: float, population_size: int, dna_bounds: Tuple[float, float] = (0, 1)) -> List[Individual]: 

        population = np.random.randn(population_size, dna_size) * mutation_sigma + dna_start_position
        np.clip(population, *dna_bounds, out=population)

        DataClass_Population = [
            Individual(genotype=population[i]) for i in range(population_size)
        ]

        return DataClass_Population

    def __crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:    

        if self.crossover_fn is not None:
            return self.crossover_fn(parent1, parent2)        

        # Select a random crossover point
        while True:
            crossover_point = random.randint(0, len(parent1.genotype) - 1)            
            if crossover_point != 0:
                break

        offspring1 = Individual(genotype=np.zeros(len(parent1.genotype)))
        offspring2 = Individual(genotype=np.zeros(len(parent1.genotype)))

        # Perform crossover
        for i in range(len(parent1.genotype)):
            if i < crossover_point:
                offspring1.genotype[i] = parent1.genotype[i]
                offspring2.genotype[i] = parent2.genotype[i]
                
            else:
                offspring1.genotype[i] = parent2.genotype[i]
                offspring2.genotype[i] = parent1.genotype[i]

        return offspring1, offspring2
    
    def __mutate(self, chromosome: Individual) -> Individual:

        if self.mutate_fn is not None:
            return self.mutate_fn(chromosome)

        mutated_chromosome = Individual(genotype=np.zeros(len(chromosome.genotype)))

        for gene, i in zip(chromosome.genotype, range(len(chromosome.genotype))):

            if random.random() < self.mutation_rate:
                mutation_gene = np.random.randn() * self.mutation_sigma            
                gene += mutation_gene
                gene = min(max(gene, self.dna_bounds[0]), self.dna_bounds[1])
                mutated_chromosome.genotype[i] += gene
            else:
                mutated_chromosome.genotype[i] += gene

        return mutated_chromosome
    
    def evole(self, simulate_F, simulate_TP):

        """
        baseline_value: control parameter
            - from now, cannot decision to receive from GeneticAlgorithm class or just create inside class already.
        w: weight
        objective_F: target fidelity
        simulate_F: calculate repeatedly from control parameter

        define ga class -> feed population into simulator -> get fidelity -> calculate cost (ga.evole) and cost collected in each individual class in population -> 
        """
        new_population = []

        # calculate cost for current population
        for individual in self.population:            
            individual.cost = total_cost_fn(
                baseline_value=self.baseline_value, 
                chromosome=individual, 
                w1=self.w_f, 
                w2=self.w_tp,
                w3=self.w_c,
                simulate_F=simulate_F,
                simulate_TP=simulate_TP,
                objective_F=self.objective_F,
                objective_TP=self.objective_TP
                )

        # collect cost in this population's generation before generate new population
        minimum_cost = np.min([individual.cost for individual in self.population])
        avg_cost = np.mean([individual.cost for individual in self.population])

        self.min_cost_history.append(minimum_cost)
        self.avg_cost_history.append(avg_cost)

        # Generate new population
        for i in range(self.amount_new // 2):
            parent_1, parent_2 = rw_genetic(self.population)            
            offspring_1, offspring_2 = self.__crossover(parent_1, parent_2)            
            offspring_1 = self.__mutate(offspring_1)
            offspring_2 = self.__mutate(offspring_2)

            # calcuate cost for new offspring            
            offspring_1.cost = total_cost_fn(
                baseline_value=self.baseline_value, 
                chromosome=offspring_1, 
                w1=self.w_f, 
                w2=self.w_tp,
                w3=self.w_c,
                simulate_F=simulate_F, 
                simulate_TP=simulate_TP,
                objective_F=self.objective_F,
                objective_TP=self.objective_TP)
            offspring_2.cost = total_cost_fn(
                baseline_value=self.baseline_value, 
                chromosome=offspring_2, 
                w1=self.w_f, 
                w2=self.w_tp,
                w3=self.w_c,
                simulate_F=simulate_F, 
                simulate_TP=simulate_TP,
                objective_F=self.objective_F,
                objective_TP=self.objective_TP)

            new_population.append(offspring_1)
            new_population.append(offspring_2)
            
        assert len(new_population) == self.amount_new

        for i in range(self.amount_old):
            index = roulette_wheel_selection(self.population)
            new_population.append(self.population[index])


        assert len(new_population) == len(self.population)
        
        self.population = new_population

        return [individual.cost for individual in self.population]

    def plot_cost_history(self):
        plt.plot(self.min_cost_history, label='Minimum Cost')
        plt.plot(self.avg_cost_history, label='Average Cost')
        plt.legend()
        plt.ylabel('Cost')
        plt.xlabel('Generation')
        plt.show()