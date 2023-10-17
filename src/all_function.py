import numpy as np
from dataclasses import dataclass
import pickle
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import json

@dataclass
class Individual:
    """
    Every time crossover, mutated, or selected, a new individual is created, cost must = None
    """
    genotype: np.ndarray
    cost: float = None

def parameterTransform(loss, coherenceTime, gateErr, meaErr, maxLoss, maxCoherenceTime, maxGateErr, maxMeaErr):
    '''
    This function change the [0-1] value into real simulation value

    # value from Bsc_Thesis
    meaErr: 0.01, 0.03, 0.05 %
    memoryTime: 0.25, 0.5, 1 second
    gate error 0.01, 0.03 %
    loss: 0.001, 0.003, 0.005, 0.007 dB/km

    # plan to normalize value in interval of previous research
    meaErr: 0 - 0.10
    memoryTime: 0 - 1
    gateError: 0 - 0.05
    loss 0.001 - 0.010

    '''

    lossSim = maxLoss*(1 - loss) # loss sim interval [0, 0.01], 0-> 0.01, 1->0
    coherenceTimeSim = coherenceTime*maxCoherenceTime
    gateErrSim = maxGateErr*(1 - gateErr)
    meaErrSim = maxMeaErr*(1 - meaErr)

    # prevent 0 value for coherence time
    if coherenceTimeSim == 0:
        coherenceTimeSim = 0.0001

    return lossSim, coherenceTimeSim, gateErrSim, meaErrSim

def paramsTransform(chomosome: Individual, maxLoss: float, maxCoherenceTime: float, maxGateErr: float, maxMeaErr: float):
    '''
    This function transform the chromosome into parameter set for simulation
    '''
    loss, coherenceTime, gateErr, meaErr = chomosome.genotype
    lossSim, coherenceTimeSim, gateErrSim, meaErrSim = parameterTransform(loss, coherenceTime, gateErr, meaErr, maxLoss, maxCoherenceTime, maxGateErr, maxMeaErr)
    return Individual(genotype=np.array([lossSim, coherenceTimeSim, gateErrSim, meaErrSim]), cost=chomosome.cost)

@dataclass
class ExperimentConfig:

    @dataclass
    class QuantumNetworkSimulationConfig:
        NetworkGeneration: str
        NumHops: int

    @dataclass
    class GeneticAlgorithmConfig:
        PopulationSize: int
        DnaStartPosition: float
        Elitism: float
        MutationRate: float
        MutationSigma: float
        ObjectiveFidelity: float
        Weight: float

    @dataclass
    class ParameterHistory:
        Loss: list
        GateError: list
        MeasurementError: list
        MemoryTime: list

    @dataclass
    class FidelityHistory:
        Max: list
        Mean: list
        Min: list
        All: list

    @dataclass
    class CostHistory:
        Max: list
        Mean: list
        Min: list
        All: list

    ExperimentName: str
    DateCreated: str
    QuantumNetworkSimulationConfig: QuantumNetworkSimulationConfig
    GeneticAlgorithmConfig: GeneticAlgorithmConfig
    ParameterHistory: ParameterHistory
    FidelityHistory: FidelityHistory
    CostHistory: CostHistory

class ExperimentResult:
    def __init__(self, ga_object, experiment_name: str, qnet_generation: str, num_hops: int):

        # Before simulation, ga_object and simulator_object are None (use structure to define self.exper_config)
        self.ga_object = ga_object     
        self.simulator_object = None   
        self.name = experiment_name       

        self.exper_config = ExperimentConfig(
            ExperimentName=self.name,
            DateCreated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            QuantumNetworkSimulationConfig=ExperimentConfig.QuantumNetworkSimulationConfig(
                NetworkGeneration=qnet_generation,
                NumHops=num_hops
            ),
            GeneticAlgorithmConfig=ExperimentConfig.GeneticAlgorithmConfig(
                PopulationSize=self.ga_object.population_size,
                DnaStartPosition=self.ga_object.dna_start_position,
                Elitism=self.ga_object.elitism,
                MutationRate=self.ga_object.mutation_rate,
                MutationSigma=self.ga_object.mutation_sigma,
                ObjectiveFidelity=self.ga_object.objective_F,
                Weight=self.ga_object.w,                
            ),
            ParameterHistory=ExperimentConfig.ParameterHistory(
                Loss=[],
                GateError=[],
                MeasurementError=[],
                MemoryTime=[]
            ),
            FidelityHistory=ExperimentConfig.FidelityHistory(
                Max=[],
                Mean=[],
                Min=[],
                All=[]
            ),
            CostHistory=ExperimentConfig.CostHistory(
                Max=[],
                Mean=[],
                Min=[],
                All=[]
            )
        )

    def save_experiment(self, file_name: str, ga_object):  

        # save object here
        self.ga_object = ga_object        

        # data_to_save = {
        #     'ga_object': self.ga_object,            
        #     'experiment_config': self.exper_config
        # }

        file_name = f'result_{file_name}_{self.exper_config.DateCreated}.pickle'            

        # Pickle the data
        with open(f'Results/{file_name}', 'wb') as f:
            pickle.dump(self, f)

    def save_parameter_history(self, parameter_set: ExperimentConfig.ParameterHistory):

        self.exper_config.ParameterHistory.Loss.append(parameter_set.Loss)
        self.exper_config.ParameterHistory.MemoryTime.append(parameter_set.MemoryTime)
        self.exper_config.ParameterHistory.GateError.append(parameter_set.GateError)
        self.exper_config.ParameterHistory.MeasurementError.append(parameter_set.MeasurementError)        

    def save_data(self, cost, fidelity):
        # self.exper_config.ParameterHistory.Loss.append(self.ga_object.population[0].genotype[0])

        assert len(cost) == len(fidelity)

        self.exper_config.FidelityHistory.All.append(fidelity)
        self.exper_config.CostHistory.All.append(cost)
        
        self.exper_config.FidelityHistory.Max.append(max(fidelity))
        self.exper_config.FidelityHistory.Mean.append(np.mean(fidelity))
        self.exper_config.FidelityHistory.Min.append(min(fidelity))

        self.exper_config.CostHistory.Max.append(max(cost))
        self.exper_config.CostHistory.Mean.append(np.mean(cost))
        self.exper_config.CostHistory.Min.append(min(cost))

    def plot(self):
        sns.set_theme(style="darkgrid")

        x = np.arange(0, len(self.exper_config.FidelityHistory.Max), 1)
        weight = self.exper_config.GeneticAlgorithmConfig.Weight
        objective_fidelity = self.exper_config.GeneticAlgorithmConfig.ObjectiveFidelity
        mutation_rate = self.exper_config.GeneticAlgorithmConfig.MutationRate
        num_population = self.exper_config.GeneticAlgorithmConfig.PopulationSize

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].set_title(f'w: {weight}, mr: {mutation_rate}, pop: {num_population}', loc='right')

        color = 'tab:red'
        ax[0, 0].set_xlabel('generation')
        ax[0, 0].set_ylabel('fidelity', color=color)
        ax[0, 0].plot(x, self.exper_config.FidelityHistory.Mean, color=color)
        ax[0, 0].tick_params(axis='y', labelcolor=color)

        ax2 = ax[0, 0].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('cost', color=color)
        ax2.plot(x, self.exper_config.CostHistory.Mean, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ax[0, 1].scatter(self.exper_config.FidelityHistory.Mean, self.exper_config.CostHistory.Mean)
        ax[0, 1].set_xlabel('fidelity')
        ax[0, 1].set_ylabel('cost')

        ax[1, 0].set_title('Fidelity of each generation')
        ax[1, 0].set_xlabel('generation')
        ax[1, 0].set_ylabel('fidelity')
        ax[1, 0].plot(x, self.exper_config.FidelityHistory.Max, label='max fidelity', color='tab:red')
        ax[1, 0].plot(x, self.exper_config.FidelityHistory.Min, label='min fidelity', color='tab:green')
        ax[1, 0].plot(x, self.exper_config.FidelityHistory.Mean, label='mean fidelity', color='tab:blue')        
        ax[1, 0].fill_between(x, self.exper_config.FidelityHistory.Max, self.exper_config.FidelityHistory.Min, alpha=0.2, label='range fidelity')
        ax[1, 0].legend()

        ax[1, 1].set_title('Evolution of cost')
        color = 'tab:red'
        ax[1, 1].set_xlabel('generation')
        ax[1, 1].set_ylabel('avg cost', color=color)
        ax[1, 1].plot(x, self.exper_config.CostHistory.Mean, 'r.-', label='mean cost', color=color)
        ax[1, 1].tick_params(axis='y', labelcolor=color)

        ax2 = ax[1, 1].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('best cost', color=color)  # we already handled the x-label with ax1
        ax[1, 1].plot(x, self.exper_config.CostHistory.Min , 'b+',label='min cost', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax[1, 1].legend()

        print(f'Experiment Name: {self.name}')        

def read_config(filename):
    with open(f'configs/{filename}', 'r') as f:
        config = json.load(f)
    return config

def decorate_prompt(prompt, experiment_name, weight, mutationRate, numIndividual, parent_size, numGeneration, QnetGeneration, num_hops, parameterMax):
    decorated_prompt = f"+{'=' * 58}+\n"
    decorated_prompt += f"| {prompt}\n"
    decorated_prompt += f"+{'=' * 58}+\n"
    decorated_prompt += f'Date Create: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
    decorated_prompt += f'+{"-" * 58}+\n'
    decorated_prompt += f'| Hyperparameters:\n'
    decorated_prompt += f'+{"-" * 58}+\n'
    decorated_prompt += f'| Experiment Name:           {experiment_name}\n'
    decorated_prompt += f'| weight1 (Fidelity):        {weight}\n'    
    decorated_prompt += f'| Mutation Rate:             {mutationRate}\n'    
    decorated_prompt += f'| Number of Individuals:     {numIndividual}\n'
    decorated_prompt += f'| Parent Size before Crossover: {parent_size}\n'
    decorated_prompt += f'| Number of Generations:     {numGeneration}\n'    
    decorated_prompt += f'| Strategy:                  {QnetGeneration}\n'
    decorated_prompt += f'| Number of Hops:            {num_hops}\n'
    decorated_prompt += f'| Loss Max:                  {parameterMax[0][0]} ({parameterMax[0][0]*100} %)\n'
    decorated_prompt += f'| Coherence Max:             {parameterMax[1][0]} second\n'
    decorated_prompt += f'| Gate Error Max:            {parameterMax[2][0]} ({parameterMax[2][0]*100} %)\n'
    decorated_prompt += f'| Measurement Error Max:     {parameterMax[3][0]} ({parameterMax[3][0]*100} %)\n'
    decorated_prompt += f'+{"-" * 58}+\n'
    return decorated_prompt