import numpy as np
from dataclasses import dataclass
import pickle
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import json
import math

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
        ObjectiveThroughput: float
        Weight_F: float
        Weight_TP: float
        Weight_C: float

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
    class ThroughtputHistory:
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
    ThroughtputHistory: ThroughtputHistory
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
                ObjectiveThroughput=self.ga_object.objective_TP,            
                Weight_F=self.ga_object.w_f,
                Weight_TP=self.ga_object.w_tp,
                Weight_C=self.ga_object.w_c
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
            ThroughtputHistory=ExperimentConfig.ThroughtputHistory(
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

    def save_data(self, cost, fidelity, throughput):
        # self.exper_config.ParameterHistory.Loss.append(self.ga_object.population[0].genotype[0])

        assert len(cost) == len(fidelity)

        self.exper_config.FidelityHistory.All.append(fidelity)
        self.exper_config.CostHistory.All.append(cost)
        
        self.exper_config.FidelityHistory.Max.append(max(fidelity))
        self.exper_config.FidelityHistory.Mean.append(np.mean(fidelity))
        self.exper_config.FidelityHistory.Min.append(min(fidelity))
        
        self.exper_config.ThroughtputHistory.Max.append(max(throughput))
        self.exper_config.ThroughtputHistory.Mean.append(np.mean(throughput))
        self.exper_config.ThroughtputHistory.Min.append(min(throughput))

        self.exper_config.CostHistory.Max.append(max(cost))
        self.exper_config.CostHistory.Mean.append(np.mean(cost))
        self.exper_config.CostHistory.Min.append(min(cost))

    def plot_pareto_frontier(self, Xs, Ys, maxX=True, maxY=True, fig=None, ax=None):
        '''Pareto frontier selection process'''
        sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if maxY:
                if pair[1] >= pareto_front[-1][1]:
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)            
        
        '''Plotting process'''
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.scatter(Xs, Ys, label='Data')
        pf_X = [pair[0] for pair in pareto_front]
        pf_Y = [pair[1] for pair in pareto_front]
        ax.plot(pf_X, pf_Y, 'x-', color='red', label='Optimal Pareto')
        ax.set_xlabel("1 - Fidelity")
        ax.set_ylabel("Cost")
        ax.set_title("Pareto frontier")
        ax.legend(loc="upper right", fontsize=13)
        plt.show()

        return pf_X, pf_Y, fig, ax

    def plot(self):
        sns.set_theme(style="darkgrid")

        x = np.arange(0, len(self.exper_config.FidelityHistory.Max), 1)
        weight_f = self.exper_config.GeneticAlgorithmConfig.Weight_F
        weight_tp = self.exper_config.GeneticAlgorithmConfig.Weight_TP
        weight_c = self.exper_config.GeneticAlgorithmConfig.Weight_C
        objective_fidelity = self.exper_config.GeneticAlgorithmConfig.ObjectiveFidelity
        mutation_rate = self.exper_config.GeneticAlgorithmConfig.MutationRate
        num_population = self.exper_config.GeneticAlgorithmConfig.PopulationSize

        fig, ax = plt.subplots(3, 2)
        ax[0, 0].set_title(f'w: {weight_f}, mr: {mutation_rate}, pop: {num_population}', loc='right')

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

        ax[2, 0].set_title('Throughput of each generation')
        ax[2, 0].set_xlabel('generation')
        ax[2, 0].set_ylabel('throughput')
        ax[2, 0].plot(x, self.exper_config.ThroughtputHistory.Max, label='max throughput', color='tab:red')
        ax[2, 0].plot(x, self.exper_config.ThroughtputHistory.Min, label='min throughput', color='tab:green')
        ax[2, 0].plot(x, self.exper_config.ThroughtputHistory.Mean, label='mean throughput', color='tab:blue')
        ax[2, 0].fill_between(x, self.exper_config.ThroughtputHistory.Max, self.exper_config.ThroughtputHistory.Min, alpha=0.2, label='range throughput')
        ax[2, 0].legend()

        ax[2, 1].set_title('Final Generation Pareto Fronteir')
        inverse_fidelity = []
        for data_point in self.exper_config.FidelityHistory.All[-1]:
            inv_F = 1 - data_point
            inverse_fidelity.append(inv_F)

        pX, pY, fig, ax[2, 1] = self.plot_pareto_frontier(inverse_fidelity, self.exper_config.CostHistory.All[-1], maxX=False, maxY=False, fig=fig, ax=ax[2, 1])
        fig.tight_layout()
        plt.show()

        print(f'Experiment Name: {self.name}')        

def read_config(filename):
    with open(f'configs/{filename}', 'r') as f:
        config = json.load(f)
    return config

def decorate_prompt(prompt, experiment_name, weight_f, weight_tp, weight_c, mutationRate, numIndividual, parent_size, numGeneration, QnetGeneration, num_hops, parameterMax, node_type):
    decorated_prompt = f"+{'=' * 58}+\n"
    decorated_prompt += f"| {prompt}\n"
    decorated_prompt += f"+{'=' * 58}+\n"
    decorated_prompt += f'Date Create: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
    decorated_prompt += f'+{"-" * 58}+\n'
    decorated_prompt += f'| Hyperparameters:\n'
    decorated_prompt += f'+{"-" * 58}+\n'
    decorated_prompt += f'| Experiment Name:           {experiment_name}\n'
    decorated_prompt += f'| weight_f (Fidelity):        {weight_f}\n'  
    decorated_prompt += f'| weight_tp (Throughput):     {weight_tp}\n'
    decorated_prompt += f'| weight_c (Cost):            {weight_c}\n'
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
    decorated_prompt += f'| Type of node:              {node_type}\n'
    decorated_prompt += f'+{"-" * 58}+\n'
    return decorated_prompt

@dataclass
class Location:
    """Class for loss parameters"""
    lat_deg: int
    lat_min: int
    lat_sec: float
    lat_dir: str
    lon_deg: int
    lon_min: int
    lon_sec: float
    lon_dir: str

# Function to convert degrees, minutes, and seconds to decimal degrees
def dms_to_dd(degrees, minutes, seconds, direction):
    dd = degrees + minutes/60 + seconds/3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

def convert_ll_to_xyz(node: Location):
    lat_dd = dms_to_dd(node.lat_deg, node.lat_min, node.lat_sec, node.lat_dir)
    lon_dd = dms_to_dd(node.lon_deg, node.lon_min, node.lon_sec, node.lon_dir)

    # Earth radius in kilometers (mean radius)
    earth_radius_km = 6371.0

    # Convert to XYZ coordinates
    x = earth_radius_km * math.cos(math.radians(lat_dd)) * math.cos(math.radians(lon_dd))
    y = earth_radius_km * math.cos(math.radians(lat_dd)) * math.sin(math.radians(lon_dd))
    z = earth_radius_km * math.sin(math.radians(lat_dd))

    return x, y, z

# def cal_relative_distance(node1: Location, node2: Location):

#     lat_dd_1 = dms_to_dd(node1.lat_deg, node1.lat_min, node1.lat_sec, node1.lat_dir)
#     lon_dd_1 = dms_to_dd(node1.lon_deg, node1.lon_min, node1.lon_sec, node1.lon_dir)
#     lat_dd_2 = dms_to_dd(node2.lat_deg, node2.lat_min, node2.lat_sec, node2.lat_dir)
#     lon_dd_2 = dms_to_dd(node2.lon_deg, node2.lon_min, node2.lon_sec, node2.lon_dir)

#     earth_radius_km = 6371.0

#     x_1 = earth_radius_km * math.cos(math.radians(lat_dd_1)) * math.cos(math.radians(lon_dd_1))
#     y_1 = earth_radius_km * math.cos(math.radians(lat_dd_1)) * math.sin(math.radians(lon_dd_1))
#     z_1 = earth_radius_km * math.sin(math.radians(lat_dd_1))
    
#     x_2 = earth_radius_km * math.cos(math.radians(lat_dd_2)) * math.cos(math.radians(lon_dd_2))
#     y_2 = earth_radius_km * math.cos(math.radians(lat_dd_2)) * math.sin(math.radians(lon_dd_2))
#     z_2 = earth_radius_km * math.sin(math.radians(lat_dd_2))

#     return math.sqrt((x_1-x_2)**2 + (y_1-y_2)**2 + (z_1-z_2)**2)
