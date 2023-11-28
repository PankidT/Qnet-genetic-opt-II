import numpy as np
from qwanta import Xperiment
from all_function import Individual, convert_ll_to_xyz, Location
import matplotlib.pyplot as plt

x_bkk, y_bkk, z_bkk = convert_ll_to_xyz(Location(lat_deg=13, lat_min=45, lat_sec=52, lat_dir=9, lon_deg=100, lon_min=31, lon_sec=31, lon_dir=5))
x_cm, y_cm, z_cm = convert_ll_to_xyz(Location(lat_deg=18, lat_min=56, lat_sec=8, lat_dir=7, lon_deg=98, lon_min=49, lon_sec=20, lon_dir=8))
x_sk, y_sk, z_sk = convert_ll_to_xyz(Location(lat_deg=6, lat_min=51, lat_sec=8, lat_dir=5, lon_deg=100, lon_min=34, lon_sec=36, lon_dir=6))

custom_node_info = {
    'Node 0': {'coordinate': (x_cm, y_cm, z_cm)},
    'Node 1': {'coordinate': (x_bkk, y_bkk, z_bkk)},
    'Node 2': {'coordinate': (x_sk, y_sk, z_sk)},
}

def adjusted_rate(rl, dl, ds, loss=0.1):
    return rl*10**((dl - ds)*loss/10)

class QwantaSimulation:
    """
    Class for Qwanta simulation:
    This class create object for Qwanta simulation with given parameter set.
    Also collect the fidelity history of the simulation.    
    """
    def __init__(self, 
            parameter_set: Individual, 
            num_hops: int, 
            network_strategy: str,
            network_generation: str ='0G',
            use_custom_node: str = "false"
        ):
        self.parameter_set = parameter_set
        self.depo_prob = 0
        self.network_generation = network_generation
        self.num_hops = num_hops
        self.num_nodes = self.num_hops + 1

        if use_custom_node == "false":
            self.node_info = {f'Node {i}': {'coordinate': (int(i*100), 0, 0)} for i in range(self.num_nodes)}
        else:
            self.node_info = custom_node_info

        self.edge_info = {
                    (f'Node {i}', f'Node {i+1}'): {
                        'connection-type': 'Space',
                        'depolarlizing error': [1 - self.depo_prob, self.depo_prob/3, self.depo_prob/3, self.depo_prob/3],
                        'loss': self.parameter_set.genotype[0],
                        'light speed': 300000,
                        'Pulse rate': 0.0001,
                    f'Node {i}':{
                        'gate error': self.parameter_set.genotype[1],
                        'measurement error': self.parameter_set.genotype[2],
                        'memory function': self.parameter_set.genotype[3]
                    },
                    f'Node {i+1}':{
                        'gate error': self.parameter_set.genotype[1],
                        'measurement error': self.parameter_set.genotype[2],
                        'memory function': self.parameter_set.genotype[3]
                    },
                    }
                for i in range(num_hops)}
        
        self.exps = Xperiment(
            timelines_path = f'networks/{network_strategy}',
            nodes_info_exp = self.node_info,
            edges_info_exp = self.edge_info,
            gate_error = self.parameter_set.genotype[1],
            measurement_error = self.parameter_set.genotype[2],
            memory_time = self.parameter_set.genotype[3],
            strategies_list=[self.network_generation]
            )
        
        self.min_fidelity_history = []
        self.avg_fidelity_history = []
        self.max_fidelity_history = []

        # self.simulation_results = None

    def collect_fidelity_history(self, results: list):
        self.min_fidelity_history.append(min(results))
        self.avg_fidelity_history.append(np.mean(results))
        self.max_fidelity_history.append(max(results))
        
    def execute(self):
        results = self.exps.execute()
        print(f'check node: {self.node_info}')
        # self.fidelity_history.append(results[self.network_generation]['fidelity'])
        # self.simulation_results = results
        return results
    
    def plot_fidelity_history(self):
        plt.plot(self.min_fidelity_history, label='min fidelity')
        plt.plot(self.avg_fidelity_history, label='avg fidelity')
        plt.plot(self.max_fidelity_history, label='max fidelity')
        plt.legend()
        plt.show()
