import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import hiplot as hip
import json
from streamlit import runtime
runtime.exists()

def pareto_frontier_3d_v2(obj1, obj2, obj3, plot_frontier=False, pareto_function='min'):
    
        def is_pareto_efficient(costs):
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                is_efficient[i] = np.all(np.any(costs[:i] <= c, axis=1)) and np.all(np.any(costs[i + 1:] <= c, axis=1))
            return is_efficient

        def is_pareto_efficient_min(costs):
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                is_efficient[i] = np.all(np.any(costs[:i] >= c, axis=1)) and np.all(np.any(costs[i + 1:] >= c, axis=1))
            return is_efficient

        # Identify Pareto front points
        objectives = np.column_stack((obj1, obj2, obj3))
        if pareto_function == 'max':
            pareto_efficient = objectives[is_pareto_efficient(objectives)]
        elif pareto_function == 'min':
            pareto_efficient = objectives[is_pareto_efficient_min(objectives)]
        else:
            raise ValueError('pareto_function must be either "min" or "max"')

        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for all points with color based on obj3
        scatter = ax.scatter(obj1, obj2, obj3, c=obj3, cmap='viridis', label='All Points', alpha=0.8)

        # Scatter plot for Pareto front points
        if plot_frontier:
            ax.scatter(pareto_efficient[:, 0], pareto_efficient[:, 1], pareto_efficient[:, 2], c='red', marker='^' ,label='Pareto Front', s=100)

            # Project Pareto front points onto each axis plane
            # ax.scatter(pareto_efficient[:, 0], pareto_efficient[:, 1], np.zeros_like(pareto_efficient[:, 2]), c='blue', marker='o', alpha=0.5, label='Projection on XY Plane')
            # ax.scatter(pareto_efficient[:, 0], np.zeros_like(pareto_efficient[:, 1]), pareto_efficient[:, 2], c='green', marker='o', alpha=0.5, label='Projection on XZ Plane')
            # ax.scatter(np.zeros_like(pareto_efficient[:, 0]), pareto_efficient[:, 1], pareto_efficient[:, 2], c='purple', marker='o', alpha=0.5, label='Projection on YZ Plane')

        # Set axis labels
        ax.set_xlabel('1 - Fidelity')
        ax.set_ylabel('Cost')
        ax.set_zlabel('1/Throughput')

        # Set plot title
        plt.title('3D Pareto Front with Color')

        # Display colorbar for obj3. But convert from 1-TP to TP
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('1/Throughput', rotation=270, labelpad=20)

        # Display legend
        ax.legend()

        # Show the plot
        st.pyplot(fig)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('2D Projections of Pareto Front')
        ax[0].scatter(pareto_efficient[:, 0], pareto_efficient[:, 1], c='red', marker='^' ,label='Pareto Front')
        ax[0].set_xlabel('1 - Fidelity')
        ax[0].set_ylabel('Cost')
        ax[0].set_title('(1-F)-C Plane Projection')
        ax[0].legend()
        ax[1].scatter(pareto_efficient[:, 0], pareto_efficient[:, 2], c='red', marker='^' ,label='Pareto Front')
        ax[1].set_xlabel('1 - Fidelity')
        ax[1].set_ylabel('1/Throughput')
        ax[1].set_title('(1-F)-(1/TP) Plane Projection')
        ax[1].legend()
        ax[2].scatter(pareto_efficient[:, 1], pareto_efficient[:, 2], c='red', marker='^' ,label='Pareto Front')
        ax[2].set_xlabel('Cost')
        ax[2].set_ylabel('1/Throughput')
        ax[2].set_title('C-(1/TP) Plane Projection')
        ax[2].legend()
        st.pyplot(fig)

st.set_page_config(layout="wide")
st.title('Qnet-genetic-opt Streamlit App')
st.write("This is app for visualizing the results of Qnet-genetic-opt")

tab1, tab2, tab3, tab4 = st.tabs(["Single Experiment", "Dataframe", "Overall graph", "Pareto Front"])

def clear_multi():
    st.session_state.multiselect = []
    return

# path = 'Results/'
# path = 'Results/Results_17_10_2023 (only default node)/'
# path = 'Results/Results_20_10_2023 (custom node and increase loss)/'
# path = 'Results/Results_30_10_2023_throughput/'
path = 'Results/Results_7_12_2023_3DCostFunction/'

file_names = os.listdir(path)
file_names = [file for file in file_names if os.path.isfile(os.path.join(path, file))]

with tab1:

    files = st.multiselect('Select your experiment', file_names)    

    pickled_data = []
    for file in files:
        with open(f'{path+file}', 'rb') as f:
            pickled_data.append(pickle.load(f))
    # print(f'pickle: {pickled_data}')

    for data in pickled_data:

        col0, col1, col2 = st.columns(3)
        
        config_show = st.checkbox('Show experiment configuration', key='checkbox', value=True)
        

        if config_show:        
            col0.subheader('Experiment configuration')
            col0.write(f'{data.exper_config.ExperimentName}')

            col1.subheader('Quantum configuration')
            col1.write(f'{data.exper_config.QuantumNetworkSimulationConfig.NumHops} hops')
            col1.write(f'{data.exper_config.QuantumNetworkSimulationConfig.NetworkGeneration} network generation')

            col2.subheader('Genetic algorithm configuration')
            col2.write(f'Population size: {data.exper_config.GeneticAlgorithmConfig.PopulationSize}')
            col2.write(f'DNA start position: {data.exper_config.GeneticAlgorithmConfig.DnaStartPosition}')
            col2.write(f'Elitism: {data.exper_config.GeneticAlgorithmConfig.Elitism}')
            col2.write(f'Mutation rate: {data.exper_config.GeneticAlgorithmConfig.MutationRate}')
            col2.write(f'Mutation sigma: {data.exper_config.GeneticAlgorithmConfig.MutationSigma}')    
            col2.write(f'Weight_F: {data.exper_config.GeneticAlgorithmConfig.Weight_F}')
            col2.write(f'Weight_TP: {data.exper_config.GeneticAlgorithmConfig.Weight_TP}')
            col2.write(f'Weight_C: {data.exper_config.GeneticAlgorithmConfig.Weight_C}')
            col2.text("")

        # st.session_state
        # st.button('Clear', on_click=clear_multi)    

        col1, col2 = st.columns(2)

        col1.subheader('Fidelity & Cost via generations')
        
        x = np.arange(0, len(data.exper_config.FidelityHistory.Max), 1)
        weight_f = data.exper_config.GeneticAlgorithmConfig.Weight_F
        weight_tp = data.exper_config.GeneticAlgorithmConfig.Weight_TP
        weight_c = data.exper_config.GeneticAlgorithmConfig.Weight_C
        objective_fidelity = data.exper_config.GeneticAlgorithmConfig.ObjectiveFidelity
        mutation_rate = data.exper_config.GeneticAlgorithmConfig.MutationRate
        num_population = data.exper_config.GeneticAlgorithmConfig.PopulationSize

        fig, ax = plt.subplots(1, figsize=(15,10))
        ax_2 = ax.twinx()

        ax.set_title(f'w_f: {weight_f}, w_tp: {weight_tp}, w_c: {weight_c},mr: {mutation_rate}, pop: {num_population}', loc='right')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fidelity')
        ax_2.set_ylabel('Cost')

        ax.plot(x, data.exper_config.FidelityHistory.Mean, label='Fidelity', color='tab:blue')
        ax_2.plot(x, data.exper_config.CostHistory.Mean, label='Cost', color='tab:red')

        ax.legend(loc='upper left')
        ax_2.legend(loc='upper right')

        col1.pyplot(fig)

        # - - - - - - - - - - - - - - - - -

        col2.subheader('Fidelity via Cost')
        col2.scatter_chart(pd.DataFrame(
            {'Fidelity': data.exper_config.FidelityHistory.Mean, 
            'Cost': data.exper_config.CostHistory.Mean,                             
            }),
            x='Fidelity',
            y='Cost')        
        
        # - - - - - - - - - - - - - - - - -

        col1.subheader('Evolutional of Fidelity')
        data_fidelity = pd.DataFrame(
            {'Max': data.exper_config.FidelityHistory.Max, 
            'Min': data.exper_config.FidelityHistory.Min, 
            'Mean': data.exper_config.FidelityHistory.Mean,                             
            }
        )
        col1.line_chart(data=data_fidelity)
        
        # - - - - - - - - - - - - - - - - -

        col2.subheader('Evolutional of Cost')
        data_cost = pd.DataFrame(
            {            
                'Max': data.exper_config.CostHistory.Max,   
            'Min': data.exper_config.CostHistory.Min,       
            'Mean': data.exper_config.CostHistory.Mean,                         
            }
        )
        col2.line_chart(data=data_cost)

        # - - - - - - - - - - - - - - - - -

        # st.subheader('Evolutional of Cost')
        # data = pd.DataFrame(
        #     {'Max': data.exper_config.CostHistory.Max, 
        #      'Min': data.exper_config.CostHistory.Min, 
        #      'Mean': data.exper_config.CostHistory.Mean,                             
        #     }
        # )
        # st.line_chart(data=data)

        st.subheader('Pareto Front')

        inverse_fidelity = []
        for data_point in data.exper_config.FidelityHistory.All[-1]:
            inv_F = 1 - data_point
            inverse_fidelity.append(inv_F)

        cost = data.exper_config.CostHistory.All[-1]

        inverse_throughput = []
        for data_point in data.exper_config.ThroughtputHistory.All[-1]:
            inv_TP = 1/data_point
            inverse_throughput.append(inv_TP)

        pareto_frontier_3d_v2(inverse_fidelity, cost, inverse_throughput, plot_frontier=True, pareto_function='min')

with tab2:
    st.write('This is the dataframe of your experiment')
    # for file in file_names:
    #     st.write(file)

    df = pd.DataFrame(
        data={
            'Experiment name': [],
            'Date created': [],
            'Elitism': [],
            'Mutation rate': [],
            'Mutation sigma': [],
            'Weight': [],
            'Final max fidelity': [],
            'Final max cost': [],
            'Final mean fidelity': [],
            'Final mean cost': [],
        }
    )

    date_filter = st.text_input('Filter date', key='date_filter')
    
    for file in file_names:

        if date_filter != '' and date_filter in file:
            with open(f'{path+file}', 'rb') as f:
                data = pickle.load(f)        

                data_to_append = {
                    'Experiment name': file,
                    'Date created': data.exper_config.DateCreated,   
                    'Elitism': data.exper_config.GeneticAlgorithmConfig.Elitism,
                    'Mutation rate': data.exper_config.GeneticAlgorithmConfig.MutationRate,
                    'Mutation sigma': data.exper_config.GeneticAlgorithmConfig.MutationSigma,
                    'Weight_F': data.exper_config.GeneticAlgorithmConfig.Weight_F,
                    'Weight_TP': data.exper_config.GeneticAlgorithmConfig.Weight_TP,
                    'Weight_C': data.exper_config.GeneticAlgorithmConfig.Weight_C,
                    'Final max fidelity': data.exper_config.FidelityHistory.Max[-1],
                    'Final cost': data.exper_config.CostHistory.Max[-1],
                    'Final mean fidelity': data.exper_config.FidelityHistory.Mean[-1],
                    'Final mean cost': data.exper_config.CostHistory.Mean[-1],
                }

                df_to_append = pd.DataFrame([data_to_append])
                df = pd.concat([df, df_to_append], ignore_index=True)

        if date_filter == '':
            with open(f'{path+file}', 'rb') as f:
                data = pickle.load(f)

                data_to_append = {
                    'Experiment name': file,
                    'Date created': data.exper_config.DateCreated,
                    'Elitism': data.exper_config.GeneticAlgorithmConfig.Elitism,
                    'Mutation rate': data.exper_config.GeneticAlgorithmConfig.MutationRate,
                    'Mutation sigma': data.exper_config.GeneticAlgorithmConfig.MutationSigma,
                    'Weight_F': data.exper_config.GeneticAlgorithmConfig.Weight_F,
                    'Weight_TP': data.exper_config.GeneticAlgorithmConfig.Weight_TP,
                    'Weight_C': data.exper_config.GeneticAlgorithmConfig.Weight_C,
                    'Final max fidelity': data.exper_config.FidelityHistory.Max[-1],
                    'Final cost': data.exper_config.CostHistory.Max[-1],
                    'Final mean fidelity': data.exper_config.FidelityHistory.Mean[-1],
                    'Final mean cost': data.exper_config.CostHistory.Mean[-1],
                }

                df_to_append = pd.DataFrame([data_to_append])
                df = pd.concat([df, df_to_append], ignore_index=True)


    st.dataframe(df)

with tab3:
    # path = 'Results/'
    # file_names = os.listdir(path)
    files = st.multiselect('Select your experiment', file_names, key='multiselect', default=file_names)
    pickled_data = []
    for file in files:
        with open(f'{path+file}', 'rb') as f:
            pickled_data.append(pickle.load(f))    

    selected_graph = st.multiselect(
        'Select graph', ['Fidelity & Cost via generations', 'Evolutional of Fidelity'], 
        key='multiselect_graph', 
        default=['Fidelity & Cost via generations', 'Evolutional of Fidelity'])

    for data in pickled_data:
        # st.write(f'file: {data.exper_config.ExperimentName}')
        col1, col2 = st.columns(2)

        if 'Fidelity & Cost via generations' in selected_graph:
            col1.subheader('Fidelity & Cost via generations')
            col1.write(f'file: {data.exper_config.ExperimentName}')
            
            x = np.arange(0, len(data.exper_config.FidelityHistory.Max), 1)
            weight_f = data.exper_config.GeneticAlgorithmConfig.Weight_F
            weight_tp = data.exper_config.GeneticAlgorithmConfig.Weight_TP
            weight_c = data.exper_config.GeneticAlgorithmConfig.Weight_C
            objective_fidelity = data.exper_config.GeneticAlgorithmConfig.ObjectiveFidelity
            mutation_rate = data.exper_config.GeneticAlgorithmConfig.MutationRate
            num_population = data.exper_config.GeneticAlgorithmConfig.PopulationSize

            fig, ax = plt.subplots(1, figsize=(15,10))
            ax_2 = ax.twinx()

            ax.set_title(f'w_f: {weight_f}, w_tp: {weight_tp}, w_c: {weight_c},mr: {mutation_rate}, pop: {num_population}', loc='right')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fidelity')
            ax_2.set_ylabel('Cost')

            ax.plot(x, data.exper_config.FidelityHistory.Mean, label='Fidelity', color='tab:blue')
            ax_2.plot(x, data.exper_config.CostHistory.Mean, label='Cost', color='tab:red')

            ax.legend(loc='upper left')
            ax_2.legend(loc='upper right')

            col1.pyplot(fig)

        # if 'Fidelity via Cost' in selected_graph:
        #     col2.subheader('Fidelity via Cost')
        #     col2.scatter_chart(pd.DataFrame(
        #         {'Fidelity': data.exper_config.FidelityHistory.Mean, 
        #         'Cost': data.exper_config.CostHistory.Mean,                             
        #         }),
        #         x='Fidelity',
        #         y='Cost')
        
        if 'Evolutional of Fidelity' in selected_graph:
            col2.subheader('Evolutional of Fidelity')
            data_fidelity = pd.DataFrame(
                {'Max': data.exper_config.FidelityHistory.Max, 
                'Min': data.exper_config.FidelityHistory.Min, 
                'Mean': data.exper_config.FidelityHistory.Mean,                             
                }
            )
            col2.line_chart(data=data_fidelity)

        # if 'Evolutional of Cost' in selected_graph:
        #     col4.subheader('Evolutional of Cost')
        #     data_cost = pd.DataFrame(
        #         {            
        #             'Max': data.exper_config.CostHistory.Max,   
        #         'Min': data.exper_config.CostHistory.Min,       
        #         'Mean': data.exper_config.CostHistory.Mean,                         
        #         }
        #     )
        #     col4.line_chart(data=data_cost)

with tab4:
    
    files = st.multiselect('Select your experiment', file_names, key='multiselect_t4', default=file_names)

    pickled_data = []
    for file in files:
        with open(f'{path+file}', 'rb') as f:
            pickled_data.append(pickle.load(f))   

    for data in pickled_data:
        data.exper_config.ExperimentName
        
        inverse_fidelity = []
        for data_point in data.exper_config.FidelityHistory.All[-1]:
            inv_F = 1 - data_point
            inverse_fidelity.append(inv_F)

        cost = data.exper_config.CostHistory.All[-1]

        inverse_throughput = []
        for data_point in data.exper_config.ThroughtputHistory.All[-1]:
            inv_TP = 1/data_point
            inverse_throughput.append(inv_TP)

        pareto_frontier_3d_v2(inverse_fidelity, cost, inverse_throughput, plot_frontier=True, pareto_function='min')
        st.markdown("""---""")