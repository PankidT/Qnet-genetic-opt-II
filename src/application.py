import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('Qnet-genetic-opt Streamlit App')
st.write("This is app for visualizing the results of Qnet-genetic-opt")

tab1, tab2 = st.tabs(["Single Experiment", "Dataframe"])

def clear_multi():
    st.session_state.multiselect = []
    return

path = 'Results/'
file_names = os.listdir(path)
file_names = [file for file in file_names if os.path.isfile(os.path.join(path, file))]

with tab1:

    # graph_type = [
    #     'Fidelity & Cost via generations',
    #     'Fidelity via Cost',
    #     'Evolutional of Fidelity ',
    #     'Evolutional of Cost'
    # ]    

    files = st.multiselect('Select your experiment', file_names)
    # print(f'file: {files}')

    pickled_data = []
    for file in files:
        with open(f'Results/{file}', 'rb') as f:
            pickled_data.append(pickle.load(f))
    # print(f'pickle: {pickled_data}')

    for data in pickled_data:

        col0, col1, col2 = st.columns(3)
        
        config_show = st.checkbox('Show experiment configuration', key='checkbox', value=True)
        

        if config_show:        
            col0.header('Experiment configuration')
            col1.subheader('Quantum configuration')
            col1.write(f'{data.exper_config.QuantumNetworkSimulationConfig.NumHops} hops')
            col1.write(f'{data.exper_config.QuantumNetworkSimulationConfig.NetworkGeneration} network generation')

            col2.subheader('Genetic algorithm configuration')
            col2.write(f'Population size: {data.exper_config.GeneticAlgorithmConfig.PopulationSize}')
            col2.write(f'DNA start position: {data.exper_config.GeneticAlgorithmConfig.DnaStartPosition}')
            col2.write(f'Elitism: {data.exper_config.GeneticAlgorithmConfig.Elitism}')
            col2.write(f'Mutation rate: {data.exper_config.GeneticAlgorithmConfig.MutationRate}')
            col2.write(f'Mutation sigma: {data.exper_config.GeneticAlgorithmConfig.MutationSigma}')    
            col2.write(f'Weight: {data.exper_config.GeneticAlgorithmConfig.Weight}')
            col2.text("")

        select = st.multiselect('Set your experiment visualization', ['Raw data', 'Graph'], key='multiselect', default=['Graph'])
        # st.session_state
        st.button('Clear', on_click=clear_multi)    

        if 'Raw data' in select:
            st.write('This is the raw data of your experiment')
            # st.write(df)

        if 'Graph' in select:

            col1, col2 = st.columns(2)

            col1.subheader('Fidelity & Cost via generations')
            
            x = np.arange(0, len(data.exper_config.FidelityHistory.Max), 1)
            weight = data.exper_config.GeneticAlgorithmConfig.Weight
            objective_fidelity = data.exper_config.GeneticAlgorithmConfig.ObjectiveFidelity
            mutation_rate = data.exper_config.GeneticAlgorithmConfig.MutationRate
            num_population = data.exper_config.GeneticAlgorithmConfig.PopulationSize

            fig, ax = plt.subplots(1, figsize=(15,10))
            ax_2 = ax.twinx()

            ax.set_title(f'w: {weight}, mr: {mutation_rate}, pop: {num_population}', loc='right')
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

            col2.subheader('Evolutional of Fidelity')
            data = pd.DataFrame(
                {'Max': data.exper_config.FidelityHistory.Max, 
                'Min': data.exper_config.FidelityHistory.Min, 
                'Mean': data.exper_config.FidelityHistory.Mean,                             
                }
            )
            col2.line_chart(data=data)
            
            # - - - - - - - - - - - - - - - - -

            # st.subheader('Evolutional of Cost')
            # data = pd.DataFrame(
            #     {'Max': data.exper_config.CostHistory.Max, 
            #      'Min': data.exper_config.CostHistory.Min, 
            #      'Mean': data.exper_config.CostHistory.Mean,                             
            #     }
            # )
            # st.line_chart(data=data)

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
            with open(f'Results/{file}', 'rb') as f:
                data = pickle.load(f)        

                data_to_append = {
                    'Experiment name': file,
                    'Date created': data.exper_config.DateCreated,   
                    'Elitism': data.exper_config.GeneticAlgorithmConfig.Elitism,
                    'Mutation rate': data.exper_config.GeneticAlgorithmConfig.MutationRate,
                    'Mutation sigma': data.exper_config.GeneticAlgorithmConfig.MutationSigma,
                    'Weight': data.exper_config.GeneticAlgorithmConfig.Weight,
                    'Final max fidelity': data.exper_config.FidelityHistory.Max[-1],
                    'Final cost': data.exper_config.CostHistory.Max[-1],
                    'Final mean fidelity': data.exper_config.FidelityHistory.Mean[-1],
                    'Final mean cost': data.exper_config.CostHistory.Mean[-1],
                }

                df_to_append = pd.DataFrame([data_to_append])
                df = pd.concat([df, df_to_append], ignore_index=True)

        if date_filter == '':
            with open(f'Results/{file}', 'rb') as f:
                data = pickle.load(f)

                data_to_append = {
                    'Experiment name': file,
                    'Date created': data.exper_config.DateCreated,   
                    'Elitism': data.exper_config.GeneticAlgorithmConfig.Elitism,
                    'Mutation rate': data.exper_config.GeneticAlgorithmConfig.MutationRate,
                    'Mutation sigma': data.exper_config.GeneticAlgorithmConfig.MutationSigma,
                    'Weight': data.exper_config.GeneticAlgorithmConfig.Weight,
                    'Final max fidelity': data.exper_config.FidelityHistory.Max[-1],
                    'Final cost': data.exper_config.CostHistory.Max[-1],
                    'Final mean fidelity': data.exper_config.FidelityHistory.Mean[-1],
                    'Final mean cost': data.exper_config.CostHistory.Mean[-1],
                }

                df_to_append = pd.DataFrame([data_to_append])
                df = pd.concat([df, df_to_append], ignore_index=True)


    st.dataframe(df)


    