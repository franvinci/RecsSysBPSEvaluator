from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
import pandas as pd
import pickle
import evaluator


if __name__ == '__main__':

    case_studies =[
        'bac',
        'bpi17_before',
        'bpi17_after'
    ]

    for case_study in case_studies:
        for benchmark in [False, True]:
            if case_study == 'bac':
                path_log = 'data/bac/logs/log.xes'
                path_net = 'data/bac/models/model.pnml'
                path_resource_availability = 'data/bac/recommendations/resources_times.pkl'
                split_time = '2019-02-09 02:36:57'
                data_attributes = ['CLOSURE_TYPE', 'CLOSURE_REASON']
                categorical_attributes = ['CLOSURE_TYPE', 'CLOSURE_REASON']
                if benchmark:
                    path_recommendations = 'data/bac/recommendations/recommendations_dataframe.csv'
                    path_simulations = 'data/bac/simulations/bac'
                else:
                    path_recommendations = 'data/bac/recommendations/recommendations_dataframe_benchmark.csv'
                    path_simulations = 'data/bac/simulations/benchmark'

            if case_study == 'bpi17_before':
                path_log = 'data/bpi17/logs/log_before.xes'
                path_net = 'data/bpi17/models/model.pnml'
                path_resource_availability = 'data/bpi17/recommendations/resources_times_before.pkl'
                split_time = '2016-05-29 19:10:28'
                data_attributes = ['ApplicationType', 'LoanGoal', 'RequestedAmount']
                categorical_attributes = ['ApplicationType', 'LoanGoal']
                if benchmark:
                    path_recommendations = 'data/bpi17/recommendations/recommendations_dataframe_before_benchmark.csv'
                    path_simulations = 'data/bpi17/simulations/before_benchmark'
                else:
                    path_recommendations = 'data/bpi17/recommendations/recommendations_dataframe_before.csv'
                    path_simulations = 'data/bpi17/simulations/before'

            if case_study == 'bpi17_after':
                path_log = 'data/bpi17/logs/log_after.xes'
                path_net = 'data/bpi17/models/model.pnml'
                path_resource_availability = 'data/bpi17/recommendations/resources_times_after.pkl'
                split_time = '2016-12-02 07:31:53'
                data_attributes = ['ApplicationType', 'LoanGoal', 'RequestedAmount']
                categorical_attributes = ['ApplicationType', 'LoanGoal'] 
                if benchmark:
                    path_recommendations = 'data/bpi17/recommendations/recommendations_dataframe_after_benchmark.csv'
                    path_simulations = 'data/bpi17/simulations/after_benchmark'
                else:
                    path_recommendations = 'data/bpi17/recommendations/recommendations_dataframe_after.csv'
                    path_simulations = 'data/bpi17/simulations/after'


            log = xes_importer.apply(path_log)
            net, initial_marking, final_marking = pm4py.read_pnml(path_net)
            recommendations_df = pd.read_csv(path_recommendations)

            with open(path_resource_availability, 'rb') as file:
                res_availability = pickle.load(file)

            simulations_df, recommendations_df = evaluator.apply(log, net, initial_marking, final_marking, recommendations_df, res_availability, split_time, data_attributes, categorical_attributes, n_sim=10)

            for j in range(len(simulations_df)):
                sim_j = simulations_df[j]
                sim_j.to_csv(path_simulations + f'/sim_{j+1}.csv', index=False)
            
            recommendations_df.to_csv(path_simulations + '/recommendations.csv', index=False)