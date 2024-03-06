import numpy as np
import pandas as pd
from datetime import datetime
from src.prefix_utils import return_prefixes_from_recommendations_df
from src.state_utils import return_attributes_from_recommendations_df, return_act_res_recommendations
from PetriNetRecsBPS.src.PetriNetBPS import SimulatorParameters, SimulatorEngine
from tqdm import tqdm


def apply(log, net, initial_marking, final_marking, recommendations_df, res_availability, split_time, data_attributes, categorical_attributes, n_sim=10):

    parameters = SimulatorParameters(net, initial_marking, final_marking)
    parameters.discover_from_eventlog(log, mode_ex_time='resource', mode_trans_weights='data_attributes', data_attributes=data_attributes, categorical_attributes=categorical_attributes, history_weights='binary')

    simulator = SimulatorEngine(net, initial_marking, final_marking, parameters)

    prefixes = return_prefixes_from_recommendations_df(recommendations_df, log)
    if data_attributes:
        attributes = return_attributes_from_recommendations_df(recommendations_df, log, data_attributes)
    else:
        attributes = [[] for _ in range(len(recommendations_df))]


    simulations = []
    top_recomm = []
    for j in range(n_sim):
        print(f'SIM. {j+1}')
        sim_j = []
        for i in tqdm(range(len(recommendations_df))):
            act_rec, res_rec = return_act_res_recommendations(recommendations_df, res_availability, i, top_k = 3)
            if act_rec:
                recommendations = {
                    "starting_time": split_time,
                    "activities": [act_rec],
                    "resources": [res_rec],
                    "prefixes": [prefixes[i]],
                    "attributes": [attributes[i]]
                }
                log_data = simulator.simulate(1, 
                                            remove_head_tail=0, 
                                            starting_time=recommendations['starting_time'], 
                                            resource_availability=res_availability,
                                            recommendations=recommendations)
                if j == 0:
                    top_recomm.append(simulator.top_k_rec[0])
            
            sim_j.append(log_data)
        
        simulations.append(sim_j)


    if 0 in top_recomm:
        print(str(round(top_recomm.count(0)/len(recommendations_df)*100, 2)) + "% not possible recommendations.")


    # save simulations
    for j in range(n_sim):
        simulations_df = []
        sims = []
        for i in range(len(recommendations_df)):
            if top_recomm[i] == 0:
                continue
            case_id = recommendations_df.iloc[i]['case:concept:name']
            sim = simulations[j][i]
            sim['case:concept:name'] = case_id
            sims.append(sim)
        sim_j = pd.concat(sims)
        sim_j.sort_values(by='time:timestamp', inplace=True)
        sim_j.index = range(len(sim_j))
        simulations_df.append(sim_j)


    act_rec = []
    res_rec = []

    for i in range(len(recommendations_df)):
        k = top_recomm[i]
        if k == 0:
            act_rec.append(None)
            res_rec.append(None)
        else:
            act_rec.append(recommendations_df.iloc[i][f'act_{k}'])
            res_rec.append(recommendations_df.iloc[i][f'res_{k}'])

    for c in recommendations_df.columns:
        if 'act' in c or 'res' in c:
            del recommendations_df[c]

    recommendations_df['act_rec'] = act_rec
    recommendations_df['res_rec'] = res_rec

    recommendations_df['top_k'] = top_recomm
    recommendations_df['top_k'][recommendations_df['top_k'] == 0] = None


    cycle_real = []
    for i in range(len(recommendations_df)): 
        case_id = recommendations_df.iloc[i]['case:concept:name']
        for trace in log:
            if trace.attributes['concept:name'] == str(case_id):
                break
        if '.' in str(trace[-1]['time:timestamp']):
            cycle_real.append((datetime.strptime(str(trace[-1]['time:timestamp']).split('.')[0], "%Y-%m-%d %H:%M:%S") - datetime.strptime(split_time, "%Y-%m-%d %H:%M:%S")).total_seconds()/60)
        else:
            cycle_real.append((datetime.strptime(str(trace[-1]['time:timestamp'])[:-6], "%Y-%m-%d %H:%M:%S") - datetime.strptime(split_time, "%Y-%m-%d %H:%M:%S")).total_seconds()/60)

    recommendations_df['cycle_real'] = cycle_real
        


    cycle_sims = []
    for j in range(n_sim):
        cycle_sims.append([])
        for i in range(len(recommendations_df)):
            if top_recomm[i] > 0:
                if len(simulations[j][i]):
                    cycle_sims[j].append((list(simulations[j][i]['time:timestamp'])[-1] - datetime.strptime(split_time, "%Y-%m-%d %H:%M:%S")).total_seconds()/60)
                else:
                    cycle_sims[j].append(0)
            else:
                cycle_sims[j].append(cycle_real[i])
        recommendations_df[f'cycle_sim_{j+1}'] = cycle_sims[-1]


    print()
    print('RESULTS AVG per trace')
    results = []
    for j in range(n_sim):
        res = np.mean(np.array(cycle_sims[j]) - np.array(cycle_real))
        results.append(res)

    print('AVG ', np.mean(results))
    print('STD. ', np.std(results))


    print()
    print('RESULTS MEDIAN REL. per trace')
    results_rel = []
    for j in range(n_sim):
        res = list((np.array(cycle_sims[j]) - np.array(cycle_real))/np.array(cycle_real))
        results_rel.append(np.median(res))

    print('MEDIAN REL. ', np.mean(results_rel))
    print('STD. REL. ', np.std(results_rel))


    print()
    print('RESULTS SUM')
    results = []
    for j in range(n_sim):
        res = np.sum(np.array(cycle_sims[j]) - np.array(cycle_real))
        results.append(res)

    print('AVG SUM ', np.mean(results))
    print('STD SUM ', np.std(results))



    print()
    print('RESULTS SUM REL')
    results = []
    for j in range(n_sim):
        res = np.sum(np.array(cycle_sims[j]) - np.array(cycle_real))
        results.append(res/np.sum(cycle_real))

    print('AVG SUM REL.', np.mean(results))
    print('STD SUM REL.', np.std(results))

    return simulations_df, recommendations_df