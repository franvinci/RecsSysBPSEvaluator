from tqdm import tqdm
import numpy as np
import pandas as pd
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from PetriNetRecsBPS.src.controlflow_utils import return_enabled_and_fired_transitions
from sklearn.linear_model import LogisticRegression


def return_transitions_frequency(log, net, initial_marking, final_marking):

    alignments_ = alignments.apply_log(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]

    frequency_t = {t: 0 for t in net.transitions}
    for trace in aligned_traces:
        for align in trace:
            name_t = align[1]
            for t in list(net.transitions):
                if t.name == name_t:
                    frequency_t[t] += 1
                    break

    return frequency_t


def build_datasets(log, net, initial_marking, final_marking, history_weights, data_attributes, net_transition_labels):

    if history_weights:
        t_dicts_dataset = {t: {a: [] for a in data_attributes} | {t_l: [] for t_l in net_transition_labels} | {'class': []} for t in net.transitions}
    else:
        t_dicts_dataset = {t: {a: [] for a in data_attributes} | {'class': []} for t in net.transitions}

    alignments_ = alignments.apply_log(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]
    i = 0

    for trace in tqdm(log):
        if data_attributes:
            trace_attributes = {a: trace[0][a] for a in data_attributes}
        trace_aligned = aligned_traces[i]
        i += 1
        visited_transitions, is_fired = return_enabled_and_fired_transitions(net, initial_marking, final_marking, trace_aligned)
        for j in range(len(visited_transitions)):
            t = visited_transitions[j]
            t_fired = is_fired[j]
            for a in data_attributes:
                t_dicts_dataset[t][a].append(trace_attributes[a])
            if history_weights:
                transitions_fired = [label for label, value in zip(visited_transitions[:j], is_fired[:j]) if value == 1]
                for t_ in net_transition_labels:
                    if history_weights == 'binary':
                        t_dicts_dataset[t][t_].append((t_ in [x.label for x in transitions_fired])*1)
                    if history_weights == 'count':
                        t_dicts_dataset[t][t_].append([x.label for x in transitions_fired].count(t_))
            t_dicts_dataset[t]['class'].append(t_fired)

    return t_dicts_dataset


def return_scaler_params(net, t_dicts_dataset, data_attributes_categorical):
    
    scaler_params = dict()
    t = list(net.transitions)[0]
    for x in list(t_dicts_dataset[t].keys()):
        if x != 'class' and (x not in data_attributes_categorical):
            scaler_params[x] = (min(t_dicts_dataset[t][x]), max(t_dicts_dataset[t][x]))
    for t in net.transitions:
        for x in list(t_dicts_dataset[t].keys()):
            if x != 'class' and (x not in data_attributes_categorical):
                scaler_params[x] = (min(min(t_dicts_dataset[t][x]),scaler_params[x][0]), max(max(t_dicts_dataset[t][x]), scaler_params[x][1]))
    
    return scaler_params


def build_models(log, 
                 net, 
                 initial_marking, 
                 final_marking,
                 data_attributes, 
                 data_attributes_categorical, 
                 attr_values_categorical, 
                 net_transition_labels, 
                 history_weights):

    t_dicts_dataset = build_datasets(log, net, initial_marking, final_marking, history_weights, data_attributes, net_transition_labels)
    models_t = dict()
    coefficients_list = []
    coeff_index = []

    for t in tqdm(net.transitions):
        data_t = pd.DataFrame(t_dicts_dataset[t])
        if len(data_t['class'].unique())<2:
            models_t[t] = None
            continue
        # if scaler:
        #     for c in list(data_t.columns):
        #         if c != 'class' and (c not in data_attributes_categorical):
        #             if scaler_params[c][0] != scaler_params[c][1]:
        #                 data_t[c] = (data_t[c] - scaler_params[c][0]) / (scaler_params[c][1] - scaler_params[c][0])
        for a in data_attributes_categorical:
            for v in attr_values_categorical[a]:
                data_t[a+'_'+v] = (data_t[a] == v)*1
            del data_t[a]

        X = data_t.drop(columns=['class'])
        y = data_t['class']

        clf_t = LogisticRegression(random_state=72).fit(X, y)
        models_t[t] = clf_t

        coeff_index.append(t)
        coefficients_list.append([clf_t.intercept_[0]] + list(clf_t.coef_[0]))

    coefficients = pd.DataFrame(coefficients_list, columns=['intercept'] + list(clf_t.feature_names_in_), index=coeff_index)

    return models_t, coefficients


def compute_proba(models_t, t, X):

    clf_t = models_t[t]
    coeff = clf_t.coef_[0]
    intercept = clf_t.intercept_[0]

    return 1/(1+np.exp(- intercept - (coeff*np.array(X)).sum()))