from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from tqdm import tqdm
import pm4py

def return_prefixes_from_recommendations_df(recommendations_df, log):

    prefixes = []
    for i in tqdm(range(len(recommendations_df))):
        case_id = recommendations_df.iloc[i]['case:concept:name']
        repl_id = recommendations_df.iloc[i]['repl_id']
        for trace in log:
            if trace.attributes['concept:name'] == str(case_id):
                break
        prefix = []
        for event in trace[:repl_id+1]:
            prefix.append(event['concept:name'])
        prefixes.append(prefix)

    return prefixes


def align_prefix(prefix, net, initial_marking, final_marking):

    model_cost_function = dict()
    sync_cost_function = dict()
    for t in net.transitions:
        if t.label is not None:
            if t.label in prefix:
                model_cost_function[t] = 0
            else:
                model_cost_function[t] = 1
            sync_cost_function[t] = 1
        else:
            model_cost_function[t] = 1

    parameters = {}
    parameters[alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
    parameters[alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function

    purpose_log = EventLog()
    trace = Trace()
    for act in prefix:
        trace.append(Event({"concept:name": act}))
    purpose_log.append(trace)

    alignments_ = alignments.apply_trace(trace, net, initial_marking, final_marking, parameters=parameters)
    prefix_aligned_to_trace = [t[1] for t in alignments_['alignment'] if t[1] and t[1]!='>>']
    prefix_aligned = []
    i = 1
    flag_stop_p = False
    for p in prefix_aligned_to_trace:
        prefix_aligned.append(p)
        if p == prefix[-i]:
            flag_stop_p = True
            break
    while not flag_stop_p and i <= len(prefix):
        i += 1
        for p in prefix_aligned_to_trace:
            prefix_aligned.append(p)
            if p == prefix[-i]:
                flag_stop_p = True
                break

    return prefix_aligned


def return_prefix_marking(prefix, net, initial_marking, final_marking):

    prefix_aligned = align_prefix(prefix, net, initial_marking, final_marking)
    purpose_log = EventLog()
    trace = Trace()
    for act in prefix_aligned:
        trace.append(Event({"concept:name": act}))
    purpose_log.append(trace)

    from pm4py.algo.conformance.tokenreplay.variants import token_replay
    parameters_tr = {
        token_replay.Parameters.CONSIDER_REMAINING_IN_FITNESS: True,
        token_replay.Parameters.TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN: False,
        token_replay.Parameters.STOP_IMMEDIATELY_UNFIT: False,
        token_replay.Parameters.WALK_THROUGH_HIDDEN_TRANS: True,
        token_replay.Parameters.ACTIVITY_KEY: 'concept:name'
    }
    res = token_replay.apply(purpose_log, net, initial_marking, final_marking, parameters=parameters_tr)[0]['reached_marking']
    return res

def view_prefix_marking(prefix, net, initial_marking, final_marking):
    res = return_prefix_marking(prefix, net, initial_marking, final_marking)
    pm4py.view_petri_net(net, res)