from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay.variants import token_replay

def return_transitions_to_rec(rec, net, initial_marking, final_marking):
    
    new_prefix = [rec]
    purpose_log = EventLog()
    trace = Trace()
    for act in new_prefix:
        trace.append(Event({"concept:name": act}))
    purpose_log.append(trace)

    parameters_tr = {
        token_replay.Parameters.CONSIDER_REMAINING_IN_FITNESS: True,
        token_replay.Parameters.TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN: False,
        token_replay.Parameters.STOP_IMMEDIATELY_UNFIT: True,
        token_replay.Parameters.WALK_THROUGH_HIDDEN_TRANS: True,
        token_replay.Parameters.ACTIVITY_KEY: 'concept:name'
    }
    res = token_replay.apply(purpose_log, net, initial_marking, final_marking, parameters=parameters_tr)

    return res[0]['activated_transitions']