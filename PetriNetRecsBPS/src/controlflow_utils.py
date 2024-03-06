import random

def return_enabled_transitions(net, tkns):
    enabled_t = set()
    for t in list(net.transitions):
        if {a.source for a in t.in_arcs}.issubset(tkns):
            enabled_t.add(t)
    return enabled_t


def return_fired_transition(transition_weights, enabled_transitions):

    total_weight = sum(transition_weights[s] for s in enabled_transitions)
    random_value = random.uniform(0, total_weight)
    
    cumulative_weight = 0
    for s in enabled_transitions:
        cumulative_weight += transition_weights[s]
        if random_value <= cumulative_weight:
            return s


def update_markings(tkns, t_fired):

    for a_in in list(t_fired.in_arcs):
        tkns.remove(a_in.source)

    for a_out in list(t_fired.out_arcs):
        tkns.extend([a_out.target])
        
    return tkns


def return_enabled_and_fired_transitions(net, initial_marking, final_marking, trace_aligned):

    visited_transitions = []
    is_fired = []
    tkns = list(initial_marking)
    enabled_transitions = return_enabled_transitions(net, tkns)
    for t_fired_name in trace_aligned:
        for t in net.transitions:
            if t.name == t_fired_name[1]:
                t_fired = t
                break
        not_fired_transitions = list(enabled_transitions-{t_fired})
        for t_not_fired in not_fired_transitions:
            visited_transitions.append(t_not_fired)
            is_fired.append(0)
        visited_transitions.append(t_fired)
        is_fired.append(1)
        tkns = update_markings(tkns, t_fired)
        if set(tkns) == set(final_marking):
            return visited_transitions, is_fired
        enabled_transitions = return_enabled_transitions(net, tkns)

    return visited_transitions, is_fired