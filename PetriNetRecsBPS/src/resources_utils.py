import pm4py
from PetriNetRecsBPS.src.temporal_utils import n_to_weekday


def find_roles(log):
    """
    {role: (listOfAct, listOfResorces)}
    """

    try:
        name_roles = list(pm4py.get_event_attribute_values(log, "org:role").keys())
        roles = {n: ([], []) for n in name_roles}
        for trace in log:
            for event in trace:
                res = event['org:resource']
                role = event['org:role']
                activity = event['concept:name']
                if activity not in roles[role][0]:
                    roles[role][0].append(activity)
                if res not in roles[role][1]:
                    roles[role][1].append(res)

        # roles = dict()
        # for n in roles_res.keys():
        #     roles[n] =  (roles_res[n][0], len(roles_res[n][1]))

    except:
        roles_pm4py = pm4py.discover_organizational_roles(log)
        roles = dict()
        for i in range(len(roles_pm4py)):
            roles['ROLE'+str(i)] =  (roles_pm4py[i].activities, list(roles_pm4py[i].originator_importance.keys()))
    return roles


def create_resources(roles):
    return {role: [role+'_'+str(i) for i in range(roles[role][1])] for role in roles.keys()}


def find_calendars(roles, mode='24/7', log=None):
    """
    {role: {WEEKDAY: (sH,eH)}}
    se quel weekday non si lavora mettere None
    """

    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if mode == '24/7':
        return {role: {wd: (0,23) for wd in weekday_labels} for role in roles.keys()}
    
    if mode == 'manual':
        calendars = dict()
        for role in roles:
            calendars[role] = dict()
            for wd in weekday_labels:
                try:
                    start_hour = int(input(role + ' ' + wd + ' ' + 'Start Hour: '))
                    end_hour = int(input(role + ' ' + wd + ' ' + 'Final Hour: '))
                    calendars[role][wd] = (start_hour, end_hour)
                except:
                    calendars[role][wd] = None
        return calendars
    
    if mode == 'discover':
        calendars = dict()
        if not log:
            print('Error: Insert Event Log for discover mode.')
            print('Mode set to manual')
            return find_calendars(roles, mode='manual')
        log_df = pm4py.convert_to_dataframe(log)
        log_df['weekday'] = log_df['time:timestamp'].apply(lambda x: n_to_weekday(x.weekday()))
        log_df['hour'] = log_df['time:timestamp'].apply(lambda x: x.hour)
        for role in roles.keys():
            calendars[role] = dict()
            role_acts = roles[role][0]
            log_df_act_roles = log_df[log_df['concept:name'].isin(role_acts)]
            for wd in weekday_labels:
                log_df_act_roles_wd = log_df_act_roles[log_df_act_roles['weekday'] == wd]
                if len(log_df_act_roles_wd) == 0:
                    calendars[role][wd] = None
                calendars[role][wd] = (log_df_act_roles_wd['hour'].min(), log_df_act_roles_wd['hour'].max())
                if not(calendars[role][wd][0] >= 0):
                    calendars[role][wd] = None
        return calendars
    

def find_frequency_act_res(log):

    activities = list(pm4py.get_event_attribute_values(log, "concept:name").keys())
    resources = list(pm4py.get_event_attribute_values(log, "org:resource").keys())
    act_res_frequency = {a: {r: 0 for r in resources} for a in activities}
    for trace in log:
        for event in trace:
            act = event['concept:name']
            res = event['org:resource']
            act_res_frequency[act][res] += 1
    
    return act_res_frequency