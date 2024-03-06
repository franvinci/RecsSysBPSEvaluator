import pm4py
from datetime import timedelta
from PetriNetRecsBPS.src.distribution_utils import find_best_fit_distribution
from tqdm import tqdm

possible_distributions = [
    'fixed',
    'normal',
    'exponential',
    'uniform',
    'triangular',
    'lognormal',
    'gamma'
    ]

def n_to_weekday(i):
    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return dict(zip(range(7), weekday_labels))[i]


def compute_execution_times(log, filter_by_res=None):

    activities = list(pm4py.get_event_attribute_values(log, 'concept:name').keys())
    activities_extimes = {a: [] for a in activities}
    for trace in log:
        for event in trace:
            act = event['concept:name']
            time_0 = event['start:timestamp']
            time_1 = event['time:timestamp']
            if filter_by_res:
                res = event['org:resource']
                if res in filter_by_res:
                    activities_extimes[act].append((time_1 - time_0).total_seconds())
            else:
                activities_extimes[act].append((time_1 - time_0).total_seconds())
    
    for a in activities:
        if not activities_extimes[a]:
            del activities_extimes[a]

    return activities_extimes


def find_execution_distributions(log, mode='activity'):
    """
    output: {ACTIVITY_NAME: (DISTRNAME, {PARAMS: VALUE})}
    """
    if mode == 'activity':
        activities_extimes = compute_execution_times(log)
        activities = list(activities_extimes.keys())
        exec_distr = {a: find_best_fit_distribution(activities_extimes[a])[:2] for a in activities}
    if mode == 'resource':
        resources = pm4py.get_event_attribute_values(log, "org:resource")
        exec_distr = dict()
        print('Finding best fit execution time distribution for each resources...')
        for res in tqdm(resources):
            activities_extimes = compute_execution_times(log, filter_by_res=[res])
            activities = list(activities_extimes.keys())
            exec_distr[res] = {a: find_best_fit_distribution(activities_extimes[a])[:2] for a in activities}
            
    return exec_distr


def find_arrival_calendar(log=None, mode='24/7'):
    """
    {WEEKDAY: (sH,eH)}
    """

    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if mode == '24/7':
        return {wd: (0,23) for wd in weekday_labels}

    if mode == 'manual':
        calendar = dict()
        for wd in weekday_labels:
            try:
                start_hour = int(input(wd + ' ' + 'Start Hour: '))
                end_hour = int(input(wd + ' ' + 'Final Hour: '))
                calendar[wd] = (start_hour, end_hour)
            except:
                calendar[wd] = None
        return calendar

    calendar_distr = {wd: [] for wd in weekday_labels}
    if mode == 'discover':
        for trace in log:
            arrival_wd = n_to_weekday(trace[0]['start:timestamp'].weekday())
            arrival_h = trace[0]['start:timestamp'].hour
            calendar_distr[arrival_wd].append(arrival_h)
        for wd in weekday_labels:
            if len(calendar_distr[wd]) > 0:
                calendar[wd] = (min(calendar_distr[wd]), max(calendar_distr[wd]))
            else:
                calendar[wd] = None
            if not(calendar[wd][0] >= 0):
                calendar[wd] = None
        return calendar


def compute_arrival_times(log, arrival_calendar):

    arrival_times = []
    for i in range(1, len(log)):
        time_1 = log[i][0]['start:timestamp']
        time_0 = log[i-1][0]['start:timestamp']
        d_1 = log[i][0]['start:timestamp'].date()
        d_0 = log[i-1][0]['start:timestamp'].date()
        if d_1 != d_0:
            continue
        wd = n_to_weekday(log[i][0]['start:timestamp'].weekday())
        h_1 = log[i][0]['start:timestamp'].hour
        h_0 = log[i-1][0]['start:timestamp'].hour
        if not (arrival_calendar[wd][0] <= h_0 <= arrival_calendar[wd][1]) and not ((arrival_calendar[wd][0] <= h_1 <= arrival_calendar[wd][1])):
            continue
        arrival_times.append((time_1-time_0).total_seconds())
    
    return arrival_times
    

def find_arrival_distribution(log, arrival_calendar):
    return find_best_fit_distribution(compute_arrival_times(log, arrival_calendar))[:2]


def return_time_from_calendar(current_time, calendar):
    n_wd = current_time.weekday()
    wd = n_to_weekday(n_wd)
    h = current_time.hour
    if calendar[wd]:
        if h < calendar[wd][0]:
            current_time = current_time.replace(hour=calendar[wd][0], minute=0, second=0)
            return current_time
        elif h > calendar[wd][1]:
            current_time = current_time + timedelta(days=1)
            n_wd = current_time.weekday()
            wd = n_to_weekday(n_wd)
        else:
            return current_time
    j_day = 0
    while not calendar[wd]:
        j_day += 1
        n_wd = (n_wd + 1) % 7
        wd = n_to_weekday(n_wd)
    if j_day > 0:
        current_time = current_time + timedelta(days=j_day)
    current_time = current_time.replace(hour=calendar[wd][0], minute=0, second=0)    
    return current_time