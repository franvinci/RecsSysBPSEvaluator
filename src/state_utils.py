from tqdm import tqdm

def return_attributes_from_recommendations_df(recommendations_df, log, data_attributes):

    attributes = []
    for i in tqdm(range(len(recommendations_df))):
        case_id = recommendations_df.iloc[i]['case:concept:name']
        for trace in log:
            if trace.attributes['concept:name'] == str(case_id):
                break
        attr = []
        for a in data_attributes:
            attr.append(trace[0][a])
        attributes.append(attr)

    return attributes


def return_act_res_recommendations(recommendations_df, res_availability, i, top_k = None):

    acts = list(recommendations_df.iloc[i][[c for c in recommendations_df.columns if c[:3]=='act']])

    resources = list(recommendations_df.iloc[i][[c for c in recommendations_df.columns if c[:3]=='res']])
    act_res_rec = dict()

    for a in acts:
        if a not in act_res_rec.keys() and a != 'missing':
            res_a = [value for idx, value in enumerate(resources) if acts[idx] == a and value != 'missing']
            act_res_rec[a] = sorted(res_a, key= lambda x: res_availability[x])

    if top_k:
        act_res_rec = dict(list(act_res_rec.items())[:top_k])

    acts_tot = [key for key, values in act_res_rec.items() for _ in range(len(values))]
    res_tot = [value for values in act_res_rec.values() for value in values]

    return acts_tot, res_tot