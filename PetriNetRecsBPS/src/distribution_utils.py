import numpy as np
from scipy import stats

possible_distributions = [
    'fixed',
    'normal',
    'exponential',
    'uniform',
    'triangular',
    'lognormal',
    'gamma'
    ]


def find_best_fit_distribution(observed_values, N=None, remove_outliars=False):

    if remove_outliars:
        q1 = np.percentile(observed_values, 25)
        q3 = np.percentile(observed_values, 75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        observed_values = observed_values[(observed_values < lower_limit) | (observed_values > upper_limit)]
    
    if not N:
        N = len(observed_values)

    generated_values = dict()
    distr_params = {d: dict() for d in possible_distributions}

    if np.min(observed_values) == np.max(observed_values):
        return 'fixed', {'value': 0}

    for distr_name in possible_distributions:
        if distr_name == 'fixed':
            distr_params[distr_name] = {'value' : np.mean(observed_values)}
            generated_values[distr_name] = np.array([distr_params[distr_name]['value']] * N)
        elif distr_name == 'normal':
            dist = stats.norm
            loc, scale = dist.fit(observed_values)
            q75, q25 = np.percentile(observed_values, [75 ,25])
            iqr = q75 - q25
            distr_params[distr_name] = {'loc': loc, 'scale': scale, 'max': q75+iqr*1.5, 'mean': np.mean(observed_values)}
            generated_values[distr_name] = dist.rvs(loc=loc, scale=scale, size=N)
        elif distr_name == 'exponential':
            dist = stats.expon
            loc, scale = dist.fit(observed_values)
            q75, q25 = np.percentile(observed_values, [75 ,25])
            iqr = q75 - q25
            distr_params[distr_name] = {'loc': loc, 'scale': scale, 'max': q75+iqr*1.5, 'mean': np.mean(observed_values)}
            generated_values[distr_name] = dist.rvs(loc=loc, scale=scale, size=N)
        elif distr_name == 'uniform':
            dist = stats.uniform
            loc, scale = dist.fit(observed_values)
            q75, q25 = np.percentile(observed_values, [75 ,25])
            iqr = q75 - q25
            distr_params[distr_name] = {'loc': loc, 'scale': scale, 'max': q75+iqr*1.5, 'mean': np.mean(observed_values)}
            generated_values[distr_name] = dist.rvs(loc=loc, scale=scale, size=N)
        elif distr_name == 'triangular':
            dist = stats.triang
            c, loc, scale = dist.fit(observed_values)
            q75, q25 = np.percentile(observed_values, [75 ,25])
            iqr = q75 - q25
            distr_params[distr_name] = {'c': c, 'loc': loc, 'scale': scale, 'max': q75+iqr*1.5, 'mean': np.mean(observed_values)}
            generated_values[distr_name] = dist.rvs(c=c, loc=loc, scale=scale, size=N)
        elif distr_name == 'lognormal':
            dist = stats.lognorm
            s, loc, scale = dist.fit(observed_values)
            q75, q25 = np.percentile(observed_values, [75 ,25])
            iqr = q75 - q25
            distr_params[distr_name] = {'s': s, 'loc': loc, 'scale': scale, 'max': q75+iqr*1.5, 'mean': np.mean(observed_values)}
            generated_values[distr_name] = dist.rvs(s=s, loc=loc, scale=scale, size=N)
        elif distr_name == 'gamma':
            dist = stats.gamma
            a, loc, scale = dist.fit(observed_values)
            q75, q25 = np.percentile(observed_values, [75 ,25])
            iqr = q75 - q25
            distr_params[distr_name] = {'a': a, 'loc': loc, 'scale': scale, 'max': q75+iqr*1.5, 'mean': np.mean(observed_values)}
            generated_values[distr_name] = dist.rvs(a=a, loc=loc, scale=scale, size=N)

    wass_distances = {d_name: stats.wasserstein_distance(observed_values, generated_values[d_name]) for d_name in possible_distributions}
    best_distr = min(wass_distances, key=wass_distances.get)

    return best_distr, distr_params[best_distr], wass_distances


def sample_time(distr, N=1):

    distr_name, distr_params = distr

    if distr_name == 'fixed':
        return [distr_params['value']]*N
    elif distr_name == 'normal':
        sampled_t = stats.norm(loc=distr_params['loc'], scale=distr_params['scale']).rvs(size=N)
    elif distr_name == 'exponential':
        sampled_t = stats.expon(loc=distr_params['loc'], scale=distr_params['scale']).rvs(size=N)
    elif distr_name == 'uniform':
        sampled_t = stats.uniform(loc=distr_params['loc'], scale=distr_params['scale']).rvs(size=N)
    elif distr_name == 'triangular':
        sampled_t = stats.triang(c=distr_params['c'], loc=distr_params['loc'], scale=distr_params['scale']).rvs(size=N)
    elif distr_name == 'lognormal':
        sampled_t = stats.lognorm(s=distr_params['s'], loc=distr_params['loc'], scale=distr_params['scale']).rvs(size=N)
    elif distr_name == 'gamma':
        sampled_t = stats.gamma(a=distr_params['a'], loc=distr_params['loc'], scale=distr_params['scale']).rvs(size=N)

    sampled_t[(sampled_t < 0) | (sampled_t > distr_params['max'])] = distr_params['mean']

    return sampled_t