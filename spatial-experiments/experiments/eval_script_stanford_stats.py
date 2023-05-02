import os
import pickle

import numpy as np

if __name__ == '__main__':
    # path = "stanford1625231520_spec1_ovlp"
    # path = "stanford1625237411_spec2_ovlp"
    # path = "stanford1625242383_spec3_ovlp"
    # path = "stanford1625253557_spec4_pvlp"
    # path = "stanford1625256917_spec1_dist"
    # path = "stanford1625263318_spec2_dist"
    # path = "stanford1625303005_spec3_dist"
    path = "stanford1625318902_spec4_dist"
    file = os.getcwd() + "/" + path + "/data.pkl"

    print('Reading {}'.format(file))
    # /Users/chris/Repos/STLogic/code/scripts/stanford1625046979_spec1/data.pkl
    # load pickle file
    # structure: {'t_s':t_s, 't_d':t_d, 'tobj':time_objects, 'sat':satisfaction_per_object , 'spec':spec}
    data = pickle.load(open(file, 'rb'))

    sat = data['sat']

    no_of_satisfaction = 0
    no_objects = 0
    sat_range = [np.inf, -np.inf]
    sat_id = [0, 0]
    satisfaction_values = list()
    for o in list(sat.keys()):
        no_objects += 1
        values = np.array(list(sat[o].values()))
        for v in values:
            satisfaction_values.append(v)
        if np.max(values) >= 0:
            no_of_satisfaction += 1
        if np.max(values) > sat_range[-1]:
            sat_range[-1] = np.max(values)
            sat_id[-1] = o
        if np.min(values) < sat_range[0]:
            sat_range[0] = np.min(values)
            sat_id[0] = o

    # print statistics
    print('Spec is \n{}'.format(data['spec']))
    print('No of objects in dataset = {}'.format(no_objects))
    print('No of satisfying objects in dataset = {} (which is {})'.format(no_of_satisfaction, no_of_satisfaction / no_objects))
    print('No of violating objects in dataset = {} (which is {})'.format(no_objects - no_of_satisfaction, 1 - no_of_satisfaction / no_objects))
    print('Satisfaction range is from {} to {}'.format(sat_range[0], sat_range[1]))
    print('Satisfaction has mean {} and stddev {}'.format(np.mean(satisfaction_values), np.std(satisfaction_values)))
    print('Satisfaction ids are {} and {}'.format(sat_id[0], sat_id[1]))
