import os
import pickle

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = ["stanford1625231520_spec1_ovlp", "stanford1625237411_spec2_ovlp", "stanford1625242383_spec3_ovlp", "stanford1625253557_spec4_pvlp",
            "stanford1625256917_spec1_dist", "stanford1625263318_spec2_dist", "stanford1625303005_spec3_dist", "stanford1625318902_spec4_dist"]

    ID = 224
    val_list = list()
    times = list()
    for p in path:
        file = os.getcwd() + "/" + p + "/data.pkl"
        print('Reading {}'.format(file))
        data = pickle.load(open(file, 'rb'))
        sat = data['sat']

        # get data for selected ID
        values = list(sat[ID].values())
        val_list.append(values)
        times = list(sat[ID].keys())

    # create plot
    font = {'family': 'normal',
            # 'weight': 'bold',
            'size': 10}
    # matplotlib.rc('font', **font)
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['font.sans-serif'] = 'Latin Modern Math'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 10

    plt.figure(figsize=(3.5, 2.5))
    plt.plot(times, val_list[0], '-r', linewidth=2)
    plt.plot(times, val_list[1], '-g', linewidth=2)
    plt.plot(times, val_list[2], '-b', linewidth=2)
    plt.plot(times, val_list[3], '-k', linewidth=2)
    plt.plot([times[0], times[-1]], [0, 0], '-k', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Satisfaction of Specification')
    plt.legend(['Spec1', 'Spec2', 'Spec3', 'Spec4'])
    plt.savefig('Stanford_satisfaction_id_' + str(ID) + '.svg')
    plt.show()
