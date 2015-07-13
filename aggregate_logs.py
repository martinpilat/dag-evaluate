__author__ = 'Martin'


import sys
import json
import matplotlib.pyplot as plt
import numpy as np

def aggregate(logs, field):
    agg = map(lambda log: [x[field] for x in log], logs)
    agg = np.array(list(zip(*agg)))
    print(agg)
    return np.array([np.min(agg, axis=1), np.mean(agg, axis=1), np.max(agg, axis=1)]).T

if __name__ == '__main__':

    log_files = sys.argv[1:]

    outname = log_files[0].split('.')[0]

    logs = []
    for lf in log_files:
        log = json.load(open(lf))
        logs.append(log)

    for field in ['max_score', 'min_score', 'avg_score', 'max_time', 'min_time', 'avg_time',
                  'max_std', 'min_std', 'avg_std', 'min_size', 'max_size', 'avg_size', 'invalid_dags']:
        aggregated = aggregate(logs, field)
        out = outname + '-agg_' + field
        np.savetxt(out + '.csv', aggregate(logs, field), delimiter=',')
        plt.figure(1,  figsize=(6, 4))
        plt.xlabel('Generation number')
        plt.ylabel(field)
        plt.errorbar(np.arange(len(aggregated.T[0])), aggregated.T[1], yerr=[aggregated.T[1]-aggregated.T[0], aggregated.T[2]-aggregated.T[1]])
        plt.tight_layout()
        plt.savefig(out+'.png')
        plt.delaxes()



