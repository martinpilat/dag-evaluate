__author__ = 'Martin'


import sys
import json
import os
import numpy as np

def iterate_logs(log_path):
    log_pattern = 'log_%03d.json'

    num = 0
    while True:
        try:
            yield json.load(open(os.path.join(log_path, log_pattern % num)))
            num += 1
        except IOError:
            raise StopIteration

if __name__ == '__main__':

    log_path = sys.argv[1]

    config = json.load(open(os.path.join(log_path, 'run_1', 'config.json')))

    outname = '%s-%d_log.json' % (config['dataset'], config['seed'])

    print('Processing logs for dataset %s, seed=%d' % (config['dataset'], config['seed']))

    logs = []
    for log in iterate_logs(log_path):
        scores = [l[1][0] for l in log if l[1]]
        stds = [l[1][1] for l in log if l[1]]
        times = [l[1][2] for l in log if l[1]]
        sizes = [len(l[0]) - 1 for l in log if l[1]]
        ldata = {
            'scores': scores,
            'stds': stds,
            'times': times,
            'sizes': sizes,
            'invalid_dags': len(log) - len(scores),
            'avg_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'avg_size': float(np.mean(sizes)),
            'max_size': float(np.max(sizes)),
            'min_size': float(np.min(sizes)),
            'avg_time': float(np.mean(times)),
            'max_time': float(np.max(times)),
            'min_time': float(np.min(times)),
            'avg_std': float(np.mean(stds)),
            'max_std': float(np.max(stds)),
            'min_std': float(np.min(stds))
        }
        logs.append(ldata)

    json.dump(logs, open(outname, 'w'), indent=1, sort_keys=True)
