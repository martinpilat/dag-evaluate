import json
import sys
import os

log_path = sys.argv[1]

log = []

for i in range(500):
    try:
        with open(os.path.join(log_path, 'log_%03d.json' % i)) as file:
            log.append(json.load(file))
    except Exception as e:
        print(e)
        with open(sys.argv[2], 'w') as f:
            json.dump(log, f, indent=1)
        break