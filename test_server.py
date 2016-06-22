import xmlrpc_interface
import json
import time


if __name__ == '__main__':

    server = xmlrpc_interface.DagEvalServer('test_logs')

    dags = json.load(open('dagz-06-1024.json', 'r'))
    individuals = [{'code': dag, 'id':id} for id, dag in enumerate(dags)]

    print('CORE_COUNT:', server.get_core_count())
    print(len(dags))

    correct = 0
    for ind in individuals:
        if server.submit(json.dumps([ind]), 'wilt.csv') == json.dumps('OK'):
            correct += 1

    print(correct, 'tasks successfully submitted')

    i = 0
    while True:
        i+= 1
        print(i, server.get_evaluated())
        time.sleep(2)

