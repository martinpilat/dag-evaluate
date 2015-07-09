__author__ = 'Martin'

from xmlrpc.server import SimpleXMLRPCServer
import json
import eval
import method_params
import pandas as pd
import os
import sys

stop_server = False

class DagEvalServer:

    def __init__(self, log_path):
        self.log_path = log_path
        self.gen_number = 0

    def eval(self, json_string, datafile):
        """
        Evaluates all dags described by the json_string on the dataset with name datafile.
        :param json_string: JSON string containing the list of dags to evaluate.
        :param datafile: The dataset to evaluate the dags on.
        :return: List of tuples with the scores of each dag on the dataset
        """
        dags = json.loads(json_string)
        ret = eval.eval_all(dags, datafile)
        log_file = os.path.join(self.log_path, 'log_%03d.json' % self.gen_number)
        json.dump(list(zip(dags, ret)), open(log_file, 'w'), indent=1)
        self.gen_number += 1
        return ret

    def get_param_sets(self, datafile):
        """
        Returns the set of possible values of parameters for each method based on the given datafile.
        :param datafile: The name of the dataset for which the parameters should be generated. Some parameters depend
            e.g. on the number of attributes in the dataset (like the n_components in PCA).
        :return: The JSON string containing the dictionary of the parameter values for each supported method.
        """
        ds = pd.read_csv(os.path.join('data', datafile), sep=';')
        num_instances, num_features = ds.shape
        return method_params.create_param_set(num_features - 1, num_instances)

    def quit(self):
        global stop_server
        stop_server = True

if __name__ == '__main__':

    log_path = sys.argv[1]
    print('log', log_path)

    server = SimpleXMLRPCServer(('localhost', 8080))
    server.register_instance(DagEvalServer(log_path))

    while not stop_server:
        server.handle_request()
