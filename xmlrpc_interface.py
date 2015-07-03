__author__ = 'Martin'

from xmlrpc.server import SimpleXMLRPCServer
import json
import eval
import method_params
import pandas as pd
import os


class DagEvalServer:

    def eval(self, json_string, datafile):
        """
        Evaluates all dags described by the json_string on the dataset with name datafile.
        :param json_string: JSON string containing the list of dags to evaluate.
        :param datafile: The dataset to evaluate the dags on.
        :return: List of tuples with the scores of each dag on the dataset
        """
        ret = eval.eval_all(json.loads(json_string)[:10], datafile)
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
        print(num_features, num_instances)
        return method_params.create_param_set(num_features, num_instances)

if __name__ == '__main__':
    server = SimpleXMLRPCServer(('localhost', 8080))
    server.register_instance(DagEvalServer())

    server.serve_forever()
