__author__ = 'Martin'

import json
import method_params


def read_json(file_name):
    """
    Reads the JSON file with name file_name and returns its contents.
    :param file_name: The name of the JSON file
    :return: The content of the JSON file
    """
    return json.load(open(file_name, 'r'))


def get_model_by_name(model):
    """
    Returns the model class and its parameters by its name
    :param model: str or (str, dict)
    :return: if model is str returns class of the model with name str and default parameters, otherwise returns
        the class of model with name str and parameters given by dict
    """
    if not isinstance(model, list):
        model = (model, {})
    return method_params.model_names[model[0]], model[1]
