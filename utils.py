__author__ = 'Martin'

import json
import method_params
import networkx as nx


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

def dag_to_nx(dag):
    graph = nx.DiGraph()
    for node in dag.keys():
        graph.add_node(node)

    edges_from = dict()
    edges_to = dict()

    for node, (ins, mod, outs) in dag.items():
        if not isinstance(outs, list):
            outs = [outs]
        for out in outs:
            edges_from[out] = node
            edges_to[out] = []

    for node, (ins, mod, outs) in dag.items():
        if not isinstance(ins, list):
            ins = [ins]
        for i in ins:
            edges_to[i].append(node)

    for edge_name, from_node in edges_from.items():
        for to_node in edges_to[edge_name]:
            graph.add_edge(from_node, to_node)

    return graph

def to_pydot_graph(dag, sub_graph=False, input_edge=None):
    import pydot_ng as pydot
    gid = dag['input'][2][:-2]
    graph = pydot.Dot(graph_type='digraph')
    if sub_graph:
        graph = pydot.Cluster(label='booster')
    dag_nx = dag_to_nx(dag)

    for n in dag_nx.nodes():
        label = 'input' if dag[n][1] == 'input' else (dag[n][1][0] + ('(' + ','.join('{}={}'.format(k, v) for k, v in dag[n][1][1].items()) + ')' if dag[n][1][1] else ''))
        label = 'input' if dag[n][1] == 'input' else (dag[n][1][0] + (
        '(' + ','.join('{}'.format(v) for k, v in dag[n][1][1].items()) + ')' if dag[n][1][1] else ''))
        node_name = n
        if label=='input':
            if sub_graph:
                continue
            node_name = label+gid
            if input_edge != None:
                outs = dag[n][2]
                if not isinstance(outs, list):
                    outs = [outs]
                for o in outs:
                    edge = pydot.Edge(input_edge, o)
                    graph.add_edge(edge)


        if dag[n][1][0] == 'booster':
            graph.add_node(pydot.Node(n, label='booster'))
            for sd in dag[n][1][1]['sub_dags']:
                subgraph = to_pydot_graph(sd, sub_graph=True, input_edge=dag[n][0])
                graph.add_subgraph(subgraph)
        else:
            node = pydot.Node(node_name, label=label)
            graph.add_node(node)

    for (f, t) in dag_nx.edges():
        if f == 'input':
            if input_edge == None:
                f = 'input' + gid
            else:
                f = input_edge
        edge = pydot.Edge(f, t)
        graph.add_edge(edge)

    return graph

def draw_dag(dag):
    graph = to_pydot_graph(dag)
    gid = dag['input'][2][:-2]
    graph.write_pdf('graph_'+gid+'.pdf')
    graph.write_dot('graph_'+gid+'.dot')




