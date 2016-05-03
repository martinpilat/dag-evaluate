# Python Evaluator for Machine Learning Workflows

This project allows for the evaluation of whole machine learning workflows
specified in a simple JSON format. As such, it can be considered an extension of
the Pipeline class in scikit-learn. Moreover, it contains a simple XMLRPC
interface which allows for parallel evaluation of a number of worflows at once.

The project originated as part of GP-ML, a framework for the automatic design
of machine learning workflows based on genetic programming, where it was used
for the evaluation of the population in a distributed way. The population 
is provided through the XMLRPC interface as a single JSON-formatted string
which contains a list of workflows. Each workflow is a dictionary of methods
indexed by (workflow-unique) identifiers. Each method is in turn described
as a triple of (list of input IDs, method with parameters, list of output IDs).

## Getting Started

To start using the project, you simply need to start the `xmlrpc_interface.py`
script to start the server on `localhost:8080`.

```bash
python -m scoop xmlrpc_interface.py <log_path>
```

Once started, the server can be called through the XML-RPC and provides three
basic functions

1. `get_param_sets(datafile)` returns the set of parameters for all 
   machine learning methods the server understands based on the input `datafile` 
   (the ranges of parameters for some methods may change depending e.g. the 
   number of attributes in the dataset)
2. `evaluate(json_string, datafile)` evaluates the machine learning workflow
   described by the `json_string` (see below) on the `datafile`
3. `quit()` stops the server

## Format of the input JSON

The input JSON contains a list of machine learning worklows `[wf_1, ..., wf_n]`
which should be evaluated on the same dataset. Each workflow in the list is a
dictionary `{"method1": method_spec1, "method2": method_spec2, ...}` which
contains the specification of at least two methods. Each method specification,
in turn, is a triple `[inputs, [method, parameters], outputs]`.

In each workflow, an `input` method must be specified, this method has no
inputs, and is used only to rename the input file to an ID which is used in the
rest of the workflow. For example, the following method specification tells the
system that the input file will be referenced as `"123:0"` in the rest of the 
workflow.

```python
{
 "input" : [ [], "input", ["123:0"] ],
 ...
}
```

The workflow further contains any number of other methods. Each method must 
provide the list of its inputs, its name and parameters and a list of outputs.
There are methods of four types

1. `splitter` is a methods which has a single input and provides a list of
   outputs, for example the `k`-means algorithm can be used to split the data
   into clusters
2. `transformer` is a method with a single input and a single output which is
   used transform the data in any way, for example PCA analysis or feature
   selection can be performed with a splitter
3. `classifier/regressor` is a method, which is capable of learning from the 
   data and provides predictions for new data
4. `aggregator` is a method with several inputs and a single output, this method
   is used to aggregate results from several classifiers or regressors,
   currently only voting is supported as an agreggator (however, the
   implementation of voting can be used to combine the results if the dataset
   was split into several smaller dataset)


Inputs and outputs of the methods are specified by unique IDs. These IDs only
connect the outputs of one method to the inputs of other methods and their
values do not affect the results in any way. The same holds for the keys of the
methods in the dictionary. The only exception to this rule is the `input` method,
which must have this name.

The last method in the workflow must contain an empty list of outputs, which 
signifies that the output its output is the output of the whole workflow.

## Example specifications

Let's provide a few examples of workflow specifications.

### A single method

This workflow contains only a single decision tree classifier with depth limited
to 10 used on the input data directly. 

```python
{
  "input" : [ [], "input", ["IN:0"] ],
  "DT" : [ ["IN:0"], ["DT", {"max_depth": 10}], [] ],
}
```

### Three methods and voting

This workflow contains a decision tree with maximum depth of 10, gaussian naive
Bayes classifier, and a support vector classifier with the complexity constant
set to 0.5. The results of these three methods are aggregated with voting.

```python
{
  "input" : [ [], "input", ["IN:0"] ],
  "DT" : [ ["IN:0"], ["DT", {"max_depth": 10}], [DT:0] ],
  "GNB" : [ ["IN:0"], ["gaussianNB", {}], [GNB:0] ],
  "SVC" : [ ["IN:0"], ["SVC", {"C": 0.5}], ["SVC:0"] ],
  "vote" : [ ["DT:0", "GNB:0", "SVC:0"], ["vote", {}], [] ]
}
```

### A more complex example

In this example, the data are first pre-processed using the PCA analysis and
then split into two groups using the `k`-means clustering. The first group is
classified using the support vector classifier, the other group is classified
with a decision tree. The results of these two methods are merged back together
with voting. Additionally, the raw data are also processed with another decision
tree. Finally, voting is used to aggregate the results of the last split.

```python
{
  "input" : [ [], "input", ["IN:0"] ],
  "PCA" : [ ["IN:0"], ["PCA", {"feat_frac": 0.1}], ["PCA:0"] ],
  "kMeans" : [ ["PCA:0"], ["kMeans", {}], ["kM:0", "kM:1"] ],
  "SVC" : [ ["kM:0"], ["SVC", {}], ["SVC:0"] ],
  "DT1" : [ ["kM:1"], ["DT", {}], ["DT1:0"] ],
  "vote1" : [ ["SVC:0", "DT1:0"], ["vote", {}], ["VT1:0"] ],
  "DT2" : [ ["IN:0"], ["DT", {}], ["DT2:0"] ],
  "vote2" : [ ["VT1:0", "DT2:0"], ["vote", {}], [] ]
}
```

## Requirements

1. Python 3
2. numpy, scipy, pandas, scikit-learn, matplotlib, scoop

## How it works

Internally, the evaluator runs in iterations. In each iteration, it checks all
the unprocessed methods and selects those that have all data available. These
are then trained on their data. It essentially implements a simple topological 
sorting algorithm.

## Add a new method

Adding new methods is simple for new machine learning methods, which implement
the scikit-learn interface, just add the method to the `model_names` dictionary
in `method_params.py` and add the possible values of its parameters to the 
`create_param_set` method.

For the feature selection methods, do the same, and additionally add the
handling of the number of selected features to the `train_dag` method in
`eval.py`. This currently cannot be handled automatically, as the methods
require the number of features which is not known in advance (e.g. if such a 
method follows another feature selection methods). Therefore, the number of
features is handled as a fraction during the parameter search and is transformed
to the actual number during training.
