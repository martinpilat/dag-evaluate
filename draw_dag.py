import eval
import utils
import json

dag = json.load(open('dag.json'))


dag = eval.normalize_dag(dag)
utils.draw_dag(dag)
