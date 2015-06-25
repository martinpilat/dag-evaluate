__author__ = 'Martin'

from xmlrpc.server import SimpleXMLRPCServer
import json
import eval

class DagEvalServer():

    def eval(self, json_string, datafile):
        ret = eval.eval_all(json.loads(json_string)[:10], datafile)
        return ret

if __name__ == '__main__':
    server = SimpleXMLRPCServer(('localhost', 8080))
    server.register_instance(DagEvalServer())

    server.serve_forever()
