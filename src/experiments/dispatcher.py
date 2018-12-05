"""
    Hard coded pars:
        - batch_size: ~128
        - trainer: Adam
        - cut_arity: 30, , max_depth: 30 ( first 150.000 skipped 500 samples)
        - adam parameter
        - flat strategy
        - activation
    early stop: no avg improvements after 3000? iteration
    runs = 1
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import json
import time
import itertools

pars = {
    'embedding_size': [100, 200, 300],
    'kld_rescale': [1.0, 0.1, 0.01, 0.001, 0.0001],
    'hidden_cell_coef': [0.1, 0.2, 0.3, 0.4]
}

from itertools import product
import itertools
all_models_pars = list(map(dict, product(*[[(k,v)  for v in pars[k]] for k in pars.keys()])))
# increasingly bigger model - GPU get big model, CPU get small model

sorted(all_models_pars, key=lambda x: x['embedding_size'])

models_to_compute = [
    json.dumps({'id': i, 'pars': p}) for p,i in zip(all_models_pars, itertools.count())
]


computing, computed, failed = 0, 0, 0


def print_status():
    global computing, computed, failed, models_to_compute
    print("%d left, %d in progress, %d done, %d failed" % (len(models_to_compute), computing, computed, failed))


with open('model_selection_result.json', 'w') as results_file:
    class RequestHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            """Dispatch a new job"""

            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            worker_id = qs.get('worker_id')[0]
            worker_type = qs.get('worker_type')[0] # GPU | CPU

            global models_to_compute
            if len(models_to_compute)>0:
                if worker_type == "GPU":
                    msg = models_to_compute.pop(-1)

                else:
                    msg = models_to_compute.pop(0)

                self.send_response(200)
                self.send_header('Content-type', 'text/json; charset=utf-8')
                self.end_headers()

                print(msg)
                self.wfile.write(bytearray(msg, encoding='utf-8'))
                global computing
                computing += 1
                print_status()
            else:
                self.send_response(204)
                self.end_headers()

        def do_POST(self):
            """Get results"""
            data = self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(204)
            self.end_headers()

            # expect {'model_id': model_id, 'status': 'success' | 'failed', 'info': {}}'
            result = json.loads(str(data, encoding='utf-8'))
            print(data)
            results_file.write(json.dumps(result) + '\n')

            global computed, computing, failed, models_map
            computing -= 1
            if result['status'] == 'success':
                computed += 1
            else:
                failed += 1

            print_status()

    HTTPServer(('', 8000), RequestHandler).serve_forever()
