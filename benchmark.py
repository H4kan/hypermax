from hypermax.optimizer import ATPEOptimizer
from hypermax.optimizer import TPEOptimizer
from hypermax.optimizer import RandomSearchOptimizer
import hpolib.benchmarks.synthetic_functions as hpobench
from hpolib.benchmarks.ml import svm_benchmark, logistic_regression
import numpy as np
from pprint import pprint
import csv

atpeOptimizer = ATPEOptimizer()
tpeOptimizer = TPEOptimizer()
randomOptimizer = RandomSearchOptimizer()

algorithms = {
    "ATPE3": atpeOptimizer,
    # "TPE": tpeOptimizer,
    # "Random": randomOptimizer
}

# Run Scipy.minimize on artificial testfunctions

h3 = hpobench.Hartmann3()
h6 = hpobench.Hartmann6()
b = hpobench.Branin()
bo = hpobench.Bohachevsky()
cb = hpobench.Camelback()
fo = hpobench.Forrester()
gp = hpobench.GoldsteinPrice()
le = hpobench.Levy()
rb = hpobench.Rosenbrock()

# logreg = svm_benchmark.SvmOnMnist()
# logreg = logistic_regression.LogisticRegression()

for f in [cb,fo,gp,le,rb]:
    info = f.get_meta_information()

    print("=" * 50)
    print(info['name'])

    space = {
        "type": "object",
        "properties": {}
    }

    for boundIndex, bound in enumerate(info['bounds']):
        space['properties'][str(boundIndex)] = {
            "type": "number",
            "scaling": "linear",
            "mode": "uniform",
            "min": bound[0],
            "max": bound[1]
        }

    increment = 0
    for name, optimizer in algorithms.items():
        print("Optimizer", name)
        losses = []
        all_round_values = []
        for round in range(50):
            best = None
            history = []
            round_values = []
            round_params = []  # To store all params for this round
            for trial in range(200):
                params = optimizer.recommendNextParameters(space, history, None, round_params)
                evalParams = [params[str(boundIndex)] for boundIndex in range(len(space['properties']))]
                val = f(evalParams)
                val += increment
                round_values.append(val)
                params['loss'] = val
                params['status'] = 'ok'
                history.append(params)
                # round_params.append(round_params[0])
                if best is None or val < best['loss']:
                    best = params
            all_round_values.append(round_values)
            print(round, best['loss'])
            losses.append(best['loss'])

            param_keys = round_params[0].keys()
            with open(f'benchmarking/{name.lower()}_{info["name"]}_{round}.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=param_keys)
                writer.writeheader()
                for params in round_params:
                    writer.writerow(params)
        averageLoss = np.mean(losses)
        averageLoss -= increment
        print("Average loss: ", averageLoss)

        with open(f'benchmarking/{info["name"]}_{name}.csv', 'w', newline='') as csvfile:
                fieldnames = [f'Round_{round}' for round in range(200)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for round_values in all_round_values:
                    row = {f'Round_{round}': val for round, val in enumerate(round_values)}
                    writer.writerow(row)