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
    "ATPE0": atpeOptimizer,
    # "TPE": tpeOptimizer,
    # "Random": randomOptimizer
}

# Run Scipy.minimize on artificial testfunctions

# h3 = hpobench.Hartmann3()
# h6 = hpobench.Hartmann6()
# b = hpobench.Branin()
# bo = hpobench.Bohachevsky()
# cb = hpobench.Camelback()
# fo = hpobench.Forrester()
# gp = hpobench.GoldsteinPrice()
le = hpobench.Levy()
rb = hpobench.Rosenbrock()

# logreg = svm_benchmark.SvmOnMnist()
# logreg = logistic_regression.LogisticRegression()

fs = [None] * 7
infos = [None] * 7

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)

    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return term1 + term2 + a + np.exp(1)

# Use Ackley function in the benchmark
fs[0] = ackley_function

# Meta information for Ackley function
infos[0] = {
    "name": "Ackley_Function",
    "bounds": [[-32.768, 32.768]] * 3  # Ackley is usually benchmarked on a large range for each dimension
}

def griewank_function(x):
    sum_part = np.sum(x ** 2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1

# Use Griewank function in the benchmark
fs[1] = griewank_function

# Meta information for Griewank function
infos[1] = {
    "name": "Griewank_Function",
    "bounds": [[-600, 600]] * 3  # Griewank function's typical domain is [-600, 600] for each dimension
}

def rastrigin_function(x):
    n = len(x)
    A = 10
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Use Rastrigin function in the benchmark
fs[2] = rastrigin_function

# Meta information for Rastrigin function
infos[2] = {
    "name": "Rastrigin_Function",
    "bounds": [[-5.12, 5.12]] * 3  # Rastrigin's typical domain is [-5.12, 5.12] for each dimension
}

def schwefel_function(x):
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Use Schwefel function in the benchmark
fs[3] = schwefel_function

# Meta information for Schwefel function
infos[3] = {
    "name": "Schwefel_Function",
    "bounds": [[-500, 500]] * 3  # Schwefel's typical domain is [-500, 500] for each dimension
}

def weierstrass_function(x, a=0.5, b=3, k_max=20):
    n = len(x)
    sum1 = np.sum([np.sum([a**k * np.cos(2 * np.pi * b**k * (x_i + 0.5)) for k in range(k_max + 1)]) for x_i in x])
    sum2 = n * np.sum([a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1)])
    return sum1 - sum2

# Use Weierstrass function in the benchmark
fs[4] = weierstrass_function

# Meta information for Weierstrass function
infos[4] = {
    "name": "Weierstrass_Function",
    "bounds": [[-0.5, 0.5]] * 3  # Weierstrass's typical domain is [-0.5, 0.5] for each dimension
}

def sphere_function(x):
    return np.sum(x ** 2)

# Use Sphere function in the benchmark
fs[5] = sphere_function

# Meta information for Sphere function
infos[5] = {
    "name": "Sphere_Function",
    "bounds": [[-5.12, 5.12]] * 3  # Sphere function's typical domain is [-5.12, 5.12] for each dimension
}

def elliptic_function(x):
    n = len(x)
    return np.sum([10**6 * (i + 1) * x_i**2 for i, x_i in enumerate(x)])

# Use Elliptic function in the benchmark
fs[6] = elliptic_function

# Meta information for Elliptic function
infos[6] = {
    "name": "Elliptic_Function",
    "bounds": [[-5.12, 5.12]] * 3  # Elliptic function's typical domain is [-5.12, 5.12] for each dimension
}



# for f in [
#     #  h3, h6, 
#     #  b, bo, cb, fo, gp, 
#         #   le, rb
#           ]:
#     info = f.get_meta_information()

for idx, f in enumerate(fs):
    info = infos[idx]

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
                
                val = f(np.array(evalParams))
                # print(evalParams)
                # print(val)
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

            # param_keys = round_params[0].keys()
            # with open(f'benchmarking/{name.lower()}_{info["name"]}_{round}.csv', 'w', newline='') as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=param_keys)
            #     writer.writeheader()
            #     for params in round_params:
            #         writer.writerow(params)
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