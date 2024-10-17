import numpy as np
import csv
from pymoo.problems import get_problem
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from hypermax.optimizer import AMOTPEOptimizer

# Initialize the WFG1 problem with 3 objectives and 10 decision variables
wfg_problems = [
    (get_problem("wfg1", n_var=9, n_obj=2), 2, 9),  # WFG1: 3 objectives, 10 decision variables
]

# Reference point for hypervolume calculation (should be worse than all expected solutions)
reference_point = np.array([3.0, 3.0, 3.0])  # Adjust this based on the expected objective range

# Loop through each WFG function and its parameters
for wfg_function, m, n in wfg_problems:

    info = {"name": f"{wfg_function.__class__.__name__}(m={m}, n={n})", "bounds": [(0.0, 1.0)] * n}
    print("=" * 50)
    print(info['name'])

    space = {
        "type": "object",
        "properties": {}
    }

    # Define the search space based on decision variables (n)
    for boundIndex, bound in enumerate(info['bounds']):
        space['properties'][str(boundIndex)] = {
            "type": "number",
            "scaling": "linear",
            "mode": "uniform",
            "min": bound[0],
            "max": bound[1]
        }

    increment = 0
    name = "AMOTPE0"
    print("Optimizer:", name)
    hvs = []

    # Multiple rounds of optimization
    for round_idx in range(1):  # Number of rounds
        optimizer = AMOTPEOptimizer()

        best = None
        history = []
        pareto_front = []
        round_params = []  # To store all parameters for this round

        # Perform optimization trials
        for trial in range(3):  # Number of trials per round
            params = optimizer.recommendNextParameters(space, history, wfg_function, None, round_params)
            eval_params = [params[f'root.{boundIndex}'] for boundIndex in range(n)]

            # Evaluate the WFG function and get objective values
            objectives = wfg_function.evaluate(np.array([eval_params]))

            # Add the solution's objectives to the Pareto front candidate list
            pareto_front.append(objectives[0])

            # Use the sum of objectives as a basic loss function for ranking
            val = sum(objectives[0]) + increment
            params['loss'] = val
            params['status'] = 'ok'
            history.append(params)

            if best is None or val < best['loss']:
                best = params

        # Convert Pareto front to a NumPy array
        pareto_front = np.array(pareto_front)

        # Perform non-dominated sorting to identify the non-dominated front
        non_dominated_solutions = NonDominatedSorting().do(pareto_front, only_non_dominated_front=True)

        # Calculate Hypervolume based on the non-dominated solutions
        hv = Hypervolume(ref_point=reference_point)
        hypervolume_value = hv.do(pareto_front[non_dominated_solutions])
        
        print(f"Round {round_idx}: HV = {hypervolume_value}")
        hvs.append(hypervolume_value)

    # Save the hypervolume results to a CSV file
    with open(f'benchmarking/{info["name"]}_{name}.csv', 'w', newline='') as csvfile:
        fieldnames = [f'Round_{round}' for round in range(len(hvs))]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({f'Round_{i}': hv for i, hv in enumerate(hvs)})
