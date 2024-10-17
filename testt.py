import numpy as np
from pymoo.problems import get_problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.indicators.hv import HV

# Set up the WFG1 problem with 2 objectives and 9 decision variables
problem = get_problem("wfg1", n_var=9, n_obj=2)

# Generate random decision variables manually as FloatRandomSampling seems to cause issues with dimensions
X = np.random.rand(100, 9)  # 100 samples with 9 decision variables

# Evaluate the WFG1 problem to get the objective values
F = problem.evaluate(X)

# Define the reference point for hypervolume calculation (all objectives worse than the known solutions)
reference_point = np.array([3, 5])

# Calculate the hypervolume using pymoo's HV indicator
hv = HV(ref_point=reference_point)

# Compute the hypervolume based on the obtained objective values
hypervolume_value = hv.do(F)

print(f"Hypervolume: {hypervolume_value}")
