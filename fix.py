import os
import re
import fileinput

# Path to the hyperopt package directory
# package_path = '/usr/local/lib/python3.10/dist-packages/hyperopt'
package_path = r'C:\Projects\hypermax\3792\Lib\site-packages\hyperopt'

# Function to replace integers with randint in a given file
def replace_in_file(file_path):
    for line in fileinput.input(file_path, inplace=True):
        line = line.replace('self.rstate.integers(', 'self.rstate.randint(')
        print(line, end='')

# Recursively find all Python files and apply the replacement
for root, dirs, files in os.walk(package_path):
    for file in files:
        if file.endswith('.py'):
            replace_in_file(os.path.join(root, file))

# Verify the installation of numpy, scipy, and hyperopt
import numpy
import scipy
import hyperopt

print("NumPy version:", numpy.__version__)
print("SciPy version:", scipy.__version__)
print("Hyperopt version:", hyperopt.__version__)
