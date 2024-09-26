import os

def replace_in_file(file_path, old_string, new_string):
    """Replaces all occurrences of old_string with new_string in the specified file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace(old_string, new_string)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def replace_in_package(package_path, old_string, new_string):
    """Recursively replaces old_string with new_string in all Python files in the specified package directory."""
    for root, _, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                replace_in_file(file_path, old_string, new_string)

# Define the package path and the strings to replace
package_path = r'C:\Projects\HPOlib1.5\venvoo\Lib\site-packages\hpolib'
old_string = 'np.float'
new_string = 'float'  # or 'np.float64'

# Run the replacement
replace_in_package(package_path, old_string, new_string)

print("Replacement completed.")
