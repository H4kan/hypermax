import os

def replace_in_file(file_path, replacements):
    """Replaces all occurrences of old strings with new strings in the specified file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    for old_string, new_string in replacements.items():
        content = content.replace(old_string, new_string)
        
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def replace_in_package(package_path, replacements):
    """Recursively replaces old strings with new strings in all Python files in the specified package directory."""
    for root, _, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                replace_in_file(file_path, replacements)

# Define the package path and the strings to replace
package_path = r'C:\Projects\HPOlib1.5\venvoo\Lib\site-packages\theano'

# Define replacements
replacements = {
    'SafeConfigParser': 'ConfigParser',
    'from configparser import ConfigParser': 'from configparser import ConfigParser as SafeConfigParser'
}

# Run the replacements
replace_in_package(package_path, replacements)

print("Replacement completed.")
