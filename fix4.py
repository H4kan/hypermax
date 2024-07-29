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
package_path = r'C:\Projects\hypermax\\3792\Lib\site-packages\hpolib'

# Define replacements
replacements = {
    'from sklearn.cross_validation import train_test_split': 'from sklearn.model_selection import train_test_split'
}

# Run the replacements
replace_in_package(package_path, replacements)

print("Replacement completed.")
