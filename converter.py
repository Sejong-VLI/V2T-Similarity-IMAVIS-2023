import os
import fileinput
import argparse
def get_args(description='VariableConverter'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root_path",  type=str, default='./', help='path to the root folder')
    parser.add_argument("--replace_variable", type=str, default=None,help='Name of replacement variable')

    parser.add_argument("--target_variable",  type=str, default="None",help='Target variable of name to replace')


    args = parser.parse_args()


    return args
def convert(args):
    # Define the directory path to search for Python files
    directory = args.root_path

    # Define the target variable name to replace
    target_variable = args.target_variable

    # Define the replacement variable name
    replacement_variable = args.replace_variable

    # Recursive file search and replacement
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                
                # Read the file contents
                with fileinput.FileInput(file_path, inplace=True) as file:
                    for line in file:
                        # Replace the target variable name with the replacement variable name
                        line = line.replace(target_variable, replacement_variable)
                        print(line, end='')
                        
                print(f"Replaced variable names in: {file_path}")

if __name__ == "__main__":
    args = get_args()
    convert(args)


