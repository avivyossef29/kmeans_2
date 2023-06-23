import os

def compare_outputs(file_path, n):
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the absolute paths of the input files
    output_file_path = os.path.join(current_dir, file_path)

    # Read the contents of output_1.txt
    with open(output_file_path, 'r') as file:
        file_output = file.read()

    # Execute kmeans_pp.py and capture the printed output
    import subprocess
    args = [
        ['python3', 'kmeans_pp.py', '3', '333', '0', 'input_1_db_1.txt', 'input_1_db_2.txt'],
        ['python3', 'kmeans_pp.py', '7', '0', 'input_2_db_1.txt', 'input_2_db_2.txt'],
        ['python3', 'kmeans_pp.py', '15', '750', '0', 'input_3_db_1.txt', 'input_3_db_2.txt']
    ]
    script_output = subprocess.check_output(args[n]).decode('utf-8')

    # Compare the two outputs
    if file_output == script_output:
        print("The outputs match.")
    else:
        # Write the outputs to a new text file
        with open('output_diff.txt', 'w') as diff_file:
            diff_file.write(script_output)
        print("The outputs do not match.")
        print("Output difference has been saved to 'output_diff.txt'.")

# Run the function to compare the outputs
compare_outputs('output_1.txt', 0)
compare_outputs('output_2.txt', 1)
#compare_outputs('output_3.txt', 2)
