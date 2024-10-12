import os
import random

def check_and_delete_file(file_path):
    """
    Check if a file exists at the specified path. 
    If it does, delete it. If not, print a message.

    Args:
        file_path (str): The path of the file to check and delete.
    """
    if os.path.exists(file_path):
        print(f"File '{file_path}' exists. Deleting it...")
        os.remove(file_path)
    else:
        print(f"File '{file_path}' does not exist.")

def create_file_with_random_weights(file_path, num_rows=20):
    """
    Create a file at the specified path and generate random weights.
    Each weight is a float value between 0 and 1, and three weights are 
    generated per row. The weights are stored in three separate lists 
    which are returned as a tuple.

    Args:
        file_path (str): The path where the file will be created.
        num_rows (int): The number of rows of random weights to generate 
                        (default is 20).

    Returns:
        tuple: A tuple containing three lists: 
               - w_perf: List of performance weights
               - w_adp: List of adaptation weights
               - w_res: List of resource weights
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    w_perf = []
    w_adp = []
    w_res = []

    with open(file_path, "w") as file:
        for _ in range(num_rows):
            perf = random.uniform(0, 1)
            adp = random.uniform(0, 1)
            res = random.uniform(0, 1)

            w_perf.append(perf)
            w_adp.append(adp)
            w_res.append(res)
            
            row = f"{perf:.2f},{adp:.2f},{res:.2f}"
            file.write(row + "\n")

    print(f"File '{file_path}' has been created with random weights.")
    return w_perf, w_adp, w_res

# file_path = 'load_balancer/Weights/q_learning_weights.txt'
# weights = create_file_with_random_weights(file_path, num_rows=20)
# print("Generated weights:", weights)
