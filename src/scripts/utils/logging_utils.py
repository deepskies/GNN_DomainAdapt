""" 
Logging utilities.

# --- Last update 08/22/23 by Andrea Roncoli --- #
"""
import time
from contextlib import redirect_stdout
import torch

def initial_logging(params_keys, params_values, optimize_params, device, study):
    """Print the initial logging information.
    
    Args:
        None
            
    Returns:
        None
    """
    with open("logs/log_optuna.txt", "w") as logfile:
        with redirect_stdout(logfile):
            print("Optuna optimization of the GNN hyperparameters")
            print("Date and time: {}".format(time.strftime("%c")))
            print("\n")
            print("Optimization study: {}".format(study.study_name))
            print("CUDA Available: {}".format(torch.cuda.is_available()))
            if torch.cuda.is_available():
                print("\tCUDA Device Name: {}".format(torch.cuda.get_device_name()))
                print("\tCUDA Device Count: {}".format(torch.cuda.device_count()))
                print("\tCUDA Device: {}".format(torch.cuda.current_device()))
                print("\tCUDA Device Properties: {}".format(torch.cuda.get_device_properties(device)))
                print("\tCUDA Device Memory: {:.2f} GB".format(torch.cuda.get_device_properties(device).total_memory/1.e9))
                print("\tGPU occupied memory before cleaning cache: {:.2f} GB".format(torch.cuda.memory_allocated()/1.e9))
                print("\n")
            else:
                print("\tCPU Device Name: {}".format(torch.device('cpu')))
                print("\n")
            
            print(50 * "-", "\n")

            # Print all params that are not being optimized and their values
            print("Default parameters: \n")
            for key, value in zip(params_keys, params_values):
                if key not in optimize_params:
                    print("\t{}: {}\n".format(key, value))
            print("\n")

            # Print all params that are being optimized
            print("Optimized parameters: \n")
            for key in optimize_params:
                print("\t{}\n".format(key))

def general_log(*str):
    """Print a general logging information.
    
    Args:
        *str: strings to be printed
            
    Returns:
        None
    """
    with open("logs/log_optuna.txt", "a") as logfile:
        with redirect_stdout(logfile):
            print(*str)