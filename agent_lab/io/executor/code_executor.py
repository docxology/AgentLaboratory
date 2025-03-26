"""Code execution module for Agent Laboratory.

This module provides utilities for executing code and capturing the output.
"""

import io
import os
import sys
import time
import traceback
import multiprocessing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def worker_run_code(code_str, output_queue):
    """Worker process for running code in isolation.
    
    Args:
        code_str: The code to execute
        output_queue: Queue to put the output
    """
    output_capture = io.StringIO()
    sys.stdout = output_capture
    try:
        # Create a globals dictionary with __name__ set to "__main__"
        globals_dict = {"__name__": "__main__"}
        exec(code_str, globals_dict)
        
        # Capture any figures that were generated
        if 'plt' in globals_dict and hasattr(globals_dict['plt'], 'get_fignums'):
            for i in globals_dict['plt'].get_fignums():
                figure = globals_dict['plt'].figure(i)
                figure.savefig(f'Figure_{i}.png')
                output_capture.write(f"[FIGURE SAVED]: Figure_{i}.png\n")
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
        traceback.print_exc(file=output_capture)
    finally:
        sys.stdout = sys.__stdout__
    output_queue.put(output_capture.getvalue())

def execute_code(code_str, timeout=600, max_output_length=10000):
    """Execute code in a separate process with timeout.
    
    Args:
        code_str: The code to execute
        timeout: Maximum execution time in seconds
        max_output_length: Maximum length of output to return
        
    Returns:
        str: Output of the code execution or error message
    """
    # Add common imports to make code execution easier, with specific focus on POMDP implementations
    code_str = """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.special as special
from tqdm import tqdm
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
# Create output directory for results
import os
os.makedirs('results', exist_ok=True)

""" + code_str
    
    # Check for forbidden operations
    if "exit(" in code_str or "quit(" in code_str:
        return "[CODE EXECUTION ERROR] The exit() and quit() commands are not allowed. Please remove them."
    
    if "eval(" in code_str or "exec(" in code_str:
        return "[CODE EXECUTION ERROR] The eval() and exec() functions are not allowed for security reasons."
    
    # Execute the code in a separate process
    output_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=worker_run_code, args=(code_str, output_queue))
    proc.start()
    proc.join(timeout)
    
    if proc.is_alive():
        proc.terminate()  # Forcefully kill the process
        proc.join()
        return (f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds. "
                "You must reduce the time complexity of your code.")
    else:
        if not output_queue.empty():
            output = output_queue.get()
        else:
            output = ""
        
        # Save figures if they were generated
        figures_paths = []
        for filename in os.listdir("."):
            if filename.startswith("Figure_") and filename.endswith(".png"):
                # Move to output directory if provided
                figures_paths.append(filename)
                try:
                    os.makedirs('results', exist_ok=True)
                    os.rename(filename, os.path.join('results', filename))
                except Exception as e:
                    output += f"\n[ERROR MOVING FIGURE]: {str(e)}"
        
        # Also list any files that were generated in the results directory
        try:
            if os.path.exists('results'):
                output += "\n\n[GENERATED FILES]:\n"
                for filename in os.listdir("results"):
                    output += f"- {filename}\n"
        except Exception as e:
            output += f"\n[ERROR LISTING RESULTS]: {str(e)}"
        
        # Truncate output if it's too long
        if len(output) > max_output_length:
            output = output[:max_output_length] + f"\n...\n[OUTPUT TRUNCATED: exceeded {max_output_length} characters]"
        
        return output 