# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:57:29 2025

@author: rices
"""

import subprocess

# The name of the script you want to run
script_name = "tinySA.py"

# Run the script 10 times
for i in range(10):
    filename = f"output_{i + 1}.csv"
    print(f"Running iteration {i + 1}")
    args = ['-S', str(100e6), '-E', str(900e6) ,'-o', filename]
    print(args)
    subprocess.run(["python", script_name] + args)