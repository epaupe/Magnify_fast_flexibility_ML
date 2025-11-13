import multiprocessing as mp
import time
import numpy as np
import gurobipy as gp
env = gp.Env(empty=True)
env.start()
print("Gurobi version:", gp.gurobi.version())
