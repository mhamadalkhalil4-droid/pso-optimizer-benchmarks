"""Standard test functions (minimum value, location) for optimiser benchmarking."""
import numpy as np
def sphere(x):      return np.sum(x**2)
def rosenbrock(x):  return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def rastrigin(x):   return 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def ackley(x):      return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e
def griewank(x):    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1
FUNCTIONS = {"sphere": (sphere, (-5, 5), 0.0), "rosenbrock": (rosenbrock, (-5, 5), 0.0),
             "rastrigin": (rastrigin, (-5.12, 5.12), 0.0), "ackley": (ackley, (-32, 32), 0.0), "griewank": (griewank, (-600, 600), 0.0)}
