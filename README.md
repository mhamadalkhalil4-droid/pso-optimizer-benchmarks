# pso_lib — a small Particle Swarm Optimiser, benchmarked honestly

The PSO I wrote for my turbofan-cycle project, extracted into a reusable module and tested
properly: 5 standard functions, 10 dimensions, 20 random seeds, against SciPy's differential
evolution with a matched evaluation budget.

Features: linearly decreasing inertia, velocity clamping, bound reflection, penalty-function
constraints, early stopping, convergence history. ~50 lines, NumPy only.

```python
from pso_lib import pso
best_x, best_f, history = pso(fun, lb, ub, n_particles=40, n_iter=300, constraint=lambda x: x[0]*x[1] - 60)
```

## Benchmark (10-D, 300 iterations × 40 particles, 20 seeds; "hit" = f < 1e-2)

| Function | PSO median | PSO hit | DE median | DE hit | time PSO / DE |
|---|---|---|---|---|---|
| Sphere | 1e-23 | 100 % | 2e-24 | 100 % | 0.05 s / 0.28 s |
| Ackley | 2e-11 | 100 % | 2e-11 | 100 % | 0.13 s / 0.49 s |
| Griewank | 0.095 | 0 % | 0.11 | 0 % | 0.09 s / 0.38 s |
| Rastrigin | 5.97 | 0 % | 2.98 | 0 % | 0.07 s / 0.34 s |
| Rosenbrock | 4.57 | 0 % | **0.078** | 10 % | 0.08 s / 0.32 s |

PSO is 4–5× faster per run and equal on the easy landscapes; DE is clearly better on Rosenbrock's
curved valley and somewhat better on Rastrigin's many local minima. That matches the literature: PSO
is a good first choice for smooth-ish engineering objectives with a few variables (which is what the
turbofan problem was), not a universal answer.

![convergence](figures/convergence.png)
![rastrigin](figures/rastrigin_2d.png)

## Run

```bash
pip install numpy scipy matplotlib
python benchmark.py
```

Next: constriction-factor variant, ring topology (local best) to fight premature convergence on
Rastrigin, and a hybrid with a local Nelder–Mead polish.
