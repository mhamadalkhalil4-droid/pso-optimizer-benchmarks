"""Benchmark PSO on 5 standard functions in 10-D, 20 seeds each, against scipy's differential evolution."""
import numpy as np, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from pso_lib import pso, FUNCTIONS

DIM, SEEDS = 10, 20
rows = []; hists = {}
for name, (f, (lo, hi), fmin) in FUNCTIONS.items():
    lb, ub = [lo] * DIM, [hi] * DIM
    t0 = time.time(); res = [pso(f, lb, ub, n_particles=40, n_iter=300, seed=s) for s in range(SEEDS)]; t_pso = (time.time() - t0) / SEEDS
    vals = np.array([r[1] for r in res]); hists[name] = np.array([r[2] for r in res])
    t0 = time.time(); de = [differential_evolution(f, list(zip(lb, ub)), seed=s, maxiter=300, popsize=4, tol=0, polish=False).fun for s in range(SEEDS)]; t_de = (time.time() - t0) / SEEDS
    rows.append((name, np.median(vals), np.min(vals), np.mean(vals < 1e-2), np.median(de), np.mean(np.array(de) < 1e-2), t_pso, t_de))
print(f"{'function':11s} {'PSO median':>11s} {'PSO best':>10s} {'PSO hit%':>8s} {'DE median':>11s} {'DE hit%':>7s} {'t_PSO':>6s} {'t_DE':>6s}")
for r in rows: print(f"{r[0]:11s} {r[1]:11.2e} {r[2]:10.2e} {r[3]*100:7.0f}% {r[4]:11.2e} {r[5]*100:6.0f}% {r[6]:5.2f}s {r[7]:5.2f}s")

fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
for a_, (name, H) in zip(ax, hists.items()):
    a_.semilogy(np.median(H, 0) + 1e-16, "b", lw=2, label="median"); a_.fill_between(range(H.shape[1]), np.min(H, 0) + 1e-16, np.max(H, 0) + 1e-16, alpha=.2)
    a_.set_title(f"{name} ({DIM}-D)"); a_.set_xlabel("iteration"); a_.grid(alpha=.3, which="both")
ax[0].set_ylabel("best f (20 seeds)"); plt.tight_layout(); plt.savefig("figures/convergence.png", dpi=130)

# 2-D landscape with swarm trajectory on Rastrigin
f = FUNCTIONS["rastrigin"][0]; rng = np.random.default_rng(0)
xs = np.linspace(-5.12, 5.12, 200); X, Y = np.meshgrid(xs, xs); Z = np.array([[f(np.array([a, b])) for a in xs] for b in xs])
bx, bf, h = pso(f, [-5.12, -5.12], [5.12, 5.12], n_particles=30, n_iter=100, seed=3)
plt.figure(figsize=(6, 5)); plt.contourf(X, Y, Z, 40, cmap="viridis"); plt.colorbar(label="f"); plt.plot(*bx, "r*", ms=15, label=f"PSO: f = {bf:.2e}")
plt.title("Rastrigin 2-D — PSO result"); plt.legend(); plt.tight_layout(); plt.savefig("figures/rastrigin_2d.png", dpi=130)
