"""Particle Swarm Optimisation — a small, dependency-free implementation.

Features: inertia weight (constant or linearly decreasing), velocity clamping, box constraints
with reflection, optional penalty function for inequality constraints, convergence history.
"""
import numpy as np

def pso(fun, lb, ub, n_particles=40, n_iter=200, w=(0.9, 0.4), c1=1.5, c2=1.5,
        vmax_frac=0.2, constraint=None, penalty=1e6, seed=None, tol=None):
    """Minimise fun(x) for lb <= x <= ub.

    constraint(x) -> array of g(x) with feasibility g(x) <= 0 (optional, penalised).
    tol: stop early when the best value improves by less than tol over 20 iterations.
    Returns best_x, best_f, history (best f per iteration).
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float); dim = lb.size
    w0, w1 = (w, w) if np.isscalar(w) else w
    def cost(x):
        f = fun(x)
        if constraint is not None:
            f += penalty * np.sum(np.maximum(constraint(x), 0) ** 2)
        return f
    x = lb + (ub - lb) * rng.random((n_particles, dim)); v = np.zeros_like(x); vmax = vmax_frac * (ub - lb)
    pbest_x, pbest_f = x.copy(), np.array([cost(xi) for xi in x])
    g = np.argmin(pbest_f); gbest_x, gbest_f = pbest_x[g].copy(), pbest_f[g]; hist = [gbest_f]
    for it in range(n_iter):
        wi = w0 + (w1 - w0) * it / max(n_iter - 1, 1)
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        v = np.clip(wi * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x), -vmax, vmax)
        x = x + v
        over, under = x > ub, x < lb                           # reflect at the bounds
        x[over] = 2 * np.broadcast_to(ub, x.shape)[over] - x[over]; v[over] *= -0.5
        x[under] = 2 * np.broadcast_to(lb, x.shape)[under] - x[under]; v[under] *= -0.5
        x = np.clip(x, lb, ub)
        f = np.array([cost(xi) for xi in x]); better = f < pbest_f
        pbest_x[better], pbest_f[better] = x[better], f[better]
        g = np.argmin(pbest_f)
        if pbest_f[g] < gbest_f: gbest_x, gbest_f = pbest_x[g].copy(), pbest_f[g]
        hist.append(gbest_f)
        if tol is not None and it > 20 and hist[-21] - hist[-1] < tol: break
    return gbest_x, gbest_f, np.array(hist)
