"""Hopfield network with asynchronous Glauber dynamics.

    P(S_i = +1 | h_i) = 1/2 [1 + tanh(h_i / T)],        h_i = sum_j W_ij S_j            (T = 0: S_i = sign(h_i))
    W_ij = (1/N) sum_mu xi^mu_i xi^mu_j,   W_ii = 0                                      (Hebb)
    E(S) = -1/2 sum_ij W_ij S_i S_j                                                      (reported per unit, E/N)
    m^mu = (1/N) sum_i xi^mu_i S_i                                                       (overlap)

One sweep updates the N units one at a time, in a new random order, each with the current state of the others.
The model, and the numbers to compare with (T_c = 1 for P/N -> 0, alpha_c = P/N = 0.138 at T = 0), are in
docs/hopfield_mean_field.pdf (Orhan, after Hertz, Krogh & Palmer) and in Gerstner et al., Neuronal Dynamics, ch. 17.2.

    python hopfield.py --N=1000 --M=3 --replicas=10 --iter=30 --Tmin=0.05 --Tmax=1.5 --Tby=0.1 --plot --threads=4
    python hopfield.py --capacity --N=500 --replicas=5 --plot --threads=4

AUTHOR: David Preti, Rome, Oct 2022.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # the vectors are short: one BLAS thread per process, the Pool does the parallelism

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------------------------- the model
def random_patterns(N, M, rng):
    """M patterns of N bits, +1 or -1 with probability 1/2 each."""
    return rng.choice(np.array([-1, 1]), size=(M, N))


def hamming_distance(x, y):
    """Fraction of the bits that differ."""
    return float(np.mean(np.asarray(x) != np.asarray(y)))


def pattern_distance(x, y):
    """Hamming distance up to the global flip: -x is as good a memory as x."""
    d = hamming_distance(x, y)
    return min(d, 1.0 - d)


class HopfieldNetwork:
    def __init__(self, x, temp=0.0):
        """x: (M, N) array of +1/-1 patterns. temp: temperature T = 1/beta."""
        self.x = np.asarray(x, dtype=np.int8)
        self.M, self.N = self.x.shape
        self.T = float(temp)
        xf = self.x.astype(float)
        self.W = xf.T @ xf / self.N
        np.fill_diagonal(self.W, 0.0)

    def energy(self, s):
        """Energy per unit, -1/2 s.W.s / N."""
        return float(-0.5 * s @ self.W @ s / self.N)

    def overlaps(self, s):
        """Overlap m^mu of the state s with every pattern, shape (M,)."""
        return self.x @ s / self.N

    def update_unit(self, s, j, rng):
        """Update unit j in place, with the current state of the others. h = 0 gives +1, as sign(0) = +1."""
        h = self.W[j] @ s
        if self.T == 0.0:
            s[j] = 1 if h >= 0 else -1
        else:
            s[j] = 1 if rng.random() < 0.5 * (1.0 + np.tanh(h / self.T)) else -1

    def sweep(self, s, rng):
        """One sweep of asynchronous Glauber dynamics, in place: all the units, one at a time, in a random order."""
        for j in rng.permutation(self.N):
            self.update_unit(s, j, rng)
        return s

    def relax(self, s0, max_sweeps, rng):
        """T = 0: sweep until a sweep flips no unit (a fixed point) or max_sweeps. Returns the state and the sweeps."""
        s = np.array(s0, dtype=float)
        for k in range(1, max_sweeps + 1):
            before = s.copy()
            self.sweep(s, rng)
            if np.array_equal(s, before):
                return s, k
        return s, max_sweeps

    def run(self, s0, sweeps, rng):
        """Run from s0. Returns the final state, the energy per unit (sweeps + 1,) and the overlaps (sweeps + 1, M)."""
        s = np.array(s0, dtype=float)
        E = [self.energy(s)]
        m = [self.overlaps(s)]
        for _ in range(sweeps):
            self.sweep(s, rng)
            E.append(self.energy(s))
            m.append(self.overlaps(s))
        return s, np.array(E), np.array(m)


def initial_state(x, start, noise, rng):
    """'pattern': pattern 0 with a fraction `noise` of the bits flipped. 'random': a random state."""
    if start == "random":
        return random_patterns(x.shape[1], 1, rng)[0].astype(float)
    s = x[0].astype(float)
    flip = rng.random(x.shape[1]) < noise
    s[flip] *= -1
    return s


# ---------------------------------------------------------------------------------------------- experiments
def mean_field_m(T, tol=1e-12):
    """Non-negative solution of m = tanh(m/T) (P/N -> 0): the PDF's Eq. 37."""
    if T >= 1.0:
        return 0.0
    m = 1.0
    for _ in range(100000):
        new = np.tanh(m / T) if T > 0 else 1.0
        if abs(new - m) < tol:
            break
        m = new
    return float(m)


def run_replicas(task):
    """All the replicas for one value of T (a function of module level, so that Pool can use it).

    Returns, for every replica: the energy per unit at every sweep, the overlap m of the final state (signed overlap
    with pattern 0 if the start is 'pattern'; the largest |m^mu| if it is 'random'), and the pattern that it is
    closest to.
    """
    x, T, replicas, sweeps, start, noise, seed = task
    rng = np.random.default_rng(seed)
    net = HopfieldNetwork(x, temp=T)
    energies, m, best, dist = [], [], [], []
    for _ in range(replicas):
        s, E, ms = net.run(initial_state(x, start, noise, rng), sweeps, rng)
        energies.append(E)
        mu = int(np.argmax(np.abs(ms[-1])))
        best.append(mu)
        m.append(ms[-1][0] if start == "pattern" else abs(ms[-1][mu]))
        dist.append(min(pattern_distance(s, x[k]) for k in range(len(x))))
    return {"T": T, "E": np.array(energies), "m": np.array(m), "best": np.array(best), "dist": np.array(dist)}


def temperature_scan(x, temps, replicas, sweeps, start, noise, threads, seed):
    seeds = np.random.SeedSequence(seed).spawn(len(temps))
    tasks = [(x, T, replicas, sweeps, start, noise, ss) for T, ss in zip(temps, seeds)]
    if threads > 1:
        with Pool(threads) as pool:
            return list(tqdm(pool.imap(run_replicas, tasks), total=len(tasks)))
    return [run_replicas(t) for t in tqdm(tasks)]


def run_capacity(task):
    """Overlap at T = 0, starting from pattern 0, for P = alpha N random patterns, after at most `sweeps` sweeps."""
    N, alpha, replicas, sweeps, seed = task
    rng = np.random.default_rng(seed)
    P = max(1, int(round(alpha * N)))
    out = []
    for _ in range(replicas):
        x = random_patterns(N, P, rng)
        net = HopfieldNetwork(x, temp=0.0)
        s, _ = net.relax(x[0], sweeps, rng)
        out.append(float(net.overlaps(s)[0]))
    return {"N": N, "alpha": alpha, "P": P, "m": np.array(out)}


def capacity_scan(N, alphas, replicas, sweeps, threads, seed):
    seeds = np.random.SeedSequence(seed).spawn(len(alphas))
    tasks = [(N, a, replicas, sweeps, ss) for a, ss in zip(alphas, seeds)]
    if threads > 1:
        with Pool(threads) as pool:
            return list(tqdm(pool.imap(run_capacity, tasks), total=len(tasks)))
    return [run_capacity(t) for t in tqdm(tasks)]


def _g(y, alpha):
    """Equation 64 of the PDF: the retrieval states at T = 0 are the roots y > 0 of g, with m = erf(y)."""
    return y * (np.sqrt(2 * alpha) + 2 / np.sqrt(np.pi) * np.exp(-y * y)) - np.array([math.erf(v) for v in np.atleast_1d(y)])


def mean_field_retrieval(alpha):
    """T = 0, N -> infinity: overlap m of the retrieval state at memory load alpha, or None above alpha_c.

    It is the largest root y of g(y) = 0 (the smaller one is the unstable state in between), m = erf(y)."""
    y = np.linspace(1e-3, 8, 80001)
    g = _g(y, alpha)
    idx = np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
    if len(idx) == 0:
        return None
    i = idx[-1]
    lo, hi = y[i], y[i + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sign(_g(mid, alpha)[0]) == np.sign(_g(lo, alpha)[0]):
            lo = mid
        else:
            hi = mid
    return math.erf(0.5 * (lo + hi))


def mean_field_capacity():
    """alpha_c: the largest alpha for which the retrieval state exists, and its overlap there (about 0.138, 0.967)."""
    lo, hi = 0.05, 0.3
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mean_field_retrieval(mid) is None:
            hi = mid
        else:
            lo = mid
    return lo, mean_field_retrieval(lo)


def finite_size_scan(Ns, alphas, replicas, sweeps, threads, seed):
    """run_capacity for every (N, alpha). Returns {N: {alpha: array of the overlaps of the replicas}}."""
    pairs = [(N, a) for N in Ns for a in alphas]
    seeds = np.random.SeedSequence(seed).spawn(len(pairs))
    tasks = [(N, a, replicas, sweeps, ss) for (N, a), ss in zip(pairs, seeds)]
    tasks.sort(key=lambda t: -t[0])  # the slow ones first
    if threads > 1:
        with Pool(threads) as pool:
            res = list(tqdm(pool.imap_unordered(run_capacity, tasks), total=len(tasks)))
    else:
        res = [run_capacity(t) for t in tqdm(tasks)]
    out = {N: {} for N in Ns}
    for r in res:
        out[r["N"]][r["alpha"]] = r["m"]
    return out


def retrieval_threshold(alphas, fraction, level=0.5):
    """alpha at which the fraction of the replicas that retrieve falls through `level` (linear interpolation)."""
    alphas = np.asarray(alphas, float)
    for i in range(len(alphas) - 1):
        f0, f1 = fraction[i], fraction[i + 1]
        if f0 >= level > f1:
            return float(alphas[i] + (alphas[i + 1] - alphas[i]) * (f0 - level) / (f0 - f1))
    return float("nan")


def alpha_c_of_N(data, Ns, alphas, threshold=0.5, boot=300, seed=0):
    """alpha_c(N) from the data of finite_size_scan: the load at which half of the replicas still retrieve (their
    final overlap is above `threshold`). The error is the spread of a bootstrap over the replicas."""
    rng = np.random.default_rng(seed)
    est, err = [], []
    for N in Ns:
        M = np.array([data[N][a] for a in alphas])            # (alphas, replicas)
        est.append(retrieval_threshold(alphas, (M > threshold).mean(1)))
        b = []
        for _ in range(boot):
            idx = rng.integers(0, M.shape[1], size=M.shape[1])
            b.append(retrieval_threshold(alphas, (M[:, idx] > threshold).mean(1)))
        b = np.array(b)[np.isfinite(b)]
        err.append(float(np.std(b, ddof=1)) if len(b) > 1 else float('nan'))
    return np.array(est), np.array(err)


# ---------------------------------------------------------------------------------------------- figures
def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.use()
    return plt, plot_style


def sem(a, axis=0):
    return np.std(a, axis=axis, ddof=1) / np.sqrt(a.shape[axis]) if a.shape[axis] > 1 else np.zeros(a.shape[1 - axis])


def plot_energy_time(res, N, M, path):
    """Energy per unit against the sweeps, for a few temperatures."""
    plt, ps = _setup()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("t", [ps.INK, ps.ACCENT])
    idx = np.unique(np.linspace(0, len(res) - 1, min(6, len(res))).round().astype(int))
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, i in enumerate(idx):
        E = res[i]["E"]
        x = np.arange(E.shape[1])
        c = cmap(k / max(1, len(idx) - 1))
        ax.fill_between(x, E.mean(0) - sem(E), E.mean(0) + sem(E), color=c, alpha=0.25, lw=0)
        ax.plot(x, E.mean(0), color=c, label=f"T = {res[i]['T']:.2f}")
    ax.set_xlabel("SWEEP")
    ax.set_ylabel(r"ENERGY PER UNIT  $\langle E \rangle / N$")
    ax.set_title(f"[ HOPFIELD ]  N = {N}  M = {M}  ENERGY AGAINST TIME")
    ps.style_axes(ax)
    ps.legend(ax, "upper left", bbox_to_anchor=(1.02, 1))
    ps.savefig(fig, path, bbox_inches="tight")
    plt.close(fig)


def plot_vs_temperature(res, N, M, start, path):
    """Final overlap and final energy against T, with the mean-field curve m = tanh(m/T)."""
    plt, ps = _setup()
    T = np.array([r["T"] for r in res])
    m = np.array([r["m"] for r in res])
    E = np.array([r["E"][:, -1] for r in res])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    tt = np.linspace(max(T.min(), 1e-3), max(T.max(), 1.2), 300)
    ax1.plot(tt, [mean_field_m(t) for t in tt], color=ps.ACCENT, lw=2.4, label=r"MEAN FIELD  $m = \tanh(m/T)$")
    ax1.errorbar(T, m.mean(1), yerr=sem(m, 1), fmt="s", ms=6, color=ps.INK, ecolor=ps.INK, elinewidth=1.2,
                 capsize=3, zorder=4, label=f"SIMULATION, {m.shape[1]} REPLICAS")
    ax1.axvline(1.0, color=ps.INK, ls=ps.DASHED, lw=1.2)
    ax1.set_xlabel("TEMPERATURE  T")
    ax1.set_ylabel(r"OVERLAP  $m$" if start == "pattern" else r"LARGEST $|m^\mu|$")
    ax1.set_title("OVERLAP WITH THE STORED PATTERN")
    ax2.errorbar(T, E.mean(1), yerr=sem(E, 1), fmt="s", ms=6, color=ps.INK, ecolor=ps.INK, elinewidth=1.2, capsize=3)
    ax2.set_xlabel("TEMPERATURE  T")
    ax2.set_ylabel(r"ENERGY PER UNIT  $E / N$")
    ax2.set_title("ENERGY AFTER THE LAST SWEEP")
    for ax in (ax1, ax2):
        ps.style_axes(ax)
    ps.legend(ax1, "lower left")
    fig.suptitle(f"[ HOPFIELD ]  N = {N}  M = {M}", x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    ps.savefig(fig, path)
    plt.close(fig)


def plot_capacity(res, N, path):
    plt, ps = _setup()
    a = np.array([r["alpha"] for r in res])
    m = np.array([r["m"] for r in res])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(a, m.mean(1), yerr=sem(m, 1), fmt="s", ms=6, color=ps.INK, ecolor=ps.INK, elinewidth=1.2, capsize=3,
                label=f"SIMULATION, N = {N}")
    ax.axvline(0.138, color=ps.ACCENT, ls=ps.DASHED, lw=2, label=r"$\alpha_c = 0.138$ (MEAN FIELD)")
    ax.set_xlabel(r"MEMORY LOAD  $\alpha = P / N$")
    ax.set_ylabel(r"OVERLAP  $m$  (START ON THE PATTERN, T = 0)")
    ax.set_title("[ HOPFIELD ]  CAPACITY")
    ps.style_axes(ax)
    ps.legend(ax, "lower left")
    ps.savefig(fig, path)
    plt.close(fig)


def fit_extrapolation(Ns, est, err, nu=None):
    """Weighted least squares  alpha_c(N) = a_inf + b N^(-nu).

    nu fixed, or free (nu=None: the exponent with the smallest chi^2 on a grid). Returns a dict with a_inf, its
    statistical error at that exponent, b, nu, chi2, and for the free fit the range of nu with chi2 within 1 of the
    minimum and the values of a_inf at its ends: the data do not know the exponent, and a_inf depends on it.
    """
    N, est, err = np.asarray(Ns, float), np.asarray(est, float), np.asarray(err, float)
    ok = np.isfinite(est)
    N, est = N[ok], est[ok]
    w = 1.0 / np.maximum(err[ok], 1e-3)  # a floor, so that one lucky point with zero spread does not dominate

    def one(v):
        A = np.vstack([np.ones(len(N)), N ** -v]).T * w[:, None]
        coef = np.linalg.lstsq(A, est * w, rcond=None)[0]
        chi2 = float(np.sum((A @ coef - est * w) ** 2))
        return coef, chi2, float(np.sqrt(np.linalg.inv(A.T @ A)[0, 0]))

    out = {}
    if nu is None:
        grid = np.linspace(0.1, 2.0, 191)
        chis = np.array([one(v)[1] for v in grid])
        nu = float(grid[int(np.argmin(chis))])
        inside = grid[chis < chis.min() + 1.0]
        out["nu_range"] = (float(inside.min()), float(inside.max()))
        out["a_inf_range"] = (float(one(inside.min())[0][0]), float(one(inside.max())[0][0]))
    coef, chi2, da = one(nu)
    out.update(a_inf=float(coef[0]), da_inf=da, b=float(coef[1]), nu=float(nu), chi2=chi2, dof=len(N) - 2)
    return out


def plot_finite_size(data, Ns, alphas, path, threshold=0.5, loads=(0.10, 0.12)):
    """Three panels: the fraction of the replicas that retrieve against alpha for every N; alpha_c(N) against
    1/sqrt(N), extrapolated to N -> infinity, against the mean field alpha_c; the overlap of the retrieval state at
    fixed alpha against 1/sqrt(N), against the mean field overlap."""
    plt, ps = _setup()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("n", [ps.INK, ps.ACCENT])
    a_mf, m_mf = mean_field_capacity()
    est, err = alpha_c_of_N(data, Ns, alphas, threshold)
    free = fit_extrapolation(Ns, est, err)           # exponent free
    half = fit_extrapolation(Ns, est, err, nu=0.5)    # the usual 1/sqrt(N), for comparison

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax1, ax2, ax3 = axes
    for k, N in enumerate(Ns):
        f = np.array([(data[N][a] > threshold).mean() for a in alphas])
        ax1.plot(alphas, f, "-s", ms=5, lw=1.4, color=cmap(k / max(1, len(Ns) - 1)), label=f"N = {N}")
    ax1.axvline(a_mf, color=ps.INK, ls=ps.DASHED, lw=1.4)
    ax1.set_xlabel(r"MEMORY LOAD  $\alpha = P / N$")
    ax1.set_ylabel(f"FRACTION OF REPLICAS WITH  m > {threshold}")
    ax1.set_title("RETRIEVAL AGAINST LOAD")
    ps.style_axes(ax1)
    ps.legend(ax1, "lower left")

    x = 1.0 / np.sqrt(np.array(Ns, float))
    xx = np.linspace(0, x.max() * 1.05, 50)
    Nf = np.array(Ns, float)
    xx = np.linspace(0, x.max() * 1.05, 200)
    curve = lambda f: f["a_inf"] + f["b"] * np.maximum(xx, 1e-9) ** (2 * f["nu"])  # N^-nu = (1/sqrt N)^(2 nu)
    ax2.axhline(a_mf, color=ps.ACCENT, lw=2.4, label=rf"MEAN FIELD  $\alpha_c = {a_mf:.3f}$")
    ax2.plot(xx, curve(half), color=ps.GREY, ls=ps.DOTTED, lw=1.6,
             label=rf"FIT  $\nu = 1/2$ (FIXED):  {half['a_inf']:.3f}")
    ax2.plot(xx, curve(free), color=ps.INK, ls=ps.DASHED, lw=1.6,
             label=rf"FIT  $\nu = {free['nu']:.2f}$ (FREE):  {free['a_inf']:.3f}")
    ax2.errorbar(x, est, yerr=err, fmt="s", ms=6, color=ps.INK, ecolor=ps.INK, elinewidth=1.2, capsize=3, zorder=4)
    ax2.errorbar([0.0], [half["a_inf"]], yerr=[half["da_inf"]], fmt="D", ms=7, mfc=ps.PAPER, mec=ps.GREY, mew=2,
                 ecolor=ps.GREY, elinewidth=2, capsize=4, zorder=5)
    ax2.errorbar([0.0], [free["a_inf"]], yerr=[free["da_inf"]], fmt="s", ms=10, mfc=ps.PAPER, mec=ps.INK, mew=2,
                 ecolor=ps.INK, elinewidth=2, capsize=5, zorder=6)
    ax2.set_xlim(-0.004, x.max() * 1.05)
    ax2.set_xlabel(r"$1/\sqrt{N}$")
    ax2.set_ylabel(r"$\alpha_c(N)$  (HALF OF THE REPLICAS RETRIEVE)")
    ax2.set_title(r"CAPACITY AGAINST SIZE:  $a + b N^{-\nu}$")
    ps.style_axes(ax2)
    ps.legend(ax2, "upper left")

    for a, c in zip(loads, (ps.INK, ps.ACCENT)):
        if not all(a in data[N] for N in Ns):
            continue
        mm, ee = [], []
        for N in Ns:
            r = data[N][a][data[N][a] > threshold]  # the replicas that retrieve
            mm.append(r.mean() if len(r) else np.nan)
            ee.append(sem(r[:, None], 0)[0] if len(r) > 1 else 0.0)
        m_th = mean_field_retrieval(a)
        ax3.axhline(m_th, color=c, ls=ps.DASHED, lw=1.4)
        ax3.errorbar(x, mm, yerr=ee, fmt="s", ms=6, color=c, ecolor=c, elinewidth=1.2, capsize=3, zorder=4,
                     label=rf"$\alpha = {a}$   (MEAN FIELD {m_th:.4f})")
    ax3.set_xlim(0, x.max() * 1.05)
    ax3.set_xlabel(r"$1/\sqrt{N}$")
    ax3.set_ylabel(r"OVERLAP  $m$  OF THE RETRIEVAL STATE")
    ax3.set_title("OVERLAP AGAINST SIZE")
    ps.style_axes(ax3)
    ps.legend(ax3, "lower left")
    fig.suptitle("[ HOPFIELD ]  FINITE SIZE, T = 0, START ON THE PATTERN", x=0.02, ha="left", fontweight="bold",
                 fontsize=10)
    fig.tight_layout()
    ps.savefig(fig, path)
    plt.close(fig)
    return est, err, free, half


# ---------------------------------------------------------------------------------------------- command line
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=1000, help="bits of a pattern")
    p.add_argument("--M", type=int, default=3, help="number of patterns")
    p.add_argument("--iter", type=int, default=30, help="sweeps")
    p.add_argument("--replicas", type=int, default=10)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1988)
    p.add_argument("--Tmin", type=float, default=0.05)
    p.add_argument("--Tmax", type=float, default=1.5)
    p.add_argument("--Tby", type=float, default=0.1)
    p.add_argument("--start", choices=["pattern", "random"], default="pattern",
                   help="start from pattern 0 (with --noise) or from a random state")
    p.add_argument("--noise", type=float, default=0.0, help="fraction of flipped bits of the starting pattern")
    p.add_argument("--capacity", action="store_true", help="scan alpha = P/N at T = 0 instead of T")
    p.add_argument("--finite-size", action="store_true", help="capacity scan for several N, extrapolated to N -> inf")
    p.add_argument("--from-json", default=None, help="finite size scan: read the data of a previous run, no simulation")
    p.add_argument("--Ns", default="100,200,400,800,1600,3200", help="sizes of the finite size scan")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default="./figs/", help="folder of the figures")
    args = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    if args.finite_size:
        Ns = [int(v) for v in args.Ns.split(",")]
        alphas = np.round(np.arange(0.08, 0.4001, 0.01), 3)
        if args.from_json:
            raw = json.load(open(args.from_json))
            data = {int(N): {float(a): np.array(m) for a, m in d.items()} for N, d in raw.items()}
            Ns = sorted(data)
            alphas = np.array(sorted(data[Ns[0]]))
        else:
            data = finite_size_scan(Ns, alphas, args.replicas, args.iter, args.threads, args.seed)
            json.dump({str(N): {str(a): list(map(float, m)) for a, m in d.items()} for N, d in data.items()},
                      open(f"{args.out}finite_size.json", "w"))
        est, err = alpha_c_of_N(data, Ns, alphas)
        a_mf, m_mf = mean_field_capacity()
        print(f"mean field: alpha_c = {a_mf:.4f}, m = {m_mf:.4f}")
        for N, e, d in zip(Ns, est, err):
            print(f"N = {N:5d}   alpha_c(N) = {e:.4f} +- {d:.4f}")
        if args.plot:
            _, _, free, half = plot_finite_size(data, Ns, alphas, f"{args.out}finite_size.png")
            for name, f in (("nu = 1/2 fixed", half), ("nu free      ", free)):
                print(f"fit {name}: alpha_c(inf) = {f['a_inf']:.4f} +- {f['da_inf']:.4f}  nu = {f['nu']:.2f}  "
                      f"chi2 = {f['chi2']:.1f} ({f['dof']} dof)")
            lo, hi = free["nu_range"]
            print(f"the free fit: nu between {lo:.2f} and {hi:.2f} within one unit of chi2, "
                  f"alpha_c(inf) between {free['a_inf_range'][0]:.4f} and {free['a_inf_range'][1]:.4f}")
    elif args.capacity:
        alphas = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.138, 0.15, 0.17, 0.20, 0.25, 0.30])
        res = capacity_scan(args.N, alphas, args.replicas, args.iter, args.threads, args.seed)
        for r in res:
            print(f"alpha = {r['alpha']:.3f}  P = {r['P']:4d}  <m> = {r['m'].mean():.3f}")
        if args.plot:
            plot_capacity(res, args.N, f"{args.out}capacity_{args.N}.png")
    else:
        x = random_patterns(args.N, args.M, rng)
        temps = np.arange(args.Tmin, args.Tmax + 1e-9, args.Tby)
        res = temperature_scan(x, temps, args.replicas, args.iter, args.start, args.noise, args.threads, args.seed)
        for r in res:
            print(f"T = {r['T']:.2f}  <m> = {r['m'].mean():+.3f}  <E/N> = {r['E'][:, -1].mean():+.3f}")
        if args.plot:
            tag = f"{args.N}-{args.M}"
            plot_energy_time(res, args.N, args.M, f"{args.out}energy_t_{tag}.png")
            plot_vs_temperature(res, args.N, args.M, args.start, f"{args.out}m_E_T_{tag}.png")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
