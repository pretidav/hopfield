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
    """Overlap at T = 0 after `sweeps` sweeps, starting from pattern 0, for P = alpha N random patterns."""
    N, alpha, replicas, sweeps, seed = task
    rng = np.random.default_rng(seed)
    P = max(1, int(round(alpha * N)))
    out = []
    for _ in range(replicas):
        x = random_patterns(N, P, rng)
        net = HopfieldNetwork(x, temp=0.0)
        s, _, ms = net.run(x[0], sweeps, rng)
        out.append(ms[-1][0])
    return {"alpha": alpha, "P": P, "m": np.array(out)}


def capacity_scan(N, alphas, replicas, sweeps, threads, seed):
    seeds = np.random.SeedSequence(seed).spawn(len(alphas))
    tasks = [(N, a, replicas, sweeps, ss) for a, ss in zip(alphas, seeds)]
    if threads > 1:
        with Pool(threads) as pool:
            return list(tqdm(pool.imap(run_capacity, tasks), total=len(tasks)))
    return [run_capacity(t) for t in tqdm(tasks)]


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
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default="./figs/", help="folder of the figures")
    args = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    if args.capacity:
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
