# Hopfield Model

Hopfield network with asynchronous Glauber dynamics, for the temperature and the memory load of the mean field analysis in [docs/hopfield_mean_field.pdf](docs/hopfield_mean_field.pdf) (Orhan, after Hertz, Krogh & Palmer) and in [Neuronal Dynamics, chapter 17.2](https://neuronaldynamics.epfl.ch/online/Ch17.S2.html).

## The model

```
W_ij = (1/N) sum_mu xi^mu_i xi^mu_j,  W_ii = 0            Hebb
h_i  = sum_j W_ij S_j
P(S_i = +1) = 1/2 [1 + tanh(h_i / T)]                     T = 0: S_i = sign(h_i), sign(0) = +1
E    = -1/2 sum_ij W_ij S_i S_j                           printed and plotted per unit, E/N
m^mu = (1/N) sum_i xi^mu_i S_i                            overlap with the pattern mu
```

A sweep updates the `N` units one at a time, in a new random order, each with the current state of the others. Expected: the stored patterns are fixed points for small `M/N`; at `M/N -> 0` the overlap is the solution of `m = tanh(m/T)`, so retrieval exists for `T < 1`; at `T = 0` it disappears at `alpha = M/N` of about 0.138.

## Use

```
pip install -e ../plot_style                 # style of the figures (once)
python hopfield.py --N=1000 --M=3 --replicas=10 --iter=30 --Tmin=0.05 --Tmax=1.5 --Tby=0.1 --plot --threads=4
python hopfield.py --capacity --N=500 --replicas=20 --iter=20 --plot --threads=4
python hopfield.py --finite-size --Ns=100,200,400,800,1600,3200 --replicas=40 --iter=100 --plot --threads=4   # 3 min
python hopfield.py --finite-size --from-json=figs/finite_size.json --plot                                      # refit, no simulation
python -m pytest test_hopfield.py
```

Run it from a file or the shell, not from `python -` or a notebook cell: `Pool` needs to import the main module.

`--start=pattern` (default) starts from pattern 0 with a fraction `--noise` of the bits flipped, and `m` is the overlap with it; `--start=random` starts from a random state and `m` is the largest `|m^mu|` reached. Everything is reproducible with `--seed`. The figures go to `figs/`.

## Results

Overlap and energy against `T` (`N = 1000`, `M = 3`, 10 replicas, 30 sweeps, start on pattern 0), against the mean field curve `m = tanh(m/T)`:

![m and E against T](figs/m_E_T_1000-3.png)

The simulation follows the mean field curve; the transition is rounded around `T = 1` by the finite `N` and by the crosstalk of the other patterns. Energy per unit against the sweeps:

![energy against time](figs/energy_t_1000-3.png)

Capacity at `T = 0` (`N = 500`, starting on the pattern):

![capacity](figs/capacity_500.png)

The overlap is above 0.97 up to `alpha = 0.12` and falls to 0.43 at `alpha = 0.2`. The mean field value 0.138 is for `N -> infinity`, here the drop is spread over `alpha` between 0.14 and 0.2: at `alpha = 0.17` the replicas end with overlaps between 0.34 and 1.0, at 0.2 most are between 0.3 and 0.5.

## Finite size: does the capacity tend to the mean field value?

The mean field `alpha_c = 0.1379` (with `m = 0.9674` there; solution of the PDF's Eq. 64, `mean_field_retrieval` and `mean_field_capacity`) is the limit `N -> infinity`. For finite `N` the load at which retrieval is lost depends on `N`. Scan: `T = 0`, `N = 100 ... 3200`, `alpha = 0.08 ... 0.40`, 40 random sets of patterns for every point, each started on pattern 0 and relaxed until no unit flips (at most 100 sweeps). `alpha_c(N)` is the load at which half of the runs still end with `m > 0.5` (linear interpolation, error from a bootstrap over the runs).

![finite size](figs/finite_size.png)

- **Left:** the fraction of runs that retrieve against `alpha`. The curves become steeper with `N` and move towards the dashed line at 0.138.
- **Middle:** `alpha_c(N)` = 0.327, 0.248, 0.199, 0.170, 0.1635, 0.156 for `N` = 100 ... 3200. It decreases with `N` and is still decreasing at the largest size. The exponent of the correction is not known, so I fitted `alpha_c(N) = a + b N^-nu` in several ways:

  | fit | `alpha_c(infinity)` | chi2 (4 dof) |
  |---|---|---|
  | `nu = 1/2` fixed | 0.128 +- 0.003 | 15.5 |
  | `nu = 0.67` fixed | 0.139 +- 0.002 | 8.1 |
  | `nu = 1` fixed | 0.151 +- 0.002 | 2.4 |
  | `nu` free | 0.1505 +- 0.0016 at `nu = 0.98` | 2.3 |

  The data prefer `nu` close to 1 (within one unit of chi2: `nu` between 0.86 and 1.10, `a` between 0.147 and 0.153) and disfavour `1/sqrt(N)`. The extrapolated value depends on the assumed exponent, so it cannot be read as a measurement of 0.138: with `nu = 1/2` it is below it, with the best `nu` it is about 0.012 above it. What the scan supports is that `alpha_c(N)` decreases towards a value between 0.13 and 0.15, which contains the mean field number but does not fix it. Possible biases that I did not remove: the definition of `alpha_c(N)` (half of the runs, `m > 0.5`), the limit of 100 sweeps (the relaxation is slow near the transition), and only 6 sizes up to `N = 3200`. Replica symmetry breaking is also known to shift the true `alpha_c` slightly above 0.138 (I remember about 0.144, Crisanti, Amit & Gutfreund 1986; I did not check it).
- **Right:** the overlap of the runs that retrieve, at `alpha = 0.10` and `0.12`, against the mean field overlaps (dashed, 0.9980 and 0.9932). The points agree with them within their error bars, which are large at small `N`: this is a weaker test than the capacity.
