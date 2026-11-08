#!/usr/bin/env python3
"""
Computations for "Strategic Displacement and Moratorium Contagion on
Jurisdictional Networks" (synthetic experiments, Sections 5-6).

Experiments
  E0  numerical checks of the exact statements:
        Proposition 3 (bilinear displacement bounds, relative to the empty set),
        Lemma 1/2 numbering as in the paper: first-order increment
        dD = d_j - 2 omega(j,S) and exact increment dE = phi_j - (p_j(S)-p_j(0)),
        Theorem 2(b) (exact part) and Proposition 6 (peak location).
  E1  exposure along moratorium cascades on planar-like graphs
        -> tables/tab_exposure.tex, figures/fig_exposure_paths.pdf
  E2  confinement and crossing with two jurisdiction types
        -> tables/tab_confinement.tex
  E3  increasing differences with spillover costs (Theorem 1)
        -> results/e3_spillover.csv

Model: choice sets C (rows of A) with weights pi_C; logit weights w_j = 1,
outside option w_0.  A jurisdiction j adopts iff p_j(a) > theta_j, where
theta_j = k_j/(c_j - b_j) > 0 (complements), and theta_j = 0 for seeds.

Usage:  python3 paper_computations.py --out .   (about 2-3 minutes)
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay


# ------------------------------------------------------------------ graphs
def make_graph(kind, n, rng):
    if kind == "grid":
        side = int(round(np.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
    elif kind == "geometric":
        pos = {i: rng.random(2) for i in range(n)}
        radius = np.sqrt(2.2 / (np.pi * n)) * 1.6
        G = nx.random_geometric_graph(n, radius, pos=pos,
                                      seed=int(rng.integers(1e9)))
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    elif kind == "delaunay":
        pts = rng.random((n, 2))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for s in Delaunay(pts).simplices:
            for u, v in itertools.combinations(s, 2):
                G.add_edge(int(u), int(v))
    else:
        raise ValueError(kind)
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def neighbourhood_choice_sets(G):
    n = G.number_of_nodes()
    sets = [sorted(set(G[h]) | {h}) for h in range(n)]
    return sets, np.full(n, 1.0 / n)


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, n, choice_sets, pi, W0):
        self.n, self.W0 = n, W0
        rows, cols = [], []
        for c, C in enumerate(choice_sets):
            rows += [c] * len(C)
            cols += list(C)
        self.A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                   shape=(len(choice_sets), n))
        self.AT = self.A.T.tocsr()
        self.pi = np.asarray(pi, float)
        self.eps = float(self.A.sum(axis=1).max()) / W0
        Om = (self.AT @ sparse.diags(self.pi) @ self.A).tocsr()
        Om.setdiag(0)
        Om.eliminate_zeros()
        self.Om = Om / W0 ** 2
        self.deg = np.asarray(self.Om.sum(axis=1)).ravel()
        self.p0 = self.p_open(np.zeros(n, bool))

    def p_open(self, closed):
        """Arrival probability of each j if j is (or were) open."""
        S = self.A @ (~closed).astype(float)
        now = self.AT @ (self.pi / (self.W0 + S))
        reopen = self.AT @ (self.pi / (self.W0 + S + 1.0))
        return np.where(closed, reopen, now)

    def cut(self, closed):
        x = closed.astype(float)
        return float(x @ (self.Om @ (1 - x)))

    def omega_to(self, closed):
        """omega(j, S) for all j."""
        return self.Om @ closed.astype(float)


def assortativity_binary(model, closed):
    volS = float(model.deg @ closed)
    volV = float(model.deg.sum())
    if not 0 < volS < volV:
        return np.nan
    W = volV / 2
    return 1 - 2 * W * model.cut(closed) / (volS * (volV - volS))


def cascade(model, theta, seeds, rng, frozen=None):
    """Asynchronous best response from the seed set; returns list of closed
    vectors and the adopter order.  `frozen` nodes are never allowed to adopt
    (used for the restricted game in E2)."""
    closed = np.zeros(model.n, bool)
    closed[list(seeds)] = True
    states, order = [closed.copy()], []
    while True:
        want = ~closed & (model.p_open(closed) > theta)
        if frozen is not None:
            want &= ~frozen
        idx = np.flatnonzero(want)
        if len(idx) == 0:
            return states, order
        j = idx[rng.integers(len(idx))]
        closed[j] = True
        states.append(closed.copy())
        order.append(j)


# ------------------------------------------------------------------ E0 checks
def check_prop2(model, rng, samples=200):
    """Proposition 2: omega(j,S)/(1+eps)^2 <= p_j(S) - p_j(empty) <= omega(j,S)."""
    worst_lo, worst_hi = np.inf, -np.inf
    for _ in range(samples):
        closed = rng.random(model.n) < rng.random()
        j = rng.integers(model.n)
        closed[j] = False
        om = model.omega_to(closed)[j]
        if om == 0:
            continue
        ratio = (model.p_open(closed)[j] - model.p0[j]) / om
        worst_lo, worst_hi = min(worst_lo, ratio), max(worst_hi, ratio)
    return worst_lo, worst_hi, 1 / (1 + model.eps) ** 2


def exact_exposure(model, closed):
    p = model.p_open(closed)
    return float(np.sum((p - model.p0)[~closed]))


def diverted(model, closed, j):
    """Arrival probability of open j that its adoption diverts to other open
    jurisdictions: sum_C pi_C w_j/(w0+s_C) * (s_C - w_j)/(w0 + s_C - w_j)."""
    s = model.A @ (~closed).astype(float)
    cols = model.AT[j].indices
    sc = s[cols]
    return float(np.sum(model.pi[cols] / (model.W0 + sc) * (sc - 1.0) / (model.W0 + sc - 1.0)))


def path_diagnostics(model, states, order, theta=None):
    D = np.array([model.cut(s) for s in states])
    E = np.array([exact_exposure(model, s) for s in states])
    pbar = model.AT @ (model.pi / (model.W0 + 1.0))       # p_j with all others closed
    exact_err, disagree, viol_exact, covered = 0.0, 0, 0, 0
    for t, j in enumerate(order):
        pj = model.p_open(states[t])[j]
        pred = diverted(model, states[t], j) - (pj - model.p0[j])
        exact_err = max(exact_err, abs((E[t + 1] - E[t]) - pred))
        dD, dE = D[t + 1] - D[t], E[t + 1] - E[t]
        if abs(dD) > 1e-15 and abs(dE) > 1e-15 and np.sign(dD) != np.sign(dE):
            disagree += 1
        if theta is not None:
            # Theorem 2(b), exact part: theta_j - p_j(empty) >= (1+eps)/(2+eps) (pbar_j - p_j(empty))
            if theta[j] - model.p0[j] >= (1 + model.eps) / (2 + model.eps) * (pbar[j] - model.p0[j]):
                covered += 1
                if dE >= 0:
                    viol_exact += 1
    v = np.array([model.deg @ s for s in states])
    W = model.deg.sum() / 2
    # Lemma 2 identity
    err = 0.0
    for t, j in enumerate(order):
        pred = model.deg[j] - 2 * model.omega_to(states[t])[j]
        err = max(err, abs((D[t + 1] - D[t]) - pred))
    # Proposition 4 bound (only if the path crosses W)
    prop4 = np.nan
    tol = 1e-9 * 2 * W
    interior = (v > tol) & (v < 2 * W - tol)
    if v[0] <= W <= v[-1] and interior.any():
        q = np.where(interior, D * 2 * W / np.where(interior, v * (2 * W - v), 1), np.nan)
        rho = np.nanmin(q) / np.nanmax(q)
        dmax = model.deg.max()
        bound = W * np.sqrt(1 - rho + rho * dmax ** 2 / (4 * W ** 2))
        tstar = int(np.argmax(D))
        prop4 = abs(v[tstar] - W) / bound          # must be <= 1
        return D, v, W, err, prop4, rho, bound / W, exact_err, disagree, viol_exact, covered
    return D, v, W, err, prop4, np.nan, np.nan, exact_err, disagree, viol_exact, covered


# ------------------------------------------------------------------ E1
def experiment_exposure(rng, out, n=225, runs=40):
    rows, examples, checks = [], {}, []
    for kind in ["grid", "geometric", "delaunay"]:
        for W0 in [2.0, 10.0, 50.0]:
            for lam in [0.05, 0.1, 0.2, 0.4]:
                for rep in range(runs):
                    G = make_graph(kind, n, rng)
                    M = Model(G.number_of_nodes(), *neighbourhood_choice_sets(G), W0)
                    theta = M.p0 * (1 + lam * rng.random(M.n))
                    seeds = rng.choice(M.n, 3, replace=False)
                    theta[seeds] = 0.0
                    states, order = cascade(M, theta, seeds, rng)
                    (D, v, W, err, prop4, rho, bnd, exerr, disagree,
                     viol_exact, covered) = path_diagnostics(M, states, order, theta)
                    size = np.array([s.sum() for s in states])
                    tD = int(np.argmax(D))
                    complete = bool(states[-1].all())
                    rows.append(dict(graph=kind, W0=W0, lam=lam, eps=M.eps,
                                     complete=complete, steps=len(order),
                                     final_frac=size[-1] / M.n,
                                     peak_frac=size[tD] / M.n,
                                     peak_vol_frac=v[tD] / (2 * W),
                                     nonmono=(0 < tD < len(D) - 1) and D[-1] < D[tD],
                                     lemma2_err=err, prop4_ratio=prop4,
                                     prop4_rho=rho, prop4_bound_over_W=bnd,
                                     exact_incr_err=exerr, sign_disagree=disagree,
                                     viol_exact_lowering=viol_exact,
                                     covered_exact=covered,
                                     adoptions=len(order)))
                    key = (kind, "complete" if complete else "partial")
                    if key not in examples and len(order) > 20:
                        examples[key] = (size / M.n, D / D.max())
                    if rep == 0:
                        lo, hi, thr = check_prop2(M, rng)
                        checks.append(dict(graph=kind, W0=W0, eps=M.eps,
                                           min_ratio=lo, max_ratio=hi, lower=thr))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "results", "e1_runs.csv"), index=False)
    ck = pd.DataFrame(checks)
    ck.to_csv(os.path.join(out, "results", "e0_prop2_checks.csv"), index=False)

    # table
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Graph & $w_0$ & $\varepsilon$ & Complete & Peak (median) & Partial & Nonmonotone \\",
             r" & & & cascades & adopter share & cascades & among partial \\",
             r"\midrule"]
    names = {"grid": "Square grid", "geometric": "Random geometric",
             "delaunay": "Delaunay"}
    for (kind, W0), g in df.groupby(["graph", "W0"], sort=False):
        comp = g[g.complete]
        part = g[~g.complete & (g.steps > 0)]
        pk = f"{comp.peak_frac.median():.2f}" if len(comp) else "--"
        nm = f"{part.nonmono.mean():.2f}" if len(part) else "--"
        lines.append(f"{names[kind]} & {W0:g} & {g.eps.median():.2f} & "
                     f"{len(comp)}/{len(g)} & {pk} & {len(part)} & {nm} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "tab_exposure.tex"), "w").write("\n".join(lines) + "\n")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, status in zip(axes, ["complete", "partial"]):
        for kind, ls in [("grid", "-"), ("geometric", "--"), ("delaunay", ":")]:
            if (kind, status) in examples:
                x, y = examples[(kind, status)]
                ax.plot(x, y, ls, color="black", lw=1.2, label=names[kind])
        ax.set_xlabel("share of jurisdictions with a moratorium")
        ax.set_title(f"{status} cascade", fontsize=10)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel(r"exposure $D(S_t)/\max_t D(S_t)$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig_exposure_paths.pdf"))
    plt.close(fig)

    summary = dict(
        runs=len(df), complete=int(df.complete.sum()),
        partial=int((~df.complete & (df.steps > 0)).sum()),
        partial_nonmono=int(df[~df.complete & (df.steps > 0)].nonmono.sum()),
        peak_median=float(df[df.complete].peak_frac.median()),
        peak_q25=float(df[df.complete].peak_frac.quantile(.25)),
        peak_q75=float(df[df.complete].peak_frac.quantile(.75)),
        peakvol_median=float(df[df.complete].peak_vol_frac.median()),
        peakvol_q25=float(df[df.complete].peak_vol_frac.quantile(.25)),
        peakvol_q75=float(df[df.complete].peak_vol_frac.quantile(.75)),
        prop4_paths=int(df.prop4_ratio.notna().sum()),
        prop4_rho_median=float(df.prop4_rho.median()),
        prop4_bound_over_W_median=float(df.prop4_bound_over_W.median()),
        never_moved=int((df.steps == 0).sum()),
        lemma2_maxerr=float(df.lemma2_err.max()),
        exact_incr_maxerr=float(df.exact_incr_err.max()),
        sign_disagreements=int(df.sign_disagree.sum()),
        adoptions_total=int(df.adoptions.sum()),
        exact_lowering_violations=int(df.viol_exact_lowering.sum()),
        exact_lowering_covered=int(df.covered_exact.sum()),
        prop4_max_ratio=float(df.prop4_ratio.max()),
        prop2_ok=bool(((ck.min_ratio >= ck.lower - 1e-12) & (ck.max_ratio <= 1 + 1e-12)).all()))
    return summary


# ------------------------------------------------------------------ E2
def experiment_confinement(rng, out, rows_=15, cols_=30, runs=60):
    """Two types: left half V1, right half V2 of a rows x cols grid.  For home
    h the developer uses N[h] restricted to h's type with probability 1-kappa
    and the full N[h] with probability kappa."""
    G = nx.grid_2d_graph(rows_, cols_)
    nodes = sorted(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    G = nx.relabel_nodes(G, idx)
    n = G.number_of_nodes()
    typ = np.array([0 if nodes[i][1] < cols_ // 2 else 1 for i in range(n)])
    V1 = typ == 0
    rec = []
    for kappa in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        sets, pis = [], []
        for h in range(n):
            full = sorted(set(G[h]) | {h})
            same = [j for j in full if typ[j] == typ[h]]
            if kappa < 1:
                sets.append(same); pis.append((1 - kappa) / n)
            if kappa > 0:
                sets.append(full); pis.append(kappa / n)
        M = Model(n, sets, pis, W0=10.0)
        rV1 = assortativity_binary(M, V1)
        for lam in [0.05, 0.1, 0.2]:
            for _ in range(runs):
                theta = M.p0 * (1 + lam * rng.random(n))
                left = np.flatnonzero(V1)
                seeds = rng.choice(left, 3, replace=False)
                theta[seeds] = 0.0
                S0 = np.zeros(n, bool); S0[seeds] = True
                pS0 = M.p_open(S0)
                margin = theta - pS0                      # m_j = theta_j - p_j(S_0)
                margin[seeds] = 0.0
                # Corollary 2(a), stronger form: omega(j, V1) <= m_j off V1
                om_V1 = M.omega_to(V1)
                confine = bool(np.all(om_V1[~V1] <= margin[~V1]))
                # restricted game: outsiders held open -> S*
                st, _ = cascade(M, theta, seeds, rng, frozen=~V1)
                Sstar = st[-1]
                om_S = M.omega_to(Sstar & ~S0)            # omega(j, S* minus S_0)
                crossing = bool(np.any(om_S[~V1] > (1 + M.eps) ** 2 * margin[~V1]))
                # full game
                st_full, _ = cascade(M, theta, seeds, rng)
                crossed = bool(st_full[-1][~V1].any())
                rec.append(dict(kappa=kappa, lam=lam, eps=M.eps, r_type=rV1,
                                Sstar_frac=Sstar[V1].mean(),
                                confine_cond=confine, cross_cond=crossing,
                                crossed=crossed,
                                final_V2_frac=st_full[-1][~V1].mean()))
    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(out, "results", "e2_runs.csv"), index=False)
    # consistency with Theorem 2
    viol_a = int((df.confine_cond & df.crossed).sum())
    viol_b = int((df.cross_cond & ~df.crossed).sum())
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$\kappa$ & $r_\omega(V_1)$ & $\lambda$ & Confinement & Crossing & Crossed & Share of $V_2$ \\",
             r" & & & condition & condition & & adopting \\",
             r"\midrule"]
    for (kappa, lam), g in df.groupby(["kappa", "lam"]):
        lines.append(f"{kappa:g} & {g.r_type.iloc[0]:.3f} & {lam:g} & "
                     f"{g.confine_cond.mean():.2f} & {g.cross_cond.mean():.2f} & "
                     f"{g.crossed.mean():.2f} & {g.final_V2_frac.mean():.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "tab_confinement.tex"), "w").write("\n".join(lines) + "\n")
    return dict(runs=len(df), viol_confine=viol_a, viol_cross=viol_b,
                undecided=int((~df.confine_cond & ~df.cross_cond).sum()))


# ------------------------------------------------------------------ E3
def experiment_spillover(rng, out, trials=400, samples=60):
    """Theorem 1 (spillover costs): increasing differences hold when
    c_j - b_j >= 2 eps (1+eps)^2 max_l gamma_jl.  Random small Delaunay
    graphs; gamma_jl ~ U[0,1] on pairs sharing a choice set.  For each trial
    draw c_j - b_j either above the bound ('cond') or a small fraction of it
    ('small'), and test Delta_j(S+i) - Delta_j(S) >= 0 on random (S,i,j)."""
    res = []
    for _ in range(trials):
        n = 14
        G = make_graph("delaunay", n, rng)
        W0 = float(rng.choice([2.0, 5.0, 20.0]))
        M = Model(n, *neighbourhood_choice_sets(G), W0)
        share = (M.Om.toarray() > 0)
        gam = rng.random((n, n)) * share
        gbar = gam.max(axis=1)
        bound = 2 * M.eps * (1 + M.eps) ** 2 * gbar
        for regime in ["cond", "small"]:
            if regime == "cond":
                cb = bound * (1 + rng.random(n))
            else:
                cb = bound * 0.02 * rng.random(n) + 1e-12
            k = np.zeros(n)

            def Delta(closed, j):
                c0 = closed.copy(); c0[j] = False
                c1 = closed.copy(); c1[j] = True
                p0, p1 = M.p_open(c0), M.p_open(c1)
                open_l = ~c1
                spill = np.sum(gam[j, open_l] * (p1[open_l] - p0[open_l]))
                return cb[j] * p0[j] - k[j] - spill

            neg = 0
            for _ in range(samples):
                closed = rng.random(n) < rng.random()
                j, i = rng.choice(n, 2, replace=False)
                closed[j] = closed[i] = False
                d0 = Delta(closed, j)
                closed[i] = True
                d1 = Delta(closed, j)
                if d1 - d0 < -1e-14:
                    neg += 1
            res.append(dict(regime=regime, W0=W0, eps=M.eps, neg=neg, samples=samples))
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(out, "results", "e3_spillover.csv"), index=False)
    g = df.groupby("regime").agg(neg=("neg", "sum"), samples=("samples", "sum"),
                                 trials_with_neg=("neg", lambda x: int((x > 0).sum())),
                                 trials=("neg", "size"))
    return {f"{r}_{c}": int(g.loc[r, c]) for r in g.index for c in g.columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    for d in ["results", "tables", "figures"]:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    s1 = experiment_exposure(rng, a.out)
    s2 = experiment_confinement(rng, a.out)
    s3 = experiment_spillover(rng, a.out)
    summary = {**{f"E1_{k}": v for k, v in s1.items()}, **{f"E2_{k}": v for k, v in s2.items()},
               **{f"E3_{k}": v for k, v in s3.items()}}
    pd.Series(summary).to_csv(os.path.join(a.out, "results", "summary.csv"), header=False)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
