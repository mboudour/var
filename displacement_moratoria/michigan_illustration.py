#!/usr/bin/env python3
"""
Michigan illustration (Section 7): data-center moratoria among cities and
townships, 2025-2026.

Inputs
  data_raw/mi_cousub_2020_500k.json  Census 2020 cartographic boundary file of
      Michigan county subdivisions (1:500k), as distributed in the Census
      Bureau's citysdk repository:
      https://raw.githubusercontent.com/uscensusbureau/citysdk/master/v2/GeoJSON/500k/2020/26/county-subdivision.json
  data_raw/sr/siting_radar_data_center_moratoria.csv  Siting Radar CSV export
      (https://sitingradar.com/data-center-moratoria.csv, downloaded 2026-11-08).
  data/mi_moratoria.csv  list compiled earlier from the Siting Radar website,
      WKAR, datacenterbans.com, the Baldori tracker and local sources.

Model objects (Sections 3-5) on the observed adjacency graph G:
  choice sets N[h], pi uniform, w_j = 1.  The displacement weights omega are
  proportional to 1/w_0^2 for every w_0, so assortativity, the volume
  fraction and the *shape* of D(S_t) do not depend on w_0; D is reported
  as the cut share  cut(S)/W.

Outputs
  results/mi_adopters.csv        adopting subdivisions, earliest dates, sources
  results/mi_timeline.csv        one row per adoption date
  tables/tab_michigan.tex        summary table
  figures/fig_mi_timeline.pdf    exposure share, assortativity, volume share
  figures/fig_mi_quotient.pdf    county quotient graph (networkx + pygraphviz)
  results/mi_summary.csv
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import shape
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20261108)


def load_units():
    d = json.load(open(os.path.join(HERE, "data_raw", "mi_cousub_2020_500k.json")))
    rows, geoms = [], []
    for f in d["features"]:
        p = f["properties"]
        kind = "city" if p["LSAD"] == "25" else "township"
        rows.append(dict(geoid=p["GEOID"], name=p["NAME"], kind=kind,
                         county=p["NAMELSADCO"].replace(" County", ""),
                         namelsad=p["NAMELSAD"]))
        geoms.append(shape(f["geometry"]).buffer(0))
    return pd.DataFrame(rows), geoms


def adjacency(geoms, tol=1e-4):
    """Rook-type adjacency: each unit is buffered by `tol` degrees (about 10 m)
    to absorb generalisation gaps of the 1:500k file, and two units are
    adjacent if the buffered polygons overlap in a strip of area > 1e-7
    square degrees (a shared boundary of roughly 50 m or more); point contacts
    are excluded."""
    buf = [g.buffer(tol) for g in geoms]
    tree = STRtree(buf)
    G = nx.Graph()
    G.add_nodes_from(range(len(geoms)))
    for i, g in enumerate(buf):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            inter = g.intersection(buf[j])
            # require an overlap strip, not a point contact
            # strip area ~ 2*tol*L; threshold corresponds to L of roughly 50 m
            if inter.area > 1e-7:
                G.add_edge(i, j)
    return G


def omega_matrix(G):
    n = G.number_of_nodes()
    rows, cols = [], []
    for h in range(n):
        for j in set(G[h]) | {h}:
            rows.append(h)
            cols.append(j)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    Om = (A.T @ A).tocsr() / n          # pi_h = 1/n, w_0 = 1 (scale-free)
    Om.setdiag(0)
    Om.eliminate_zeros()
    return Om


def binary_stats(Om, deg, x):
    volV = deg.sum()
    W = volV / 2
    cut = float(x @ (Om @ (1 - x)))
    volS = float(deg @ x)
    r = 1 - 2 * W * cut / (volS * (volV - volS)) if 0 < volS < volV else np.nan
    return cut / W, r, volS / volV


def categorical_assortativity(Om, labels):
    """Newman's weighted assortativity of a categorical attribute on H."""
    cats = sorted(set(labels))
    k = {c: i for i, c in enumerate(cats)}
    lab = np.array([k[c] for c in labels])
    C = sparse.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)),
                          shape=(len(lab), len(cats)))
    E = (C.T @ Om @ C).toarray()
    E = E / E.sum()
    a = E.sum(axis=1)
    return float((np.trace(E) - a @ a) / (1 - a @ a)), E, cats



def build_adopters(units):
    """One row per county subdivision that adopted a data-center moratorium,
    with the earliest known adoption date.

    Primary source: Siting Radar export (data_raw/sr/...csv), rows with
    state MI, action 'moratorium' and stage 'adopted' or 'extended'
    (an extension dates an adoption from above). Townships carry county-
    subdivision GEOIDs; cities carry place GEOIDs and are mapped to the
    county subdivision(s) with the same 5-digit code. Rows with stage
    'rescinded' and no adoption row are not used. Villages and counties are
    not county subdivisions and are dropped.
    Secondary source: data/mi_moratoria.csv (compiled earlier from the Siting
    Radar website, WKAR, datacenterbans.com, the Baldori tracker and local
    sources), matched by name, kind and county."""
    sr = pd.read_csv(os.path.join(HERE, "data_raw", "sr",
                                  "siting_radar_data_center_moratoria.csv"))
    sr = sr[(sr.state == "MI") & (sr.action == "moratorium") &
            sr.stage.isin(["adopted", "extended"])].copy()
    code = units.geoid.str[-5:]
    rows = []
    for _, r in sr.iterrows():
        g = str(int(r.census_geoid)).zfill(10) if pd.notna(r.census_geoid) else ""
        kind = r.jurisdiction_kind
        if kind in ("township", "charter_township"):
            hits = units.geoid[units.geoid == g].tolist()
        elif kind == "city":
            hits = units.geoid[(code == g[-5:]) & (units.kind == "city")].tolist()
        else:
            hits = []
        for h in hits:
            rows.append(dict(geoid=h, date=r.event_date, stage=r.stage,
                             source="SitingRadar"))
    tr = pd.read_csv(os.path.join(HERE, "data", "mi_moratoria.csv"))
    for _, r in tr.iterrows():
        hit = units[(units.name == r["name"]) & (units.kind == r["kind"]) &
                    (units.county == r["county"])]
        if len(hit) == 1:
            rows.append(dict(geoid=hit.geoid.iloc[0], date=r["date"],
                             stage="adopted", source="trackers"))
    a = pd.DataFrame(rows)
    a["date"] = pd.to_datetime(a["date"])
    src = a.groupby("geoid").source.agg(lambda x: "+".join(sorted(set(x))))
    ext_only = a.groupby("geoid").stage.agg(lambda x: set(x) == {"extended"})
    a = a.sort_values(["date", "geoid"]).drop_duplicates("geoid")
    a["sources"] = a.geoid.map(src)
    a["dated_by_extension"] = a.geoid.map(ext_only)
    a = a.merge(units[["geoid", "name", "kind", "county"]], on="geoid")
    return a[["geoid", "name", "kind", "county", "date", "sources",
              "dated_by_extension"]].sort_values(["date", "geoid"]).reset_index(drop=True)


def main():
    for d in ["results", "tables", "figures"]:
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    units, geoms = load_units()
    G = adjacency(geoms)
    n = G.number_of_nodes()
    Om = omega_matrix(G)
    deg = np.asarray(Om.sum(axis=1)).ravel()

    # ---- adopters: Siting Radar export (primary) + compiled trackers
    adopt = build_adopters(units)
    adopt.to_csv(os.path.join(HERE, "results", "mi_adopters.csv"), index=False)
    first = pd.Series(pd.to_datetime(adopt.date).values, index=adopt.geoid.values)
    mor = adopt
    gid2i = {g: i for i, g in enumerate(units.geoid)}

    # ---- timeline by adoption date
    dates = sorted(first.unique())
    x = np.zeros(n)
    tl = []
    prev_cut = 0.0
    increments, weight_frac = [], []
    for dte in dates:
        new = [gid2i[g] for g in first[first == dte].index]
        for i in new:
            # Lemma 3 increment for single additions (order within a date arbitrary)
            w_to_S = float((Om[i] @ x).item())
            increments.append(deg[i] - 2 * w_to_S)
            weight_frac.append((w_to_S / deg[i], units.name[i] + " " + units.kind[i]))
            x[i] = 1.0
        cs, r, vs = binary_stats(Om, deg, x)
        # permutation baseline for r: random sets of the same size
        k = int(x.sum())
        rb = []
        for _ in range(200):
            y = np.zeros(n)
            y[RNG.choice(n, k, replace=False)] = 1.0
            rb.append(binary_stats(Om, deg, y)[1])
        tl.append(dict(date=pd.Timestamp(dte).date(), adopters=k, cut_share=cs,
                       r=r, r_null_mean=float(np.mean(rb)),
                       r_null_q975=float(np.quantile(rb, .975)), vol_share=vs))
    tl = pd.DataFrame(tl)
    tl.to_csv(os.path.join(HERE, "results", "mi_timeline.csv"), index=False)
    inc = np.array(increments)

    # ---- county assortativity and quotient graph
    r_cty, E, cats = categorical_assortativity(Om, list(units.county))
    adopt_share = units.assign(adopt=x).groupby("county")["adopt"].mean()
    size = units.groupby("county").size()
    Q = nx.Graph()
    for c in cats:
        Q.add_node(c, size=int(size[c]), share=float(adopt_share[c]))
    for i, c1 in enumerate(cats):
        for j in range(i + 1, len(cats)):
            if E[i, j] > 0:
                Q.add_edge(c1, cats[j], weight=float(E[i, j]))
    # geographic anchor: county centroid (mean of unit centroids)
    cen = {}
    for c in cats:
        pts = np.array([[geoms[i].centroid.x, geoms[i].centroid.y]
                        for i in np.flatnonzero(units.county == c)])
        cen[c] = pts.mean(axis=0)
    A = nx.nx_agraph.to_agraph(Q)
    for c in cats:
        nd = A.get_node(c)
        nd.attr["pos"] = f"{cen[c][0]*40:.2f},{cen[c][1]*40:.2f}!"
    A.layout(prog="neato", args="-n")
    pos = {c: tuple(map(float, A.get_node(c).attr["pos"].rstrip("!").split(",")))
           for c in cats}
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    wmax = max(d["weight"] for *_, d in Q.edges(data=True))
    nx.draw_networkx_edges(Q, pos, ax=ax, edge_color="0.6",
                           width=[0.3 + 2.5 * d["weight"] / wmax
                                  for *_, d in Q.edges(data=True)])
    sizes = [12 * Q.nodes[c]["size"] for c in cats]
    shares = [Q.nodes[c]["share"] for c in cats]
    nodes = nx.draw_networkx_nodes(Q, pos, ax=ax, node_size=sizes,
                                   node_color=shares, cmap="Greys", vmin=0,
                                   vmax=max(shares), edgecolors="black",
                                   linewidths=0.5)
    for c in cats:   # all labels directly below their node
        ax.annotate(c, pos[c], xytext=(0, -7), textcoords="offset points",
                    ha="center", va="top", fontsize=4.5)
    cb = fig.colorbar(nodes, ax=ax, shrink=0.5)
    cb.set_label("share of jurisdictions with a moratorium", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_mi_quotient.pdf"))
    plt.close(fig)

    # ---- timeline figure
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))
    t = pd.to_datetime(tl.date)
    axes[0].step(t, tl.cut_share, where="post", color="black")
    axes[0].set_title(r"exposure share $D(S_t)/W$", fontsize=9)
    axes[1].step(t, tl.r, where="post", color="black", label="observed")
    axes[1].step(t, tl.r_null_q975, where="post", color="0.5", ls="--",
                 label="random sets, 97.5%")
    axes[1].set_title(r"adoption assortativity $r_\omega(S_t)$", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].step(t, tl.vol_share, where="post", color="black")
    axes[2].set_title(r"volume share $\mathrm{vol}(S_t)/2W$", fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_mi_timeline.pdf"))
    plt.close(fig)

    # ---- summary and table
    comps = nx.number_connected_components(G)
    degs = np.array([d for _, d in G.degree()])
    S = x.astype(bool)
    Sg = G.subgraph(np.flatnonzero(S))
    summ = dict(units=n, edges=G.number_of_edges(), components=comps,
                mean_degree=float(degs.mean()), isolates=int((degs == 0).sum()),
                from_sitingradar_only=int((adopt.sources == "SitingRadar").sum()),
                from_trackers_only=int((adopt.sources == "trackers").sum()),
                from_both=int((adopt.sources == "SitingRadar+trackers").sum()),
                dated_by_extension=int(adopt.dated_by_extension.sum()),
                adopters=int(S.sum()),
                adopter_components=nx.number_connected_components(Sg),
                largest_adopter_component=max(len(c) for c in nx.connected_components(Sg)),
                final_cut_share=float(tl.cut_share.iloc[-1]),
                final_r=float(tl.r.iloc[-1]),
                final_r_null_mean=float(tl.r_null_mean.iloc[-1]),
                final_r_null_q975=float(tl.r_null_q975.iloc[-1]),
                final_vol_share=float(tl.vol_share.iloc[-1]),
                adoptions_lowering_exposure=int((inc < -1e-12).sum()),
                adoptions_unchanged_exposure=int((abs(inc) <= 1e-12).sum()),
                adoptions_total=len(inc),
                max_weight_frac=max(weight_frac)[0],
                max_weight_frac_unit=max(weight_frac)[1],
                r_county=r_cty, first_date=str(tl.date.iloc[0]),
                last_date=str(tl.date.iloc[-1]))
    pd.Series(summ).to_csv(os.path.join(HERE, "results", "mi_summary.csv"), header=False)
    lines = [r"\begin{tabular}{lr}", r"\toprule",
             f"Cities and townships (nodes of $G$) & {n} \\\\",
             f"Adjacencies (edges of $G$); mean degree & {G.number_of_edges()}; {degs.mean():.2f} \\\\",
             f"Subdivisions with a recorded moratorium & {summ['adopters']} \\\\",
             f"Connected groups of adopters in $G$; largest & {summ['adopter_components']}; {summ['largest_adopter_component']} \\\\",
             f"Volume share $\\vol_\\omega(S)/2W$ at {summ['last_date']} & {summ['final_vol_share']:.3f} \\\\",
             f"First-order exposure share $D(S)/W$ & {summ['final_cut_share']:.3f} \\\\",
             f"Adoption assortativity $r_\\omega(S)$ & {summ['final_r']:.3f} \\\\",
             f"\\quad random sets of equal size: mean; 97.5\\% quantile & {summ['final_r_null_mean']:.3f}; {summ['final_r_null_q975']:.3f} \\\\",
             f"Adoptions that lowered first-order exposure; left it unchanged (Lemma~\\ref{{lem:increment}}) & {summ['adoptions_lowering_exposure']}; {summ['adoptions_unchanged_exposure']} of {summ['adoptions_total']} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "tables", "tab_michigan.tex"), "w").write("\n".join(lines) + "\n")
    for k, v in summ.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
