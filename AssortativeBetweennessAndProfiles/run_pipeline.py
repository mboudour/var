#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, gzip, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- Leiden (python-igraph + leidenalg) community detection ---
def _require_leiden():
    try:
        import igraph as ig
        import leidenalg as la
        return ig, la
    except Exception as e:
        raise ImportError(
            "Leiden partitioning requires 'python-igraph' and 'leidenalg'. "
            "Install with: pip install python-igraph leidenalg"
        ) from e

def nx_to_igraph(G: nx.Graph):
    """Convert a NetworkX (Di)Graph to an igraph Graph with optional edge weights."""
    ig, _ = _require_leiden()
    nodes = list(G.nodes())
    idx = {v:i for i,v in enumerate(nodes)}
    edges = []
    weights = []
    has_w = False
    for u, v, data in G.edges(data=True):
        edges.append((idx[u], idx[v]))
        w = data.get("weight", 1.0)
        weights.append(float(w))
        if "weight" in data:
            has_w = True
    g = ig.Graph(n=len(nodes), edges=edges, directed=G.is_directed())
    if has_w:
        g.es["weight"] = weights
        return g, nodes, "weight"
    return g, nodes, None

def leiden_partition(G: nx.Graph, resolution: float = 1.0, seed: int = 0):
    """Run Leiden on G (directed or undirected). Returns (part, K)."""
    ig, la = _require_leiden()
    g, nodes, wkey = nx_to_igraph(G)
    # RBConfigurationVertexPartition supports directed modularity (Leicht-Newman) in leidenalg/igraph.
    kwargs = {"resolution_parameter": float(resolution)}
    if wkey is not None:
        kwargs["weights"] = g.es[wkey]
    # leidenalg uses its own RNG; set seed via igraph if available
    try:
        ig.set_random_number_generator(ig.RandomNumberGenerator(seed))
    except Exception:
        pass

    # Leiden cannot accept negative weights. If any edge has weight < 0, drop weights.
    has_neg_weight = False
    for _, _, edata in G.edges(data=True):
        w = edata.get("weight", 1.0)
        if w is not None and w < 0:
            has_neg_weight = True
            break

    if has_neg_weight:
        kwargs.pop("weights", None)   # run unweighted

    part_obj = la.find_partition(g, la.RBConfigurationVertexPartition, **kwargs)
    # part_obj = la.find_partition(g, la.RBConfigurationVertexPartition, **kwargs)
    membership = part_obj.membership
    part = {nodes[i]: int(membership[i]) for i in range(len(nodes))}
    K = len(set(membership))
    return part, K


HERE = Path(__file__).resolve()
COMP = HERE.parent              # computations/
ROOT = COMP.parent              # project/
MANIFEST = COMP / "manifest.json"
RAW = COMP / "data" / "raw"
OUTT = COMP / "out" / "tables"
OUTF = COMP / "out" / "figures"

def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))

def read_gz_lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            yield line

def load_txt_gz_edgelist(path: Path, directed: bool):
    G = nx.DiGraph() if directed else nx.Graph()
    for line in read_gz_lines(path):
        a=line.split()
        if len(a) < 2:
            continue
        u,v=a[0],a[1]
        if u==v: 
            continue
        if G.has_edge(u,v):
            G[u][v]["weight"]=G[u][v].get("weight",1.0)+1.0
        else:
            G.add_edge(u,v, weight=1.0)
    return G

def load_signed_csv_gz(path: Path):
    import csv
    G = nx.DiGraph()
    with gzip.open(path,"rt",encoding="utf-8",errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2: 
                continue
            u,v=row[0],row[1]
            if u==v: 
                continue
            w=float(row[2]) if len(row)>2 and row[2] else 1.0
            # t=int(float(row[3])) if len(row)>3 and row[3] else None
            t=int(float(row[3])) if len(row)>3 and row[3] else None
            G.add_edge(u,v, weight=w, time=t)
    return G

def load_epstein_csv(path: Path):
    import csv
    with open(path,"r",encoding="utf-8",errors="ignore",newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        low = [c.lower() for c in fields]
        def find(cands):
            for c in cands:
                if c in low:
                    return fields[low.index(c)]
            return None
        c_sender = find(["sender","from","from_email","fromaddress","from_addr"])
        c_rec = find(["recipient","to","to_email","toaddress","to_addr","recipient_email"])
        c_time = find(["date","datetime","time","timestamp","sent","sent_at"])
        if c_sender is None or c_rec is None:
            raise ValueError(f"Epstein CSV missing sender/recipient cols. Found: {fields}")
        G = nx.DiGraph()
        for row in reader:
            u=(row.get(c_sender) or "").strip()
            v=(row.get(c_rec) or "").strip()
            if not u or not v or u==v:
                continue
            if G.has_edge(u,v):
                G[u][v]["weight"] += 1.0
            else:
                G.add_edge(u,v, weight=1.0)
            if c_time and row.get(c_time):
                G[u][v]["time_last"] = row.get(c_time)
    return G

def load_enron_edges(path: Path):
    G = nx.DiGraph()
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            a=line.split()
            if len(a)<2: 
                continue
            u,v=a[0],a[1]
            if u==v: 
                continue
            if G.has_edge(u,v):
                G[u][v]["weight"] += 1.0
            else:
                G.add_edge(u,v, weight=1.0)
    return G

# def load_congress_zip(path: Path):
    # download_snap.py unzips congress_network.zip into RAW/congress_network/
    folder = path.parent / path.stem
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Congress folder not found: {folder}. Re-run: python computations/src/download_snap.py --all"
        )

    files = []
    for ext in ("*.txt", "*.csv", "*.tsv", "*.edges", "*.dat"):
        files += list(folder.rglob(ext))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No data files found under {folder}")

    best_edges = None
    debug = []

    for f in files:
        edges = []
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(",") if ("," in line and "	" not in line) else line.split()
            if len(parts) < 2:
                continue

            u = parts[0].strip()
            v = parts[1].strip()
            if not u or not v or u == v:
                continue

            # If a third column exists, it MUST be numeric; otherwise it's prose -> skip the line
            w = 1.0
            if len(parts) >= 3:
                try:
                    w = float(parts[2])
                except Exception:
                    continue

            edges.append((u, v, w))

        debug.append((f.name, len(edges)))
        if len(edges) >= 10:
            best_edges = edges
            break

    if best_edges is None:
        msg = "Could not parse Congress edge list. Parsed edge counts by file:\n" + "\n".join(
            [f"  {name}: {cnt}" for name, cnt in debug]
        )
        raise ValueError(msg)

    G = nx.DiGraph()
    for u, v, w in best_edges:
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G



def load_congress_zip(path: Path):
    # download_snap.py unzips congress_network.zip into:
    # RAW/congress_network/congress_network/congress.edgelist
    folder = path.parent / path.stem / "congress_network"
    edgelist = folder / "congress.edgelist"
    if not edgelist.exists():
        raise FileNotFoundError(f"Missing {edgelist}. Check unzip output.")

    G = nx.DiGraph()
    for line in edgelist.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        u = parts[0]
        v = parts[1]
        if u == v:
            continue

        # If a 3rd column exists, treat it as weight; otherwise weight=1
        w = 1.0
        if len(parts) >= 3:
            try:
                w = float(parts[2])
            except Exception:
                w = 1.0

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G



def get_partition(G, spec, labels=None, seed=0):
    """
    Partition protocol:
      - attribute-induced: use provided labels/colors (filtered to nodes in G)
      - structural: run Leiden directly on the given graph (directed or undirected)
    """
    if spec["type"]=="attribute" and labels is not None:
        return {v: labels[v] for v in G.nodes() if v in labels}

    # structural: Leiden on the actual graph (no undirected projection)
    resolution = spec.get("params", {}).get("resolution", 1.0)
    part, K = leiden_partition(G, resolution=float(resolution), seed=seed)
    return part

def interior_flags(G, part):
    if G.is_directed():
        is_internal={}
        for v in G.nodes():
            g=part.get(v,None)
            if g is None:
                is_internal[v]=False; continue
            ok=True
            for u in G.predecessors(v):
                if part.get(u)!=g: ok=False; break
            if ok:
                for u in G.successors(v):
                    if part.get(u)!=g: ok=False; break
            is_internal[v]=ok
        return is_internal
    else:
        is_internal={}
        for v in G.nodes():
            g=part.get(v,None)
            if g is None:
                is_internal[v]=False; continue
            ok=True
            for u in G.neighbors(v):
                if part.get(u)!=g: ok=False; break
            is_internal[v]=ok
        return is_internal

def edge_buckets(G, part, is_internal):
    """
    Intra-group edge/arc strata induced by interior/boundary labeling.
    Directed: I->I, I->B, B->I, B->B  (restricted to arcs with endpoints in same group)
    Undirected: II, IB, BB            (restricted to edges with endpoints in same group)
    """
    if G.is_directed():
        b={"I->I":[],"I->B":[],"B->I":[],"B->B":[]}
        for u,v in G.edges():
            if part.get(u)!=part.get(v):
                continue
            iu = bool(is_internal.get(u,False)); iv = bool(is_internal.get(v,False))
            if iu and iv: b["I->I"].append((u,v))
            elif iu and (not iv): b["I->B"].append((u,v))
            elif (not iu) and iv: b["B->I"].append((u,v))
            else: b["B->B"].append((u,v))
        return b
    else:
        b={"II":[],"IB":[],"BB":[]}
        for u,v in G.edges():
            if part.get(u)!=part.get(v):
                continue
            iu = bool(is_internal.get(u,False)); iv = bool(is_internal.get(v,False))
            if iu and iv: b["II"].append((u,v))
            elif (not iu) and (not iv): b["BB"].append((u,v))
            else: b["IB"].append((u,v))
        return b

def pearson(pairs):
    if len(pairs) < 2:
        return np.nan
    x=np.array([p[0] for p in pairs],dtype=float)
    y=np.array([p[1] for p in pairs],dtype=float)
    if np.std(x)<1e-15 or np.std(y)<1e-15:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])

def betweenness(G, seed=0):
    n=G.number_of_nodes()
    k=None
    if n>20000:
        k=300
    elif n>5000:
        k=200
    try:
        return nx.betweenness_centrality(G, normalized=True, k=k, seed=seed)
    except TypeError:
        return nx.betweenness_centrality(G, normalized=True, k=k)

def modularity_Q(G, part, gamma: float = 1.0):
    """
    Directed modularity (Leicht-Newman) for DiGraph, undirected modularity for Graph.
    Uses edge weights if present.
    For directed graphs:
      Q = (1/m) * sum_{ij} (A_ij - gamma * k_out(i) k_in(j) / m) * 1[part(i)=part(j)]
    which can be computed in O(m + K) via community totals.
    """
    if not part:
        return np.nan
    if G.is_directed():
        # weights
        m = 0.0
        kout = defaultdict(float)
        kin = defaultdict(float)
        wintra = defaultdict(float)
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))
            m += w
            kout[u] += w
            kin[v] += w
            gu = part.get(u, None)
            gv = part.get(v, None)
            if gu is not None and gu == gv:
                wintra[gu] += w
        if m <= 0:
            return np.nan
        # community out/in totals
        comm_out = defaultdict(float)
        comm_in = defaultdict(float)
        for node, g in part.items():
            comm_out[g] += kout.get(node, 0.0)
            comm_in[g] += kin.get(node, 0.0)
        Q = 0.0
        for g in set(part.values()):
            Q += wintra.get(g, 0.0) - gamma * (comm_out.get(g, 0.0) * comm_in.get(g, 0.0) / m)
        return float(Q / m)
    else:
        # undirected modularity via NetworkX quality module on the given partition
        H = G
        comms = defaultdict(list)
        for v, g in part.items():
            comms[g].append(v)
        try:
            from networkx.algorithms.community.quality import modularity
            return float(modularity(H, list(comms.values())))
        except Exception:
            return np.nan

def conductance_phi_max(G, part):
    H = G.to_undirected() if G.is_directed() else G

    H_nodes = set(H.nodes())
    comms = defaultdict(set)

    for v, g in part.items():
        if v in H_nodes:
            comms[g].add(v)

    all_nodes = H_nodes

    def vol(S):
        return sum(dict(H.degree(S)).values())

    best = 0.0
    for S in comms.values():
        if not S or S == all_nodes:
            continue

        cut = 0
        for u in S:
            for v in H.neighbors(u):
                if v not in S:
                    cut += 1

        denom = min(vol(S), vol(all_nodes - S))
        if denom <= 0:
            continue

        phi = cut / denom
        if phi > best:
            best = phi

    return best if best > 0 else np.nan



def write_latex_table(df, path, caption, label):
    def esc(s):
        s=str(s)
        s=s.replace("\\","\\textbackslash ")
        for ch,rep in [("_","\\_"),("&","\\&"),("%","\\%"),("#","\\#"),("{","\\{"),("}","\\}")]:
            s=s.replace(ch,rep)
        return s
    cols=list(df.columns)
    aligns="l"+"r"*(len(cols)-1)
    lines=[]
    lines += ["\\begin{table}[t]","\\centering",f"\\caption{{{caption}}}",f"\\label{{{label}}}",f"\\begin{{tabular}}{{{aligns}}}","\\toprule"]
    lines.append(" & ".join([esc(c) for c in cols]) + " \\\\")
    lines.append("\\midrule")
    for _,r in df.iterrows():
        out=[]
        for c in cols:
            v=r[c]
            if isinstance(v,(float,np.floating)):
                out.append("" if np.isnan(v) else f"{float(v):.3f}")
            elif isinstance(v,(int,np.integer)):
                out.append(str(int(v)))
            else:
                out.append(esc(v))
        lines.append(" & ".join(out) + " \\\\")
    lines += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")

def plot_scatter(a_map, buckets, out_png, seed=0, max_points=20000):
    rng=random.Random(seed)
    plt.figure()
    for name,edges in buckets.items():
        if not edges:
            continue
        if len(edges)>max_points:
            edges=rng.sample(edges,max_points)
        xs=[a_map[u] for u,v in edges]
        ys=[a_map[v] for u,v in edges]
        plt.scatter(xs,ys,s=6,alpha=0.35,label=name)
    plt.xlabel("a(u)")
    plt.ylabel("a(v)")
    plt.title(out_png.stem)
    plt.legend(markerscale=2)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def auc_cv(X,y,seed=0):
    y=np.asarray(y,dtype=int); X=np.asarray(X,dtype=float)
    if len(np.unique(y))<2:
        return np.nan
    npos=int(y.sum()); nneg=int((1-y).sum())
    folds=min(5,npos,nneg)
    if folds<2:
        return np.nan
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    pipe=Pipeline([("scaler",StandardScaler()),
                   ("clf",LogisticRegression(max_iter=5000,class_weight="balanced"))])
    aucs=[]
    for tr,te in cv.split(X,y):
        pipe.fit(X[tr],y[tr])
        p=pipe.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te],p))
    return float(np.mean(aucs))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()

    OUTT.mkdir(parents=True, exist_ok=True)
    OUTF.mkdir(parents=True, exist_ok=True)

    m=load_manifest()
    ds=m["datasets"][:args.limit] if args.limit else m["datasets"]

    # EU-core labels
    eu_labels=None
    labels_gz = RAW/"email-Eu-core-department-labels.txt.gz"
    if labels_gz.exists():
        eu_labels={}
        for line in read_gz_lines(labels_gz):
            u, lab = line.split()
            eu_labels[int(u)] = lab

    sum_rows=[]; prof_rows=[]; ml_rows=[]
    for d in ds:
        dsid=d["id"]
        seed=int(d.get("partition",{}).get("params",{}).get("seed",args.seed))
        print("[dataset]", dsid)

        # Load graph
        G=None
        if "raw_path" in d:
            p=(ROOT/d["raw_path"]).resolve()
            if not p.exists():
                print("  [skip] missing", p); continue
            if dsid=="epstein_emails":
                G=load_epstein_csv(p)
            elif dsid=="enron_emails_directed":
                G=load_enron_edges(p)
        else:
            url=d["snap_url"]; fname=url.split("/")[-1]
            p=RAW/fname
            if not p.exists():
                print("  [skip] download missing", fname, "-> run python computations/src/download_snap.py --all"); continue
            if fname.endswith(".txt.gz"):
                G = load_txt_gz_edgelist(p, directed=bool(d["directed"]))
                # EU-core edge list nodes are numeric; enforce int node IDs to match department labels
                if dsid == "email_Eu_core":
                    G = nx.relabel_nodes(G, lambda x: int(x))
            elif fname.endswith(".csv.gz"):
                G=load_signed_csv_gz(p)
            elif fname.endswith(".zip"):
                G=load_congress_zip(p)
        if G is None:
            print("  [skip] loader not available"); continue

        labels = eu_labels if dsid=="email_Eu_core" else None
        part=get_partition(G, d["partition"], labels=labels, seed=seed)
        is_internal=interior_flags(G, part)
        buckets=edge_buckets(G, part, is_internal)
        a=betweenness(G, seed=seed)

        # summary
        K=len(set(part.values())) if part else 0
        sum_rows.append({
            "id":dsid,"n":G.number_of_nodes(),"m":G.number_of_edges(),"directed":int(G.is_directed()),
            "partition_type":d["partition"]["type"],"partition_source":("leiden" if d["partition"]["type"]=="structural" else d["partition"]["source"]),
            "K":K,"K_leiden":(K if d["partition"]["type"]=="structural" else np.nan),"Q":modularity_Q(G,part),"phi_max":conductance_phi_max(G,part)
        })

        # profile
        if G.is_directed():
            row={"id":dsid,"n":G.number_of_nodes(),"m":G.number_of_edges(),"directed":1,"K":K}
            for tname,edges in buckets.items():
                row["r_"+tname.replace("->","to")] = pearson([(a[u],a[v]) for u,v in edges])
        else:
            row={"id":dsid,"n":G.number_of_nodes(),"m":G.number_of_edges(),"directed":0,"K":K}
            for tname,edges in buckets.items():
                row["r_"+tname] = pearson([(a[u],a[v]) for u,v in edges])
        prof_rows.append(row)

        # figure
        fig=OUTF/f"{dsid}_separated_groups.png"
        if not fig.exists():
            plot_scatter(a, buckets, fig, seed=seed)
            print("  [fig]", fig.name)

        # ML: predict top 5% betweenness
        nodes=list(G.nodes())
        bvals=np.array([a[u] for u in nodes],dtype=float)
        thr=np.quantile(bvals,0.95) if len(bvals)>=20 else np.max(bvals)
        y=(bvals>=thr).astype(int)
        boundary=np.array([0.0 if is_internal.get(u,False) else 1.0 for u in nodes],dtype=float)
        if G.is_directed():
            deg=np.log1p(np.array([G.in_degree(u)+G.out_degree(u) for u in nodes],dtype=float))
        else:
            deg=np.log1p(np.array([G.degree(u) for u in nodes],dtype=float))
        X=np.column_stack([boundary,deg])
        ml_rows.append({"id":dsid,"n":G.number_of_nodes(),"m":G.number_of_edges(),"directed":int(G.is_directed()),
                        "K":K,"AUC":auc_cv(X,y,seed=seed),"features":"boundary+logdeg"})

    df_sum=pd.DataFrame(sum_rows).sort_values(["directed","n","id"]).reset_index(drop=True)
    df_prof=pd.DataFrame(prof_rows).sort_values(["directed","n","id"]).reset_index(drop=True)
    df_ml=pd.DataFrame(ml_rows).sort_values(["directed","n","id"]).reset_index(drop=True)

    # Pretty column names for LaTeX tables
    df_sum_tex = df_sum.rename(columns={"K_leiden": "Number of Leiden communities"})

    (OUTT/"dataset_partition_summary.csv").write_text(df_sum.to_csv(index=False), encoding="utf-8")
    (OUTT/"assortativity_profiles.csv").write_text(df_prof.to_csv(index=False), encoding="utf-8")
    (OUTT/"ml_boundary_prediction.csv").write_text(df_ml.to_csv(index=False), encoding="utf-8")

    write_latex_table(df_sum_tex, OUTT/"dataset_partition_summary.tex",
        caption="Datasets and partition protocol (structural partitions are computed by the Leiden algorithm; attribute-induced partitions use provided labels/colors).",
        label="tab:datasets")
    write_latex_table(df_prof, OUTT/"assortativity_profiles.tex",
        caption="Interior--boundary assortativity profiles for attribute $a=b$ (betweenness). Directed graphs report four components ($I\to I$, $I\to B$, $B\to I$, $B\to B$); undirected graphs report three (II, IB, BB).",
        label="tab:profiles")
    write_latex_table(df_ml, OUTT/"ml_boundary_prediction.tex",
        caption="Compact ML: predicting top mediation nodes (top 5\\% betweenness) from boundary and degree features (AUC).",
        label="tab:ml")

    print("[done] wrote tables:", OUTT)
    print("[done] wrote figures:", OUTF)

if __name__=="__main__":
    main()
