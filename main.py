"""
Bovine SNP Platform v5.0 - Etape 2
Chargement donnees + QC + LD Pruning + Structure (PCA/MDS)
"""
import gzip
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import MDS as SklearnMDS

st.set_page_config(
    page_title="Bovine SNP Platform v5.0",
    page_icon="DNA",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_THRESHOLDS = {
    "geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6, "het_sd": 3.0,
    "king_cutoff": 0.354, "ld_r2": 0.2,
}

HWE_EXACT_MAX_SNP = 20000

ARS_UCD12_LENGTHS = {
    "1": 158534110, "2": 136231102, "3": 121005158, "4": 120000166,
    "5": 120089699, "6": 117806340, "7": 110682743, "8": 113384748,
    "9": 105708134, "10": 103308737, "11": 107310763, "12": 91163125,
    "13": 84246514, "14": 84648346, "15": 85207080, "16": 81726628,
    "17": 75176999, "18": 66059976, "19": 64089169, "20": 72042983,
    "21": 71599096, "22": 61416492, "23": 52531573, "24": 62384193,
    "25": 42959610, "26": 51680158, "27": 46772073, "28": 46333854,
    "29": 51319414, "X": 139009144, "Y": 50927933, "MT": 16338,
}
BOVINE_AUTOSOMES = [str(i) for i in range(1, 30)]


def _hash_ndarray(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return "empty"
    return (str(x.shape) + "|" + str(x.dtype) + "|" +
            str(float(np.nansum(x))))


HASH_FUNCS = {np.ndarray: _hash_ndarray}


def cache_data(func=None, **kw):
    def _decorate(f):
        return st.cache_data(show_spinner=False,
                             hash_funcs=HASH_FUNCS, **kw)(f)
    if func is None:
        return _decorate
    return _decorate(func)


def _autosome_weights():
    lens = np.array([ARS_UCD12_LENGTHS[c] for c in BOVINE_AUTOSOMES],
                    dtype=float)
    return lens / lens.sum()


def _chr_sort_key(chrom):
    s = str(chrom).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit():
        return (0, int(s), "")
    special = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    if s in special:
        return (1, special[s], "")
    return (2, 0, s)


def _detect_encoding(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return "gzip"
    if len(raw) >= 2 and raw[:2] == b"\x6c\x1b":
        return "plink_binary"
    return "text"


def step_header(numero, titre, ce_qu_on_fait, pourquoi, attendu):
    with st.expander("Etape " + str(numero) + " - " + titre, expanded=False):
        st.markdown("**Ce qu'on fait :** " + ce_qu_on_fait)
        st.markdown("**Pourquoi :** " + pourquoi)
        st.markdown("**Ce qu'on attend :** " + attendu)


def extract_fid_from_iid(iid):
    for sep in ("_", "-", "."):
        if sep in iid:
            return iid.split(sep)[0]
    return iid


def reconstruct_fid_from_iid(ind_df):
    df = ind_df.copy()
    unique_fids = df["FID"].nunique() if "FID" in df.columns else 0
    if unique_fids <= 1:
        df["FID"] = df["IID"].astype(str).apply(extract_fid_from_iid)
        st.info("FID reconstruit depuis IID : " +
                str(df["FID"].nunique()) + " races.")
    return df


def parse_map(map_bytes):
    enc = _detect_encoding(map_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(map_bytes).decode("utf-8",
                                                     errors="replace")
        except Exception as e:
            raise ValueError("Fichier .map.gz corrompu : " + str(e))
    elif enc == "plink_binary":
        raise ValueError("Le fichier .map est en binaire PLINK.")
    else:
        text = map_bytes.decode("utf-8", errors="replace")

    rows, rejected = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            rejected += 1
            continue
        try:
            cm = float(parts[2]) if parts[2] not in ("0", ".", "NA") else 0.0
        except ValueError:
            cm = 0.0
        try:
            bp = int(float(parts[3]))
        except (ValueError, TypeError):
            rejected += 1
            continue
        rows.append({"CHR": str(parts[0]), "SNP": parts[1],
                     "CM": cm, "BP": bp})

    if not rows:
        raise ValueError("Fichier .map vide ou invalide.")
    return pd.DataFrame(rows), rejected


def _diagnose_ped_first_line(line, n_snp_map):
    parts = line.split()
    n_cols = len(parts)
    return {
        "n_cols": n_cols,
        "expected_cols": 6 + 2 * n_snp_map,
        "n_snp_inferred": max(0, (n_cols - 6) // 2),
    }


def parse_ped(ped_bytes, n_snp_map):
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(ped_bytes).decode("utf-8",
                                                     errors="replace")
        except Exception as e:
            raise ValueError("Fichier .ped.gz corrompu : " + str(e))
    elif enc == "plink_binary":
        raise ValueError("Ce fichier est un PLINK binaire renomme.")
    else:
        text = ped_bytes.decode("utf-8", errors="replace")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Fichier .ped vide.")

    diag = _diagnose_ped_first_line(lines[0], n_snp_map)
    n_cols_first = diag["n_cols"]
    n_snp_ped = diag["n_snp_inferred"]

    if n_cols_first < 7:
        raise ValueError("Format PED invalide : " + str(n_cols_first) +
                         " colonnes.")

    if 0 < n_snp_ped < 10 and len(lines) > 100:
        raise ValueError("Fichier probablement transpose.")

    if n_snp_ped != n_snp_map and n_snp_ped > 0:
        st.warning("Desalignement PED/MAP : " + str(n_snp_ped) +
                   " vs " + str(n_snp_map) + " SNPs.")
        n_snp_eff = min(n_snp_map, n_snp_ped)
    else:
        n_snp_eff = n_snp_map

    expected_cols_eff = 6 + 2 * n_snp_eff
    fids, iids, geno_rows = [], [], []
    rejected_short, rejected_empty = 0, 0

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            rejected_empty += 1
            continue
        if len(parts) < expected_cols_eff:
            rejected_short += 1
            continue
        fids.append(parts[0])
        iids.append(parts[1])
        geno_rows.append(parts[6:6 + 2 * n_snp_eff])

    n_ind = len(iids)
    if n_ind == 0:
        raise ValueError("Aucun individu charge. Colonnes attendues : " +
                         str(expected_cols_eff))

    if rejected_short > 0:
        st.warning(str(rejected_short) + " ligne(s) ignoree(s).")

    allele_counts = [Counter() for _ in range(n_snp_eff)]
    for row in geno_rows:
        for j in range(n_snp_eff):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            c = allele_counts[j]
            c[a1] += 1
            c[a2] += 1

    minor = [None] * n_snp_eff
    for j, c in enumerate(allele_counts):
        if len(c) >= 2:
            minor[j] = min(c, key=c.get)

    gt = np.full((n_ind, n_snp_eff), np.nan, dtype=np.float32)
    for i, row in enumerate(geno_rows):
        for j in range(n_snp_eff):
            m = minor[j]
            if m is None:
                continue
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            gt[i, j] = (1 if a1 == m else 0) + (1 if a2 == m else 0)

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})
    return gt, ind_df, rejected_short + rejected_empty


def generate_demo_data(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng = np.random.default_rng(seed)
    pool = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
            "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]
    n_pop = max(1, min(n_pop, len(pool)))
    pop_names = pool[:n_pop]

    per_pop = n_ind // n_pop
    remainder = n_ind - per_pop * n_pop
    counts = [per_pop + (1 if i < remainder else 0) for i in range(n_pop)]
    n_ind_eff = sum(counts)

    p_anc = rng.beta(0.6, 0.6, n_snp)
    drift = 0.15
    gt = np.zeros((n_ind_eff, n_snp), dtype=np.float32)
    ind_rows = []
    k = 0
    for pop, n_i in zip(pop_names, counts):
        p_k = np.clip(p_anc + rng.normal(0, drift, n_snp), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(n_snp) < p_k).astype(np.float32)
            a2 = (rng.random(n_snp) < p_k).astype(np.float32)
            gt[k] = a1 + a2
            ind_rows.append({"FID": pop,
                             "IID": pop + "_" + str(i + 1).zfill(3)})
            k += 1

    gt[rng.random(gt.shape) < 0.02] = np.nan

    ind_df = pd.DataFrame(ind_rows)
    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c])
                    for c in chr_names], dtype=np.int64)
    snp_df = pd.DataFrame({
        "CHR": chr_names,
        "SNP": ["rs" + str(i).zfill(7) for i in range(n_snp)],
        "CM": 0.0, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    snp_df = (snp_df.sort_values(["_k", "BP"]).drop(columns="_k")
              .reset_index(drop=True))
    return gt, ind_df, snp_df


def impute_mean(gt):
    gt2 = gt.astype(np.float32, copy=True)
    col_mean = np.nanmean(gt2, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(gt2)
    if not nan_mask.any():
        return gt2
    gt2[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return gt2


def missingness_per_ind(gt):
    return np.isnan(gt).mean(axis=1)


def missingness_per_snp(gt):
    return np.isnan(gt).mean(axis=0)


def allele_freq(gt):
    return np.nanmean(gt, axis=0) / 2.0


def maf(gt):
    p = allele_freq(gt)
    return np.minimum(p, 1.0 - p)


def heterozygosity(gt):
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt == 1, axis=1)


def hwe_exact_p(n_het, n_hom1, n_hom2):
    n = n_het + n_hom1 + n_hom2
    if n == 0:
        return np.nan
    if n_het == 0 and (n_hom1 == 0 or n_hom2 == 0):
        return 1.0
    rare = 2 * min(n_hom1, n_hom2) + n_het
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2:
        mid += 1
    probs = np.zeros(rare + 1)
    probs[mid] = 1.0
    mysum = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = (probs[ch] * 4 * chr_ * chc /
                         ((ch + 2) * (ch + 1)))
        mysum += probs[ch + 2]
        ch += 2
        chr_ -= 1
        chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = (probs[ch] * ch * (ch - 1) /
                         (4 * (chr_ + 1) * (chc + 1)))
        mysum += probs[ch - 2]
        ch -= 2
        chr_ += 1
        chc += 1
    p_obs = probs[n_het] if n_het < len(probs) else 0.0
    return float(min(probs[probs <= p_obs + 1e-7].sum() / mysum, 1.0))


@cache_data
def hwe_pvalues(gt):
    _, n_snp = gt.shape
    pvals = np.full(n_snp, np.nan, dtype=np.float64)
    nan_mask = np.isnan(gt)
    n0 = ((gt == 0) & ~nan_mask).sum(axis=0)
    n1 = ((gt == 1) & ~nan_mask).sum(axis=0)
    n2 = ((gt == 2) & ~nan_mask).sum(axis=0)
    n_valid = n0 + n1 + n2

    if n_snp <= HWE_EXACT_MAX_SNP:
        for j in range(n_snp):
            if n_valid[j] < 5:
                continue
            pvals[j] = hwe_exact_p(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0, n_valid, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_het = 2.0 * p * (1.0 - p) * n_valid
            valid = (exp_het >= 5) & np.isfinite(exp_het)
            chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
            pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals


def filter_autosomes(gt, snp_df, keep_auto_only=True):
    chr_str = snp_df["CHR"].astype(str).str.upper().str.replace("CHR", "")
    if keep_auto_only:
        keep = chr_str.isin([str(i) for i in range(1, 30)])
    else:
        keep = chr_str.isin([str(i) for i in range(1, 30)] +
                            ["X", "Y", "MT"])
    n_before = len(snp_df)
    gt2 = gt[:, keep.values]
    snp2 = snp_df[keep].reset_index(drop=True)
    if keep_auto_only:
        st.info("Autosomes 1-29 : " + str(n_before - len(snp2)) +
                " SNPs exclus, " + str(len(snp2)) + " conserves.")
    return gt2, snp2


def _align_shapes(gt, ind_df, snp_df):
    n_gt, m_gt = gt.shape
    n_ind, n_snp = min(n_gt, len(ind_df)), min(m_gt, len(snp_df))
    return (gt[:n_ind, :n_snp],
            ind_df.iloc[:n_ind].reset_index(drop=True),
            snp_df.iloc[:n_snp].reset_index(drop=True))


def apply_qc_filters(gt, ind_df, snp_df, params):
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    trace = {}

    keep = missingness_per_snp(gt) <= params["geno"]
    gt = gt[:, keep]
    snp_df = snp_df[keep].reset_index(drop=True)
    trace["geno_exclus"] = int(m0 - gt.shape[1])

    keep = missingness_per_ind(gt) <= params["mind"]
    gt = gt[keep]
    ind_df = ind_df[keep].reset_index(drop=True)
    trace["mind_exclus"] = int(n0 - gt.shape[0])

    if gt.shape[1] == 0 or gt.shape[0] == 0:
        raise ValueError("Tous les SNPs ou individus exclus (missingness).")

    m_before = gt.shape[1]
    m = maf(gt)
    keep = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep]
    snp_df = snp_df[keep].reset_index(drop=True)
    trace["maf_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par MAF.")

    m_before = gt.shape[1]
    pv = hwe_pvalues(gt)
    keep = np.isnan(pv) | (pv >= 1e-6)
    gt = gt[:, keep]
    snp_df = snp_df[keep].reset_index(drop=True)
    if gt.shape[1] > 0:
        pv2 = hwe_pvalues(gt)
        keep = np.isnan(pv2) | (pv2 >= params["hwe"])
        gt = gt[:, keep]
        snp_df = snp_df[keep].reset_index(drop=True)
    trace["hwe_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par HWE.")

    het = heterozygosity(gt)
    keep = np.ones(len(het), dtype=bool)
    for fid in ind_df["FID"].unique():
        mask = (ind_df["FID"] == fid).values
        sub = het[mask]
        if len(sub) < 5 or np.nanstd(sub) < 1e-9:
            continue
        z = (sub - np.nanmean(sub)) / np.nanstd(sub)
        sub_keep = np.abs(z) <= params["het_sd"]
        idx = np.where(mask)[0]
        keep[idx[~sub_keep]] = False
    n_before = gt.shape[0]
    gt = gt[keep]
    ind_df = ind_df[keep].reset_index(drop=True)
    trace["het_exclus"] = int(n_before - gt.shape[0])

    if gt.shape[0] == 0:
        raise ValueError("Tous les individus exclus (heterozygotie).")

    return gt, ind_df, snp_df, {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]),
        "trace": trace,
    }


@cache_data
def ld_pruning(gt, window=50, step=5, r2_thr=0.2):
    n_snp = gt.shape[1]
    if n_snp < 3:
        return np.ones(n_snp, dtype=bool)

    keep = np.ones(n_snp, dtype=bool)
    X = impute_mean(gt)
    X = X - X.mean(axis=0)

    i = 0
    while i < n_snp:
        win_end = min(i + step, n_snp)
        while win_end < n_snp and (win_end - i) < window:
            j = win_end
            if keep[j] and keep[i]:
                a, b = X[:, i], X[:, j]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    r = np.corrcoef(a, b)[0, 1]
                    if r * r > r2_thr:
                        keep[j] = False
            win_end += 1
        i += step
    return keep


@cache_data
def ld_decay(gt, snp_bp, max_kb=1000, max_snp=1500, seed=42):
    n_snp = gt.shape[1]
    if n_snp < 2:
        return pd.DataFrame(columns=["dist_kb", "r2"])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
           if n_snp > max_snp else np.arange(n_snp))
    gt_sub = impute_mean(gt[:, idx])
    bp_sub = np.asarray(snp_bp)[idx].astype(np.float64)
    X = gt_sub - gt_sub.mean(0)
    std = gt_sub.std(0)
    std[std < 1e-8] = np.nan
    X = X / std
    C = (X.T @ X) / X.shape[0]
    R2 = C ** 2
    iu, ju = np.triu_indices(len(idx), k=1)
    dist_kb = (bp_sub[ju] - bp_sub[iu]) / 1000.0
    mask = (dist_kb > 0) & (dist_kb <= max_kb)
    return pd.DataFrame({"dist_kb": dist_kb[mask],
                         "r2": R2[iu[mask], ju[mask]]})


@cache_data
def pca_analysis(gt, n_components=10):
    X = impute_mean(gt)
    X = X - X.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X), pca.explained_variance_ratio_ * 100.0


@cache_data
def mds_analysis(gt, n_components=5):
    X = impute_mean(gt)
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(n_components=n_components,
                      dissimilarity="precomputed",
                      random_state=42, n_init=1, max_iter=300,
                      normalized_stress=False).fit_transform(D)


def plot_hist(values, title, xlabel, color="#3498db"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Frequence", height=380,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missingness / individu",
                                        "Missingness / SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60,
                               marker_color="skyblue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60,
                               marker_color="coral"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Frequence manquante", row=1, col=1)
    fig.update_xaxes(title_text="Frequence manquante", row=1, col=2)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    return fig


def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty:
        return None
    d = ld_df.copy()
    d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb
    agg = d.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(agg, x="bin", y="r2",
                  labels={"bin": "Distance (kb)", "r2": "r2 moyen"},
                  title="LD decay", height=450)
    fig.update_traces(line=dict(color="royalblue", width=3))
    return fig


def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Population",
                     title=("PCA - PC1 (" + str(round(var_pct[0], 1)) +
                            " pct) vs PC2 (" +
                            str(round(var_pct[1], 1)) + " pct)"),
                     height=550)
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=1, color="white")))
    return fig


def plot_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS (IBS) - Structure", height=550)
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=1, color="white")))
    return fig


def interpret_qc(qc_stats):
    pct_ind = (qc_stats["excluded_ind"] /
               max(qc_stats["n_ind_init"], 1) * 100)
    pct_snp = (qc_stats["excluded_snp"] /
               max(qc_stats["n_snp_init"], 1) * 100)
    if pct_ind < 2 and pct_snp < 10:
        v, e = "EXCELLENTE", "Qualite de genotypage tres elevee."
    elif pct_ind < 5 and pct_snp < 25:
        v, e = "ACCEPTABLE", "Qualite correcte."
    else:
        v, e = "FAIBLE", "Exclusion elevee."
    return {"verdict": v, "explanation": e,
            "metrics": {"ind_exclus_pct": round(pct_ind, 2),
                        "snp_exclus_pct": round(pct_snp, 2)},
            "recommendation": "Continuer." if "EXCELLENTE" in v
                              else "Verifier la qualite."}


def interpret_maf(maf_values):
    m = np.asarray(maf_values)
    m = m[np.isfinite(m)]
    mm = float(m.mean())
    pct = float((m < 0.05).mean() * 100)
    if mm > 0.25:
        v, e = "Spectre riche", "Bonne diversite."
    elif mm > 0.15:
        v, e = "Diversite moderee", "Population moyenne."
    else:
        v, e = "Diversite faible", "Population consanguine."
    return {"verdict": v, "explanation": e,
            "metrics": {"MAF_moyen": round(mm, 4),
                        "pct_MAF_inf_005": round(pct, 2)},
            "recommendation": "Filtre MAF 0.05 elimine " +
                              str(round(pct, 1)) + " pct des SNPs."}


_STATE_KEYS = ["gt", "ind_df", "snp_df",
               "gt_filt", "ind_filt", "snp_filt", "qc_stats",
               "gt_pruned", "snp_pruned", "ld_df",
               "pca_scores", "pca_var", "mds_coords"]


def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None


def has_data():
    return st.session_state.gt is not None


def has_qc():
    return (st.session_state.gt_filt is not None
            and st.session_state.qc_stats is not None
            and st.session_state.ind_filt is not None
            and st.session_state.snp_filt is not None)


def has_pruned():
    return (st.session_state.gt_pruned is not None
            and st.session_state.snp_pruned is not None)


def _fid_array():
    return st.session_state.ind_filt["FID"].astype(str).to_numpy()


def main():
    init_state()
    st.title("Bovine SNP Platform v5.0")
    st.caption("Etape 2 : QC + LD Pruning + Structure (PCA/MDS)")

    with st.sidebar:
        st.header("Donnees")
        mode = st.radio("Source :",
                        ["Demo", "Upload PED/MAP"],
                        index=0, key="sb_mode_source")

        if mode == "Demo":
            c1, c2 = st.columns(2)
            n_ind = c1.number_input("Individus", 20, 1000, 150, 10,
                                    key="sb_n_ind")
            n_snp = c2.number_input("SNPs", 100, 10000, 800, 100,
                                    key="sb_n_snp")
            n_pop = st.slider("Populations", 2, 10, 4, key="sb_n_pop")
            if st.button("Generer le jeu de demo",
                         use_container_width=True, key="sb_btn_demo"):
                with st.spinner("Generation..."):
                    gt, ind_df, snp_df = generate_demo_data(
                        int(n_ind), int(n_snp), int(n_pop))
                    ind_df = reconstruct_fid_from_iid(ind_df)
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind_df
                    st.session_state.snp_df = snp_df
                    st.session_state.gt_filt = None
                    st.session_state.ind_filt = None
                    st.session_state.snp_filt = None
                    st.session_state.qc_stats = None
                    st.session_state.gt_pruned = None
                    st.session_state.snp_pruned = None
                    st.session_state.pca_scores = None
                    st.session_state.mds_coords = None
                st.success(str(gt.shape[0]) + " ind x " +
                           str(gt.shape[1]) + " SNPs")

        else:
            st.info("Formats : .ped, .ped.gz, .map, .map.gz")
            ped_f = st.file_uploader("Fichier .ped",
                                     type=["ped", "gz", "txt"],
                                     key="sb_ped_file")
            map_f = st.file_uploader("Fichier .map",
                                     type=["map", "gz", "txt"],
                                     key="sb_map_file")
            if ped_f and map_f:
                if st.button("Charger PED + MAP",
                             use_container_width=True, key="sb_btn_ped"):
                    try:
                        with st.spinner("Parsing .map..."):
                            map_df, _ = parse_map(map_f.read())
                        with st.spinner("Parsing .ped..."):
                            gt, ind_df, _ = parse_ped(ped_f.read(),
                                                      len(map_df))
                        if len(map_df) != gt.shape[1]:
                            map_df = (map_df.iloc[:gt.shape[1]]
                                      .reset_index(drop=True))
                        ind_df = reconstruct_fid_from_iid(ind_df)
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = map_df
                        st.session_state.gt_filt = None
                        st.session_state.ind_filt = None
                        st.session_state.snp_filt = None
                        st.session_state.qc_stats = None
                        st.session_state.gt_pruned = None
                        st.session_state.snp_pruned = None
                        st.session_state.pca_scores = None
                        st.session_state.mds_coords = None
                        st.success(str(gt.shape[0]) + " ind x " +
                                   str(gt.shape[1]) + " SNPs")
                    except Exception as e:
                        st.error("Erreur : " + str(e))

        st.divider()
        st.header("Seuils QC")
        geno = st.slider("Missingness SNP", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["geno"], 0.01, key="sb_geno")
        mind = st.slider("Missingness individu", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["mind"], 0.01, key="sb_mind")
        maf_thr = st.slider("MAF minimal", 0.0, 0.5,
                            DEFAULT_THRESHOLDS["maf"], 0.01, key="sb_maf")
        hwe_thr = st.number_input("HWE p-value",
                                  value=DEFAULT_THRESHOLDS["hwe"],
                                  format="%.0e", key="sb_hwe")
        het_sd = st.slider("Heterozygotie sigma intra-race", 1.0, 5.0,
                           DEFAULT_THRESHOLDS["het_sd"], 0.1,
                           key="sb_het_sd")
        st.divider()
        ld_r2 = st.slider("LD Pruning r2 seuil", 0.05, 0.5,
                          DEFAULT_THRESHOLDS["ld_r2"], 0.05,
                          key="sb_ld_r2")

    if not has_data():
        st.info("Generez un jeu de demo ou importez PED/MAP.")
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["Apercu", "QC", "LD Pruning", "Structure"])

    with tabs[0]:
        step_header("0", "Apercu des donnees",
                    "Comptage des individus, SNPs, races.",
                    "Verifier la coherence avant analyse.",
                    "Apercu visuel du jeu de donnees.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations (FID)", ind_df["FID"].nunique())

        st.subheader("Individus")
        st.dataframe(ind_df.head(20), use_container_width=True)
        st.subheader("SNPs")
        st.dataframe(snp_df.head(5), use_container_width=True)

    with tabs[1]:
        st.subheader("Controle Qualite (QC)")
        step_header("3", "QC Missingness, MAF, HWE",
                    "Cascade : geno mind maf hwe het_sd.",
                    "Eliminer SNPs et individus de mauvaise qualite.",
                    "Jeu de donnees propre.")

        if st.button("Filtrer autosomes 1-29", key="qc_btn_auto"):
            try:
                gt_a, snp_a = filter_autosomes(st.session_state.gt,
                                               st.session_state.snp_df)
                st.session_state.gt = gt_a
                st.session_state.snp_df = snp_a
                st.success("Autosomes : " + str(gt_a.shape[1]) + " SNPs")
            except Exception as e:
                st.error("Erreur : " + str(e))

        if st.button("Lancer le QC complet", type="primary",
                     key="qc_btn_run"):
            try:
                with st.spinner("Filtrage..."):
                    params = {"geno": geno, "mind": mind, "maf": maf_thr,
                              "hwe": hwe_thr, "het_sd": het_sd}
                    gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                        st.session_state.gt, st.session_state.ind_df,
                        st.session_state.snp_df, params)
                    st.session_state.gt_filt = gt_f
                    st.session_state.ind_filt = ind_f
                    st.session_state.snp_filt = snp_f
                    st.session_state.qc_stats = qc_stats
                    st.session_state.gt_pruned = None
                    st.session_state.snp_pruned = None
                    st.session_state.pca_scores = None
                    st.session_state.mds_coords = None
                st.success("QC termine.")
            except Exception as e:
                st.error("Erreur : " + str(e))

        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus finaux", s["n_ind_final"],
                      delta=-s["excluded_ind"], delta_color="inverse")
            c2.metric("SNPs finaux", s["n_snp_final"],
                      delta=-s["excluded_snp"], delta_color="inverse")
            c3.metric("Exclus (ind)", s["excluded_ind"])
            c4.metric("Exclus (SNPs)", s["excluded_snp"])

            st.markdown("**Cascade de filtrage**")
            st.json(s.get("trace", {}))

            gt_full = st.session_state.gt
            st.plotly_chart(
                plot_missingness_dashboard(
                    missingness_per_ind(gt_full),
                    missingness_per_snp(gt_full)),
                use_container_width=True, key="qc_plot_missing")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_hist(maf(gt_full), "Spectre MAF",
                              "MAF", "#2ecc71"),
                    use_container_width=True, key="qc_plot_maf")
            with c2:
                st.plotly_chart(
                    plot_hist(heterozygosity(gt_full),
                              "Heterozygotie observee",
                              "HET", "#9b59b6"),
                    use_container_width=True, key="qc_plot_het")

            st.divider()
            st.subheader("Interpretation IA QC")
            qc_interp = interpret_qc(s)
            st.success("**" + str(qc_interp["verdict"]) + "** - " +
                       str(qc_interp["explanation"]))
            st.json(qc_interp["metrics"])
            st.info(qc_interp["recommendation"])

            st.divider()
            st.subheader("Interpretation IA MAF")
            maf_interp = interpret_maf(maf(gt_full))
            st.success("**" + str(maf_interp["verdict"]) + "** - " +
                       str(maf_interp["explanation"]))
            st.json(maf_interp["metrics"])
            st.info(maf_interp["recommendation"])

    with tabs[2]:
        st.subheader("LD Pruning")
        step_header("4.5", "Elagage par desequilibre de liaison",
                    "Retire les SNPs correles (r2 > seuil) dans fenetre de "
                    "50 SNPs.",
                    "PCA et ADMIXTURE supposent marqueurs independants.",
                    "Sous-ensemble quasi-independant.")

        if not has_qc():
            st.warning("Lancez d'abord le QC.")
        else:
            if st.button("Lancer le LD Pruning", type="primary",
                         key="ld_btn_run"):
                try:
                    with st.spinner("LD pruning..."):
                        keep = ld_pruning(st.session_state.gt_filt,
                                          window=50, step=5,
                                          r2_thr=ld_r2)
                        st.session_state.gt_pruned = (
                            st.session_state.gt_filt[:, keep])
                        st.session_state.snp_pruned = (
                            st.session_state.snp_filt[keep]
                            .reset_index(drop=True))
                    st.success(str(st.session_state.gt_pruned.shape[1]) +
                               " SNPs conserves.")
                except Exception as e:
                    st.error("Erreur : " + str(e))

            if has_pruned():
                c1, c2, c3 = st.columns(3)
                c1.metric("SNPs avant",
                          st.session_state.gt_filt.shape[1])
                c2.metric("SNPs apres",
                          st.session_state.gt_pruned.shape[1])
                pct = (100 * (1 - st.session_state.gt_pruned.shape[1] /
                              st.session_state.gt_filt.shape[1]))
                c3.metric("Reduction", str(round(pct, 1)) + " pct")

                with st.spinner("LD decay..."):
                    ld_df = ld_decay(st.session_state.gt_pruned,
                                     st.session_state.snp_pruned["BP"].values,
                                     max_kb=1000, max_snp=1000)
                    st.session_state.ld_df = ld_df
                fig = plot_ld_decay(ld_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    key="ld_plot_decay")

    with tabs[3]:
        st.subheader("Structure des populations")
        step_header("5.3-5.4", "PCA et MDS",
                    "PCA et MDS sur les SNPs independants.",
                    "Visualiser la structure genetique.",
                    "Nuages 2D des individus.")

        if not has_qc():
            st.warning("Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)

            if st.button("Lancer PCA + MDS", type="primary",
                         key="struct_btn_run"):
                try:
                    with st.spinner("PCA..."):
                        s_, v_ = pca_analysis(gt_use, 10)
                        st.session_state.pca_scores = s_
                        st.session_state.pca_var = v_
                    with st.spinner("MDS..."):
                        st.session_state.mds_coords = mds_analysis(gt_use, 5)
                    st.success("Termine.")
                except Exception as e:
                    st.error("Erreur : " + str(e))

            if st.session_state.pca_scores is not None:
                labels = _fid_array()
                st.plotly_chart(
                    plot_pca(st.session_state.pca_scores,
                             st.session_state.pca_var, labels),
                    use_container_width=True, key="struct_plot_pca")

                var_tab = pd.DataFrame({
                    "PC": ["PC" + str(i + 1) for i in range(10)],
                    "Variance_pct": np.round(
                        st.session_state.pca_var[:10], 3)})
                st.dataframe(var_tab, use_container_width=True)

            if st.session_state.mds_coords is not None:
                labels = _fid_array()
                st.plotly_chart(
                    plot_mds(st.session_state.mds_coords, labels),
                    use_container_width=True, key="struct_plot_mds")


if __name__ == "__main__":
    main()
