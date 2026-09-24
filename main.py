"""
Bovine SNP Platform v5.0 - Etape 4
Chargement + QC + LD Pruning + Structure + GWAS
+ FST + Selection + ROH + Demographie (Ne historique)
"""
import gzip
import io
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
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
    # Cas tableau de chaines (FID, IID, etc)
    if x.dtype.kind in ("U", "S", "O"):
        preview = "|".join(str(v) for v in x.ravel()[:500])
        return ("strarr|" + str(x.shape) + "|" + preview)
    # Cas numerique
    try:
        return (str(x.shape) + "|" + str(x.dtype) + "|" +
                str(float(np.nansum(x))) + "|" +
                str(float(np.nansum(np.abs(x)))))
    except Exception:
        return "ndarray|" + str(x.shape) + "|" + str(x.dtype)


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


@cache_data
def king_kinship(gt):
    n_ind = gt.shape[0]
    het = np.nansum(gt == 1, axis=1).astype(np.float64)
    K = np.zeros((n_ind, n_ind))
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            both_het = np.nansum((gt[i] == 1) & (gt[j] == 1))
            denom = het[i] + het[j]
            phi = (denom - 2 * both_het) / denom if denom > 0 else 0.0
            K[i, j] = K[j, i] = phi
    np.fill_diagonal(K, 0.5)
    return K


@cache_data
def fst_per_snp(gt, pop_labels):
    pops = np.unique(np.asarray(pop_labels))
    if len(pops) < 2:
        return np.full(gt.shape[1], np.nan)
    fst = np.full(gt.shape[1], np.nan, dtype=np.float64)
    masks = {p: (np.asarray(pop_labels) == p) for p in pops}
    for j in range(gt.shape[1]):
        pl, nl = [], []
        for p in pops:
            vals = gt[masks[p], j]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 3:
                continue
            pl.append(vals.mean() / 2.0)
            nl.append(len(vals))
        if len(pl) < 2:
            continue
        p_arr, n_arr = np.asarray(pl), np.asarray(nl)
        p_bar = np.average(p_arr, weights=n_arr)
        h_s = np.average(2.0 * p_arr * (1.0 - p_arr), weights=n_arr)
        h_t = 2.0 * p_bar * (1.0 - p_bar)
        if h_t > 1e-9:
            fst[j] = (h_t - h_s) / h_t
    return fst


@cache_data
def fst_pairwise(gt, pop_labels):
    labels = np.asarray(pop_labels)
    pops = sorted(np.unique(labels))
    K = len(pops)
    matrix = np.full((K, K), np.nan)
    np.fill_diagonal(matrix, 0.0)
    masks = {p: (labels == p) for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            g1 = gt[masks[pops[i]]]
            g2 = gt[masks[pops[j]]]
            p1 = np.nanmean(g1, 0) / 2.0
            p2 = np.nanmean(g2, 0) / 2.0
            n1 = (~np.isnan(g1)).sum(0)
            n2 = (~np.isnan(g2)).sum(0)
            denom = np.where(n1 + n2 > 0, n1 + n2, np.nan)
            p_bar = (p1 * n1 + p2 * n2) / denom
            hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / denom
            ht = 2 * p_bar * (1 - p_bar)
            with np.errstate(divide="ignore", invalid="ignore"):
                fst_j = (ht - hs) / ht
            matrix[i, j] = matrix[j, i] = float(np.nanmean(fst_j))
    return matrix, pops


@cache_data
def selection_signatures(gt, snp_df, pop_labels):
    fst = fst_per_snp(gt, np.asarray(pop_labels))
    hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    df = snp_df.copy()
    df["FST"] = fst
    df["HOM"] = hom
    hom_std = df["HOM"].std()
    df["SCORE"] = (df["FST"].fillna(0) *
                   (df["HOM"] - df["HOM"].mean()) /
                   (hom_std if hom_std > 1e-9 else 1.0))
    return df.sort_values("SCORE", ascending=False)


@cache_data
def detect_roh(gt, snp_df_json, min_snps=30, min_kb=500.0):
    snp_df = pd.read_json(io.StringIO(snp_df_json))
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    n_ind = gt.shape[0]
    rohs, froh_bp = [], np.zeros(n_ind)
    genome_bp_total = 0.0

    for chrom in pd.unique(chr_arr):
        idxs = np.where(chr_arr == chrom)[0]
        if len(idxs) < min_snps:
            continue
        order = np.argsort(bp[idxs])
        idxs = idxs[order]
        chrom_span = bp[idxs[-1]] - bp[idxs[0]]
        if chrom_span < 1:
            continue
        genome_bp_total += chrom_span

        for i in range(n_ind):
            col = gt[i, idxs]
            start_k = None
            for k in range(len(idxs)):
                is_hom = (col[k] == 0) or (col[k] == 2)
                if is_hom:
                    if start_k is None:
                        start_k = k
                else:
                    if start_k is not None:
                        n_run = k - start_k
                        length_kb = ((bp[idxs[k - 1]] -
                                      bp[idxs[start_k]]) / 1000.0)
                        if n_run >= min_snps and length_kb >= min_kb:
                            rohs.append({
                                "IID_idx": i, "CHR": chrom,
                                "start_bp": int(bp[idxs[start_k]]),
                                "end_bp": int(bp[idxs[k - 1]]),
                                "n_snp": int(n_run),
                                "length_kb": float(length_kb)})
                            froh_bp[i] += (bp[idxs[k - 1]] -
                                           bp[idxs[start_k]])
                        start_k = None
            if start_k is not None:
                n_run = len(idxs) - start_k
                length_kb = (bp[idxs[-1]] - bp[idxs[start_k]]) / 1000.0
                if n_run >= min_snps and length_kb >= min_kb:
                    rohs.append({
                        "IID_idx": i, "CHR": chrom,
                        "start_bp": int(bp[idxs[start_k]]),
                        "end_bp": int(bp[idxs[-1]]),
                        "n_snp": int(n_run),
                        "length_kb": float(length_kb)})
                    froh_bp[i] += bp[idxs[-1]] - bp[idxs[start_k]]

    froh = froh_bp / genome_bp_total if genome_bp_total > 0 else froh_bp
    roh_df = (pd.DataFrame(rohs) if rohs else
              pd.DataFrame(columns=["IID_idx", "CHR", "start_bp",
                                    "end_bp", "n_snp", "length_kb"]))
    return roh_df, froh


@cache_data
def estimate_ne_historical(gt, snp_df, max_snp=2000):
    bp = snp_df["BP"].astype(np.float64).values
    chr_arr = snp_df["CHR"].astype(str).values
    n_snp = gt.shape[1]
    if n_snp > max_snp:
        idx = np.linspace(0, n_snp - 1, max_snp).astype(int)
        gt = gt[:, idx]
        bp = bp[idx]
        chr_arr = chr_arr[idx]

    X = impute_mean(gt)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r2_list, dist_list = [], []
    n = X.shape[1]
    for i in range(0, n - 1):
        for j in range(i + 1, min(i + 50, n)):
            if chr_arr[i] != chr_arr[j]:
                continue
            d = abs(bp[j] - bp[i]) / 1e6
            if d < 0.001 or d > 50:
                continue
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            r2_list.append(r * r)
            dist_list.append(d)

    if not dist_list:
        return pd.DataFrame({"GenAgo": [], "Ne": []})

    r2 = np.array(r2_list)
    dist = np.array(dist_list)
    bins = np.logspace(np.log10(dist.min() + 1e-6),
                       np.log10(dist.max()), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    r2b = np.array([
        r2[(dist >= bins[k]) & (dist < bins[k + 1])].mean()
        if ((dist >= bins[k]) & (dist < bins[k + 1])).sum() > 0 else np.nan
        for k in range(len(bins) - 1)])
    m = np.isfinite(r2b)
    if m.sum() < 3:
        return pd.DataFrame({"GenAgo": [], "Ne": []})

    c_morgans = bc[m] * 0.01
    r2v = r2b[m]
    ne = np.where(r2v > 1e-4, 1.0 / (4 * c_morgans * r2v), np.nan)
    gen_ago = bc[m] * 100
    df = pd.DataFrame({"GenAgo": gen_ago, "Ne": ne}).dropna()
    df = df[df["Ne"] > 0].sort_values("GenAgo")
    if len(df) > 3:
        df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df


def load_phenotypes_file(file_bytes, filename=""):
    if filename.endswith(".gz"):
        try:
            raw = gzip.decompress(file_bytes)
        except Exception:
            raw = file_bytes
    else:
        raw = file_bytes
    text = raw.decode("utf-8", errors="replace")
    first_line = text.splitlines()[0] if text else ""
    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def match_phenotypes(ind_df, pheno_df, iid_col=None):
    if iid_col is None:
        for c in ["IID", "id", "ID", "sample", "Sample", "animal",
                  "Animal"]:
            if c in pheno_df.columns:
                iid_col = c
                break
    if iid_col is None:
        raise ValueError("Aucune colonne IID trouvee. Colonnes dispo : " +
                         str(list(pheno_df.columns)))

    p = pheno_df.copy()
    p[iid_col] = p[iid_col].astype(str)
    ind = ind_df.copy()
    ind["_key"] = ind["IID"].astype(str)
    p["_key"] = p[iid_col]

    merged = ind[["_key", "FID", "IID"]].merge(
        p.drop(columns=[iid_col]), on="_key", how="left")
    merged = merged.drop(columns="_key")
    return merged, iid_col


def _vc_estimate_delta(Y, X0, K):
    n = len(Y)
    d, U = np.linalg.eigh(K)
    d = np.maximum(d, 0)
    Yt = U.T @ Y
    X0t = U.T @ X0

    def neg_loglik(delta):
        if delta <= 0:
            return 1e18
        w = 1.0 / np.sqrt(delta * d + 1.0)
        Yw = Yt * w
        Xw = X0t * w[:, None]
        try:
            beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except np.linalg.LinAlgError:
            return 1e18
        resid = Yw - Xw @ beta
        s2 = float(np.sum(resid ** 2) / n)
        if s2 <= 0:
            return 1e18
        return -(-0.5 * (n * np.log(s2) +
                         np.sum(np.log(delta * d + 1.0))))

    grid = np.logspace(-3, 3, 30)
    lls = np.array([neg_loglik(dl) for dl in grid])
    i_best = int(np.argmin(lls))
    lo = grid[max(0, i_best - 1)]
    hi = grid[min(len(grid) - 1, i_best + 1)]
    try:
        res = minimize_scalar(neg_loglik, bounds=(lo, hi),
                              method="bounded",
                              options={"xatol": 1e-4})
        delta_opt = float(res.x)
    except Exception:
        delta_opt = float(grid[i_best])
    return max(delta_opt, 1e-6)


@cache_data
def run_gwas_lmm(gt, pheno_vec, K, covariates,
                 maf_min=0.05, missing_max=0.05, use_lmm=True):
    n_all, n_snp = gt.shape
    pheno_vec = np.asarray(pheno_vec, dtype=float)

    valid_ind = np.isfinite(pheno_vec)
    if valid_ind.sum() < 10:
        raise ValueError("Moins de 10 individus avec phenotype valide.")

    Y = pheno_vec[valid_ind]
    gt_v = gt[valid_ind]
    K_v = np.asarray(K)[np.ix_(valid_ind, valid_ind)]

    if covariates is None or covariates.size == 0:
        X0 = np.ones((len(Y), 1))
    else:
        cov = np.asarray(covariates)[valid_ind]
        X0 = np.column_stack([np.ones(len(Y)), cov])

    keep_cols = np.ones(X0.shape[1], dtype=bool)
    for j in range(1, X0.shape[1]):
        if X0[:, j].std() < 1e-10:
            keep_cols[j] = False
    X0 = X0[:, keep_cols]

    n = len(Y)

    if use_lmm:
        delta = _vc_estimate_delta(Y, X0, K_v)
        d, U = np.linalg.eigh(K_v)
        d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(delta * d + 1.0)
        Yt = (U.T @ Y) * w
        X0t = (U.T @ X0) * w[:, None]
    else:
        Yt = Y
        X0t = X0

    out = {"SNP_idx": [], "MAF": [], "BETA": [], "SE": [],
           "T": [], "P": []}

    for j in range(n_snp):
        g = gt_v[:, j].astype(float)
        nan_mask = ~np.isfinite(g)
        miss_j = nan_mask.mean()
        if miss_j > missing_max:
            continue

        if nan_mask.any():
            m_g = np.nanmean(g)
            if not np.isfinite(m_g):
                continue
            g = np.where(nan_mask, m_g, g)

        p = g.mean() / 2.0
        maf_j = min(p, 1.0 - p)
        if maf_j < maf_min:
            continue

        if use_lmm:
            gw = (U.T @ g) * w
            X = np.column_stack([X0t, gw])
        else:
            X = np.column_stack([X0t, g])

        try:
            beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
        except np.linalg.LinAlgError:
            continue

        resid = Yt - X @ beta
        dof = max(n - X.shape[1], 1)
        s2 = float(np.sum(resid ** 2) / dof)
        if s2 <= 0:
            continue

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        se = float(np.sqrt(s2 * XtX_inv[-1, -1]))
        if se <= 0 or not np.isfinite(se):
            continue

        t_val = beta[-1] / se
        p_val = float(2 * t_dist.sf(abs(t_val), df=dof))

        out["SNP_idx"].append(j)
        out["MAF"].append(float(maf_j))
        out["BETA"].append(float(beta[-1]))
        out["SE"].append(se)
        out["T"].append(float(t_val))
        out["P"].append(p_val)

    if not out["SNP_idx"]:
        return pd.DataFrame(
            columns=["SNP_idx", "MAF", "BETA", "SE", "T", "P"])

    return pd.DataFrame(out)


def annotate_gwas_results(res_df, snp_df):
    if res_df.empty:
        return res_df
    idx = res_df["SNP_idx"].astype(int).values
    res_df = res_df.copy()
    res_df["SNP"] = snp_df["SNP"].astype(str).values[idx]
    res_df["CHR"] = snp_df["CHR"].astype(str).values[idx]
    res_df["BP"] = snp_df["BP"].astype(int).values[idx]
    return res_df[["SNP_idx", "SNP", "CHR", "BP", "MAF",
                   "BETA", "SE", "T", "P"]]


def compute_lambda_gc(pvals):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return np.nan
    chi2_obs = stats.chi2.isf(p, df=1)
    return float(np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1))


def gwas_thresholds(pvals, alpha=0.05):
    p = np.asarray(pvals)
    p = p[np.isfinite(p)]
    n = len(p)
    if n == 0:
        return {"bonferroni": np.nan, "fdr_05": np.nan, "n_tests": 0}
    bonf = alpha / n
    ps = np.sort(p)
    bh = ps * n / np.arange(1, n + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    sig_idx = np.where(bh <= alpha)[0]
    fdr_thr = float(ps[sig_idx[-1]]) if len(sig_idx) else np.nan
    return {"bonferroni": float(bonf), "fdr_05": fdr_thr, "n_tests": int(n)}


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


def plot_manhattan(fst, chr_col, threshold_q=0.999):
    df = pd.DataFrame({"FST": np.asarray(fst), "CHR": np.asarray(chr_col)})
    df = df.dropna(subset=["FST"]).reset_index(drop=True)
    if df.empty:
        return None
    order = df["CHR"].map(_chr_sort_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df["_k2"] = order.map(lambda t: t[2])
    df = df.sort_values(["_k0", "_k1", "_k2"]).reset_index(drop=True)

    cumulative, ticks, labels = 0, [], []
    cum_pos = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        sub_idx = df.index[df["CHR"] == chrom]
        s, e = cumulative, cumulative + len(sub_idx)
        cum_pos[s:e] = np.arange(s, e)
        ticks.append((s + e) / 2)
        labels.append(str(chrom))
        cumulative = e
    df["x"] = cum_pos

    fig = go.Figure()
    for idx_chr, chrom in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == chrom]
        color = "#2c3e50" if idx_chr % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"],
                                 mode="markers",
                                 marker=dict(size=5, color=color),
                                 showlegend=False, hoverinfo="skip"))
    q_upper = float(np.nanquantile(fst, threshold_q))
    fig.add_hline(y=q_upper, line_dash="dash", line_color="red",
                  annotation_text="Top " +
                  str(round(100 * (1 - threshold_q), 1)) + " pct = " +
                  str(round(q_upper, 4)),
                  annotation_position="top right")
    fig.update_layout(title="Manhattan Plot - FST/SNP",
                      xaxis_title="Chromosome", yaxis_title="FST",
                      height=500, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_fst_pairwise(matrix, pops):
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=pops, y=pops, colorscale="Viridis",
        colorbar=dict(title="FST"),
        text=np.round(matrix, 4), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>FST = %{z:.4f}<extra></extra>"))
    fig.update_layout(title="FST pairwise", height=550)
    return fig


def plot_roh_histogram(froh, labels):
    df = pd.DataFrame({"FROH": froh, "Population": labels})
    return px.histogram(df, x="FROH", color="Population", nbins=50,
                        title="Distribution de FROH", height=450)


def plot_ne_curves(ne_dict):
    fig = go.Figure()
    palette = (px.colors.qualitative.Set2
               + px.colors.qualitative.Set3)
    for i, (race, df) in enumerate(ne_dict.items()):
        if df is None or df.empty:
            continue
        fig.add_trace(go.Scatter(x=df["GenAgo"], y=df["Ne"],
                                 mode="lines+markers", name=race,
                                 line=dict(color=palette[i % len(palette)],
                                           width=3)))
    fig.update_layout(
        title="Evolution historique de la taille efficace (Ne)",
        xaxis_title="Generations passees",
        yaxis_title="Ne (interpretation relative)",
        height=550)
    return fig


def plot_gwas_manhattan(res_df, bonferroni=None, fdr=None,
                        title="GWAS Manhattan"):
    df = res_df.copy()
    df = df[np.isfinite(df["P"])].reset_index(drop=True)
    if df.empty:
        return None
    df["-log10P"] = -np.log10(np.clip(df["P"], 1e-300, 1))

    order = df["CHR"].map(_chr_sort_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df["_k2"] = order.map(lambda t: t[2])
    df = df.sort_values(["_k0", "_k1", "BP"]).reset_index(drop=True)

    cum, ticks, labels = 0, [], []
    pos = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        idx = np.where(df["CHR"] == chrom)[0]
        pos[idx] = np.arange(cum, cum + len(idx))
        ticks.append((cum + cum + len(idx) - 1) / 2)
        labels.append(str(chrom))
        cum += len(idx)
    df["x"] = pos

    fig = go.Figure()
    for i, chrom in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == chrom]
        color = "#2c3e50" if i % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["-log10P"], mode="markers",
            marker=dict(size=5, color=color),
            text=[s + " CHR" + str(c) + ":" + str(b) +
                  " P=" + str(round(p, 6)) + " beta=" + str(round(bb, 3))
                  for s, c, b, p, bb in zip(sub["SNP"], sub["CHR"],
                                            sub["BP"], sub["P"],
                                            sub["BETA"])],
            hoverinfo="text", showlegend=False))

    if bonferroni is not None and np.isfinite(bonferroni):
        fig.add_hline(y=-np.log10(bonferroni), line_dash="dash",
                      line_color="red",
                      annotation_text="Bonferroni",
                      annotation_position="top right")
    if fdr is not None and np.isfinite(fdr):
        fig.add_hline(y=-np.log10(fdr), line_dash="dot",
                      line_color="orange",
                      annotation_text="FDR 5 pct",
                      annotation_position="bottom right")

    fig.update_layout(title=title, xaxis_title="Chromosome",
                      yaxis_title="-log10(P)", height=520,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_gwas_qq(pvals, lambda_gc=None):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return None
    p = np.sort(p)
    n = len(p)
    expected = (np.arange(1, n + 1) - 0.5) / n
    obs = -np.log10(p)
    exp = -np.log10(expected)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exp, y=obs, mode="markers",
                             marker=dict(size=5, color="#2c3e50"),
                             name="Observe"))
    lim = max(exp.max(), obs.max()) * 1.05
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                             line=dict(color="red", dash="dash"),
                             name="Attendu (H0)"))
    sub = ("lambdaGC = " + str(round(lambda_gc, 3))
           if lambda_gc is not None and np.isfinite(lambda_gc) else "")
    fig.update_layout(title="QQ plot " + sub,
                      xaxis_title="Attendu -log10(P)",
                      yaxis_title="Observe -log10(P)",
                      height=520,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_locuszoom(res_df, snp_df, chrom, center_bp, window_kb=500):
    chrom = str(chrom)
    window_bp = window_kb * 1000
    df = res_df.copy()
    df = df[df["CHR"].astype(str) == chrom]
    df = df[(df["BP"] >= center_bp - window_bp) &
            (df["BP"] <= center_bp + window_bp)]
    if df.empty:
        return None
    df["-log10P"] = -np.log10(np.clip(df["P"], 1e-300, 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["BP"] / 1e6, y=df["-log10P"], mode="markers",
        marker=dict(size=8, color=df["-log10P"],
                    colorscale="Viridis",
                    showscale=True, colorbar=dict(title="-log10P")),
        text=[s + " P=" + str(round(p, 6)) + " beta=" + str(round(b, 3))
              for s, p, b in zip(df["SNP"], df["P"], df["BETA"])],
        hoverinfo="text"))
    fig.add_vline(x=center_bp / 1e6, line_dash="dash", line_color="red",
                  annotation_text="Lead SNP",
                  annotation_position="top")
    fig.update_layout(
        title="LocusZoom - CHR " + chrom + " : " +
              str(round(center_bp / 1e6, 2)) + " Mb +/- " +
              str(window_kb) + " kb",
        xaxis_title="Position (Mb)",
        yaxis_title="-log10(P)", height=500)
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


def interpret_fst(fst_values):
    m = float(np.nanmean(np.asarray(fst_values)))
    if m < 0.05:
        v, e = "Faible", "Fort flux genique."
    elif m < 0.15:
        v, e = "MODEREE", "Races distinctes."
    elif m < 0.25:
        v, e = "FORTE", "Races isolees genetiquement."
    else:
        v, e = "TRES FORTE", "Quasi-especes distinctes."
    return {"verdict": v, "explanation": e,
            "metrics": {"FST_moyen": round(m, 4)},
            "recommendation": "Utiliser PC1-PC10 comme covariables."}


def interpret_roh(froh):
    m = float(np.nanmean(np.asarray(froh)))
    if m < 0.02:
        v, e = "Consanguinite faible", "Bonne diversite."
    elif m < 0.10:
        v, e = "Consanguinite moderee", "Niveau normal."
    else:
        v, e = "Consanguinite elevee", "Risque de depression."
    return {"verdict": v, "explanation": e,
            "metrics": {"FROH_moyen": round(m, 4),
                        "FROH_max": round(float(np.nanmax(froh)), 4)},
            "recommendation": ("Croiser avec lignees exterieures"
                               if m > 0.10 else "Population saine.")}


_STATE_KEYS = ["gt", "ind_df", "snp_df",
               "gt_filt", "ind_filt", "snp_filt", "qc_stats",
               "gt_pruned", "snp_pruned", "ld_df",
               "pca_scores", "pca_var", "mds_coords", "king_matrix",
               "pheno_matched", "gwas_results",
               "gwas_pheno_name", "gwas_lambda",
               "fst", "fst_pairwise_matrix", "fst_pops",
               "selection_df", "roh_df", "froh", "ne_dict"]


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
    st.caption("Etape 4 : QC + LD + Structure + GWAS + FST + ROH + Ne")

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
                    for k in _STATE_KEYS[3:]:
                        st.session_state[k] = None
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
                        with st.spinner("Parsing..."):
                            map_df, _ = parse_map(map_f.read())
                            gt, ind_df, _ = parse_ped(ped_f.read(),
                                                      len(map_df))
                        if len(map_df) != gt.shape[1]:
                            map_df = (map_df.iloc[:gt.shape[1]]
                                      .reset_index(drop=True))
                        ind_df = reconstruct_fid_from_iid(ind_df)
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = map_df
                        for k in _STATE_KEYS[3:]:
                            st.session_state[k] = None
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
        het_sd = st.slider("Heterozygotie sigma", 1.0, 5.0,
                           DEFAULT_THRESHOLDS["het_sd"], 0.1,
                           key="sb_het_sd")
        st.divider()
        ld_r2 = st.slider("LD Pruning r2", 0.05, 0.5,
                          DEFAULT_THRESHOLDS["ld_r2"], 0.05,
                          key="sb_ld_r2")

    if not has_data():
        st.info("Generez un jeu de demo ou importez PED/MAP.")
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["Apercu", "QC", "LD Pruning", "Structure",
                    "GWAS", "Selection", "Demographie"])

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
                    for k in ["gt_pruned", "snp_pruned", "pca_scores",
                              "mds_coords", "king_matrix", "gwas_results",
                              "fst", "roh_df", "froh", "ne_dict"]:
                        st.session_state[k] = None
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

    with tabs[2]:
        st.subheader("LD Pruning")
        step_header("4.5", "Elagage par desequilibre de liaison",
                    "Retire les SNPs correles (r2 > seuil) fenetre 50.",
                    "PCA et GWAS supposent marqueurs independants.",
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
        step_header("5.3-5.4", "PCA, MDS, KING",
                    "PCA, MDS (IBS), matrice KING.",
                    "Visualiser structure + calculer parente.",
                    "Nuages 2D + matrice de parente.")

        if not has_qc():
            st.warning("Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)

            if st.button("Lancer PCA + MDS + KING", type="primary",
                         key="struct_btn_run"):
                try:
                    with st.spinner("PCA..."):
                        s_, v_ = pca_analysis(gt_use, 10)
                        st.session_state.pca_scores = s_
                        st.session_state.pca_var = v_
                    with st.spinner("MDS..."):
                        st.session_state.mds_coords = mds_analysis(gt_use, 5)
                    with st.spinner("KING..."):
                        st.session_state.king_matrix = king_kinship(gt_use)
                    st.success("Termine.")
                except Exception as e:
                    st.error("Erreur : " + str(e))

            if st.session_state.pca_scores is not None:
                labels = _fid_array()
                st.plotly_chart(
                    plot_pca(st.session_state.pca_scores,
                             st.session_state.pca_var, labels),
                    use_container_width=True, key="struct_plot_pca")

            if st.session_state.mds_coords is not None:
                labels = _fid_array()
                st.plotly_chart(
                    plot_mds(st.session_state.mds_coords, labels),
                    use_container_width=True, key="struct_plot_mds")

            if st.session_state.king_matrix is not None:
                st.info("Matrice KING calculee : " +
                        str(st.session_state.king_matrix.shape))

    with tabs[4]:
        st.subheader("GWAS - Association genotype phenotype")
        step_header("Extension", "GWAS avec LMM (EMMAX-like)",
                    "Association SNP par SNP, controle par PCs et K.",
                    "Identifier variants associes.",
                    "Manhattan + QQ + lambdaGC + LocusZoom.")

        if not has_qc():
            st.warning("Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            snp_use = (st.session_state.snp_pruned if has_pruned()
                       else st.session_state.snp_filt)

            st.markdown("### 1. Charger les phenotypes")
            pheno_file = st.file_uploader(
                "Fichier phenotypes (CSV / TSV / TXT / .gz)",
                type=["csv", "tsv", "txt", "gz"], key="gwas_pheno_file")

            col_a, col_b = st.columns(2)
            with col_a:
                if pheno_file is not None:
                    if st.button("Charger phenotypes",
                                 key="gwas_btn_load_pheno",
                                 use_container_width=True):
                        try:
                            raw = pheno_file.read()
                            df_p = load_phenotypes_file(raw,
                                                        pheno_file.name)
                            merged, iid_col = match_phenotypes(
                                st.session_state.ind_filt, df_p)
                            st.session_state.pheno_matched = merged
                            st.success(str(len(df_p)) +
                                       " lignes, ID : " + str(iid_col))
                        except Exception as e:
                            st.error("Erreur : " + str(e))

            with col_b:
                if st.button("Simuler un phenotype demo",
                             key="gwas_btn_demo_pheno",
                             use_container_width=True):
                    ind = st.session_state.ind_filt
                    n = len(ind)
                    rng = np.random.default_rng(42)
                    if (st.session_state.pca_scores is not None
                            and len(st.session_state.pca_scores) == n):
                        trait = (st.session_state.pca_scores[:, 0] * 2.0
                                 + rng.normal(0, 0.5, n))
                    else:
                        trait = rng.normal(0, 1, n)
                    merged = pd.DataFrame({
                        "FID": ind["FID"].astype(str).values,
                        "IID": ind["IID"].astype(str).values,
                        "trait_simu": trait})
                    st.session_state.pheno_matched = merged
                    st.success("Phenotype cree.")

            if st.session_state.pheno_matched is not None:
                merged = st.session_state.pheno_matched
                st.dataframe(merged.head(10), use_container_width=True)

                num_cols = [c for c in merged.columns
                            if c not in ("FID", "IID")
                            and pd.api.types.is_numeric_dtype(merged[c])]
                if num_cols:
                    st.markdown("### 2. Parametres")
                    c1, c2, c3 = st.columns(3)
                    pheno_col = c1.selectbox("Phenotype", num_cols,
                                             key="gwas_pheno_col")
                    n_pcs = c2.slider("PCs covariables", 0, 10, 5,
                                      key="gwas_n_pcs")
                    maf_thr_gwas = c3.slider("MAF min", 0.0, 0.2, 0.05,
                                             0.01, key="gwas_maf")
                    use_lmm = st.checkbox(
                        "LMM (recommande)", value=True,
                        key="gwas_use_lmm")

                    covs = np.empty((len(merged), 0))
                    if (n_pcs > 0 and
                            st.session_state.pca_scores is not None):
                        pcs = st.session_state.pca_scores
                        if pcs.shape[0] == len(merged):
                            covs = pcs[:, :n_pcs]

                    st.markdown("### 3. Lancer")
                    if st.button("Lancer GWAS", type="primary",
                                 use_container_width=True,
                                 key="gwas_btn_run"):
                        try:
                            pheno_vec = merged[pheno_col].to_numpy(
                                dtype=float)
                            K_use = (st.session_state.king_matrix
                                     if st.session_state.king_matrix
                                     is not None
                                     else np.eye(len(merged)))
                            with st.spinner("GWAS..."):
                                res = run_gwas_lmm(
                                    gt_use, pheno_vec, K_use, covs,
                                    maf_min=float(maf_thr_gwas),
                                    use_lmm=bool(use_lmm))
                                res = annotate_gwas_results(res, snp_use)
                                st.session_state.gwas_results = res
                                st.session_state.gwas_pheno_name = pheno_col
                                st.session_state.gwas_lambda = \
                                    compute_lambda_gc(res["P"].values)
                            st.success(str(len(res)) + " SNPs testes.")
                        except Exception as e:
                            st.error("Erreur : " + str(e))

                    if st.session_state.gwas_results is not None:
                        res = st.session_state.gwas_results
                        if not res.empty:
                            thr = gwas_thresholds(res["P"].values)
                            lam = st.session_state.gwas_lambda

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("SNPs testes", len(res))
                            c2.metric("lambdaGC",
                                      str(round(lam, 3))
                                      if np.isfinite(lam) else "N/A")
                            c3.metric("Bonferroni",
                                      str(round(thr["bonferroni"], 10)))
                            n_sig = int((res["P"] <=
                                         thr["bonferroni"]).sum())
                            c4.metric("Hits Bonferroni", n_sig)

                            fig_m = plot_gwas_manhattan(
                                res, bonferroni=thr["bonferroni"],
                                fdr=thr.get("fdr_05"),
                                title="Manhattan - " +
                                      str(st.session_state.gwas_pheno_name))
                            if fig_m is not None:
                                st.plotly_chart(fig_m,
                                                use_container_width=True,
                                                key="gwas_plot_manhattan")

                            fig_qq = plot_gwas_qq(res["P"].values, lam)
                            if fig_qq is not None:
                                st.plotly_chart(fig_qq,
                                                use_container_width=True,
                                                key="gwas_plot_qq")

                            st.subheader("Top 20 SNPs")
                            top = res.sort_values("P").head(20).copy()
                            top["-log10P"] = -np.log10(
                                np.clip(top["P"], 1e-300, 1))
                            st.dataframe(top, use_container_width=True)

                            if not top.empty:
                                lead = st.selectbox(
                                    "SNP pour LocusZoom",
                                    top["SNP"].astype(str).tolist(),
                                    key="gwas_lead_snp")
                                row = top[
                                    top["SNP"].astype(str) == lead].iloc[0]
                                window_kb = st.slider(
                                    "Fenetre (kb)", 50, 2000, 500, 50,
                                    key="gwas_window_kb")
                                fig_lz = plot_locuszoom(
                                    res, snp_use, row["CHR"],
                                    int(row["BP"]),
                                    window_kb=window_kb)
                                if fig_lz is not None:
                                    st.plotly_chart(
                                        fig_lz,
                                        use_container_width=True,
                                        key="gwas_plot_locuszoom")

                            csv = res.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Telecharger GWAS (CSV)",
                                data=csv,
                                file_name=("gwas_" +
                                           str(st.session_state.gwas_pheno_name)
                                           + ".csv"),
                                mime="text/csv",
                                key="gwas_dl_results",
                                use_container_width=True)

    with tabs[5]:
        st.subheader("Signatures de selection")
        step_header("5.2", "FST/SNP, Manhattan, selection",
                    "FST par SNP + FST pairwise + score FST x homosite.",
                    "Identifier les regions sous selection.",
                    "Manhattan plot, matrice FST, top regions.")

        if not has_qc():
            st.warning("Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            sub1, sub2, sub3 = st.tabs(["FST/SNP", "FST pairwise",
                                        "Signatures"])

            with sub1:
                threshold_q = st.slider("Quantile outliers", 0.95, 0.9999,
                                        0.999, 0.0001, format="%.4f",
                                        key="sel_threshold_q")
                if st.button("FST par SNP", key="sel_btn_fst",
                             use_container_width=True):
                    try:
                        with st.spinner("FST..."):
                            st.session_state.fst = fst_per_snp(
                                gt_use, _fid_array())
                        st.success("FST calcule.")
                    except Exception as e:
                        st.error("Erreur : " + str(e))
                if st.session_state.fst is not None:
                    fst_c = st.session_state.fst[
                        np.isfinite(st.session_state.fst)]
                    if len(fst_c) > 0:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("FST moyen",
                                  str(round(fst_c.mean(), 4)))
                        c2.metric("FST median",
                                  str(round(np.median(fst_c), 4)))
                        c3.metric("Top outliers",
                                  str(round(np.quantile(fst_c, threshold_q),
                                            4)))
                        fig = plot_manhattan(
                            st.session_state.fst,
                            st.session_state.snp_filt["CHR"].values,
                            threshold_q=threshold_q)
                        if fig:
                            st.plotly_chart(fig,
                                            use_container_width=True,
                                            key="sel_plot_manhattan")

                        st.divider()
                        st.subheader("Interpretation IA FST")
                        fst_interp = interpret_fst(fst_c)
                        st.success("**" + str(fst_interp["verdict"]) +
                                   "** - " + str(fst_interp["explanation"]))
                        st.json(fst_interp["metrics"])
                        st.info(fst_interp["recommendation"])

            with sub2:
                if st.button("FST pairwise", key="sel_btn_fstp",
                             use_container_width=True):
                    try:
                        with st.spinner("FST pairwise..."):
                            mat, pops = fst_pairwise(gt_use, _fid_array())
                            st.session_state.fst_pairwise_matrix = mat
                            st.session_state.fst_pops = pops
                        st.success("Matrice calculee.")
                    except Exception as e:
                        st.error("Erreur : " + str(e))
                if st.session_state.fst_pairwise_matrix is not None:
                    st.plotly_chart(
                        plot_fst_pairwise(
                            st.session_state.fst_pairwise_matrix,
                            st.session_state.fst_pops),
                        use_container_width=True,
                        key="sel_plot_fst_pairwise")

            with sub3:
                if st.button("Detecter signatures",
                             key="sel_btn_signatures",
                             use_container_width=True):
                    try:
                        with st.spinner("Analyse FST + homosite..."):
                            sel_df = selection_signatures(
                                gt_use, st.session_state.snp_filt,
                                _fid_array())
                            st.session_state.selection_df = sel_df
                        st.success("Signatures detectees.")
                    except Exception as e:
                        st.error("Erreur : " + str(e))
                if st.session_state.selection_df is not None:
                    st.subheader("Top 20 regions sous selection")
                    st.dataframe(
                        st.session_state.selection_df.head(20),
                        use_container_width=True, key="sel_df_top20")

    with tabs[6]:
        st.subheader("Demographie LD decay, ROH, Ne")
        step_header("6.1, 6.5", "LD decay, ROH, SNeP",
                    "LD distance, ROH, Ne historique.",
                    "Reconstituer histoire demographique.",
                    "Courbes + histogrammes.")

        if not has_qc():
            st.warning("Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            sub1, sub2, sub3 = st.tabs(["LD decay", "ROH", "Ne (SNeP)"])

            with sub1:
                if st.button("LD decay", key="demo_btn_ld",
                             use_container_width=True):
                    try:
                        with st.spinner("LD decay..."):
                            st.session_state.ld_df = ld_decay(
                                gt_use,
                                st.session_state.snp_filt["BP"].values,
                                max_kb=1000, max_snp=1000)
                        st.success(str(len(st.session_state.ld_df)) +
                                   " paires")
                    except Exception as e:
                        st.error("Erreur : " + str(e))
                if (st.session_state.ld_df is not None
                        and not st.session_state.ld_df.empty):
                    st.plotly_chart(
                        plot_ld_decay(st.session_state.ld_df),
                        use_container_width=True,
                        key="demo_plot_ld_decay")

            with sub2:
                c1, c2 = st.columns(2)
                min_snps_roh = c1.slider("Min SNPs par ROH", 5, 200, 30, 5,
                                         key="demo_min_snps_roh")
                min_kb_roh = c2.slider("Longueur min (kb)", 50, 5000,
                                       500, 50, key="demo_min_kb_roh")
                if st.button("Detecter les ROH", key="demo_btn_roh",
                             use_container_width=True):
                    try:
                        with st.spinner("Detection ROH..."):
                            snp_json = (st.session_state.snp_filt[
                                ["CHR", "BP"]].to_json())
                            roh_df, froh = detect_roh(
                                gt_use, snp_json,
                                min_snps=int(min_snps_roh),
                                min_kb=float(min_kb_roh))
                            st.session_state.roh_df = roh_df
                            st.session_state.froh = froh
                        st.success(str(len(roh_df)) + " ROH")
                    except Exception as e:
                        st.error("Erreur : " + str(e))
                if st.session_state.froh is not None:
                    froh = st.session_state.froh
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FROH moyen", str(round(np.mean(froh), 4)))
                    c2.metric("FROH median", str(round(np.median(froh), 4)))
                    c3.metric("Total ROH", len(st.session_state.roh_df))
                    labels = _fid_array()
                    st.plotly_chart(
                        plot_roh_histogram(froh, labels),
                        use_container_width=True,
                        key="demo_plot_roh_hist")

                    st.divider()
                    st.subheader("Interpretation IA ROH")
                    roh_interp = interpret_roh(froh)
                    st.success("**" + str(roh_interp["verdict"]) +
                               "** - " + str(roh_interp["explanation"]))
                    st.json(roh_interp["metrics"])
                    st.info(roh_interp["recommendation"])

            with sub3:
                st.markdown("**SNeP-like** : Ne historique par race.")
                if st.button("Estimer Ne par race", key="demo_btn_ne",
                             use_container_width=True):
                    try:
                        with st.spinner("Estimation Ne..."):
                            ne_dict = {}
                            pops = (st.session_state.ind_filt["FID"]
                                    .unique())
                            for fid in pops:
                                mask = (st.session_state.ind_filt["FID"]
                                        == fid).values
                                sub = gt_use[mask]
                                if sub.shape[0] < 5:
                                    continue
                                ne_df = estimate_ne_historical(
                                    sub, st.session_state.snp_filt)
                                ne_dict[str(fid)] = ne_df
                            st.session_state.ne_dict = ne_dict
                        st.success("Ne estime pour " +
                                   str(len(ne_dict)) + " races.")
                    except Exception as e:
                        st.error("Erreur : " + str(e))
                if st.session_state.ne_dict:
                    st.plotly_chart(
                        plot_ne_curves(st.session_state.ne_dict),
                        use_container_width=True,
                        key="demo_plot_ne_curves")
                    st.warning("Interpretation RELATIVE uniquement.")


if __name__ == "__main__":
    main()
