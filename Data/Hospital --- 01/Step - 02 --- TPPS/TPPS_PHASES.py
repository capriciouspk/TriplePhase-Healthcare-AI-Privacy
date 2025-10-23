#!/usr/bin/env python3
"""
Fixed end-to-end processing:
 - encryption (Fernet)
 - local DP noise added to soft labels (Gaussian mechanism)
 - synthetic table generation (GMM numeric + empirical categorical + Dirichlet for soft labels)
 - anonymization (adaptive quantile binning + k-anonymity via suppression with fallback)
Also sets LOKY_MAX_CPU_COUNT early to avoid loky/joblib physical-core error.
"""

import os
# ---- FIX for loky/joblib physical-core detection (must be before sklearn/joblib imports) ---
# Use environment override if present; otherwise default to logical core count
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import math
import json
import numpy as np
import pandas as pd

from cryptography.fernet import Fernet
from sklearn.mixture import GaussianMixture
# Note: we deliberately avoid KBinsDiscretizer to silence the quantile_method warning and bins warnings.
# We'll use pandas.qcut per-column (duplicates='drop') which is robust on small datasets.

# -------------------------
# PARAMETERS (tune these)
# -------------------------
INPUT_CSV = "combined_phase_fkd.csv"   # path to your uploaded softlabels CSV
OUT_DIR = "TPPS-data"                  # output directory
os.makedirs(OUT_DIR, exist_ok=True)

ENCRYPTED_FILE = os.path.join(OUT_DIR, "softlabels_encrypted.enc")
ENCRYPTION_KEY_FILE = os.path.join(OUT_DIR, "softlabels_encryption_key.key")
NOISED_CSV = os.path.join(OUT_DIR, "softlabels_noised.csv")
SYNTHETIC_CSV = os.path.join(OUT_DIR, "softlabels_synthetic.csv")
ANONYMIZED_CSV = os.path.join(OUT_DIR, "softlabels_anonymized.csv")
README_JSON = os.path.join(OUT_DIR, "softlabels_processing_readme.json")

# DP params (Gaussian mech)
EPSILON = 1.0      # privacy budget; increase (e.g., 2.0 or 5.0) for better utility
DELTA = 1e-5
SENSITIVITY = 1.0  # conservative per-row L2 sensitivity

# Synthetic params
GMM_COMPONENTS = 5
SYNTH_FACTOR = 1.0   # synthetic size = SYNTH_FACTOR * original_n

# Anonymization params
INITIAL_NUM_BINS = 5
K_ANON = 5
MIN_BINS = 2

# RNG
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# -------------------------
# Helper functions
# -------------------------
def normalize_rows(mat, eps=1e-12):
    mat = np.asarray(mat, dtype=float)
    sums = mat.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    return mat / np.maximum(sums, eps)

def adaptive_qcut_column(series, q):
    """
    Robust quantile binning for a single column. Uses pandas.qcut with duplicates='drop'.
    Returns integer bins (0..k-1). If qcut cannot create q bins because of duplicates,
    it reduces bins until it can or returns a single constant column.
    """
    x = series.dropna()
    if x.empty:
        return pd.Series([np.nan] * len(series), index=series.index)
    # Max bins limited by number of unique values
    uniq = x.unique().shape[0]
    max_q = min(q, uniq)
    if max_q < 2:
        # all identical or single unique value -> return zeros
        return pd.Series([0] * len(series), index=series.index)
    # Try qcut and fall back by decreasing q
    for q_try in range(max_q, 1, -1):
        try:
            binned = pd.qcut(series, q=q_try, labels=False, duplicates='drop')
            # qcut may still produce fewer bins if duplicates dropped; ensure non-NaNs exist
            if binned.notna().sum() > 0:
                # fill NaNs (from extreme values) by nearest bin (forward/backward fill)
                binned = binned.astype('float').fillna(method='ffill').fillna(method='bfill')
                return binned.astype('Int64')  # pandas nullable int
        except Exception:
            continue
    # If all fails, return zeros column
    return pd.Series([0] * len(series), index=series.index)

# -------------------------
# Load input CSV
# -------------------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
n_rows, n_cols = df.shape

print(f"Loaded input: {INPUT_CSV}  (rows={n_rows}, cols={n_cols})")

# -------------------------
# 1) Encryption (Fernet symmetric)
# -------------------------
key = Fernet.generate_key()
fernet = Fernet(key)
with open(INPUT_CSV, "rb") as f:
    original_bytes = f.read()
encrypted = fernet.encrypt(original_bytes)
with open(ENCRYPTED_FILE, "wb") as f:
    f.write(encrypted)
with open(ENCRYPTION_KEY_FILE, "wb") as f:
    f.write(key)
print(f"Encrypted file written to: {ENCRYPTED_FILE}")
print(f"Encryption key written to: {ENCRYPTION_KEY_FILE}")

# -------------------------
# 2) Noised data (Local Gaussian DP on soft-label columns)
# -------------------------
# Detect soft-label columns heuristically
soft_cols = [c for c in df.columns if ('soft' in c.lower() or 'prob' in c.lower())]
if not soft_cols:
    # fallback to last two numeric columns
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    soft_cols = numeric_cols_all[-2:] if len(numeric_cols_all) >= 2 else numeric_cols_all

if len(soft_cols) == 0:
    raise RuntimeError("Cannot detect soft-label columns; please set soft_cols manually.")

print("Detected soft-label columns:", soft_cols)

noised_df = df.copy()
probs = noised_df[soft_cols].astype(float).to_numpy()
probs = normalize_rows(probs)

# compute Gaussian sigma (standard formula)
sigma = SENSITIVITY * math.sqrt(2 * math.log(1.25 / DELTA)) / EPSILON

# draw noise and apply
noise = rng.normal(loc=0.0, scale=sigma, size=probs.shape)
noisy = probs + noise
# Clip negative values and renormalize
noisy = np.clip(noisy, 0.0, None)
noisy = normalize_rows(noisy)
noised_df[soft_cols] = noisy
noised_df.to_csv(NOISED_CSV, index=False)
print(f"Noised CSV saved: {NOISED_CSV} (epsilon={EPSILON}, delta={DELTA}, sigma={sigma:.6f})")

# -------------------------
# 3) Synthetic data generation
#    - numeric: GMM if possible, otherwise gaussian around means
#    - categorical: empirical sampling
#    - soft labels: Dirichlet sampling (keeps probability-structure)
# -------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# treat small-unique numeric columns as categorical candidates
categorical_candidates = []
for c in numeric_cols:
    if c in soft_cols:
        continue
    nunique = int(df[c].nunique(dropna=True))
    if nunique <= 10:
        categorical_candidates.append(c)

synth_categorical_cols = categorical_candidates.copy()
synth_numeric_cols = [c for c in numeric_cols if c not in synth_categorical_cols and c not in soft_cols]

# Prepare numeric sampling
if len(synth_numeric_cols) > 0:
    X_num = df[synth_numeric_cols].astype(float).fillna(df[synth_numeric_cols].mean()).values
    n_components = min(GMM_COMPONENTS, max(1, X_num.shape[0] // 5))
    n_samples = max(1, int(SYNTH_FACTOR * n_rows))
    sampled_num_df = pd.DataFrame()
    if X_num.shape[0] >= n_components and X_num.shape[0] >= 2:
        gmm = GaussianMixture(n_components=n_components, random_state=RNG_SEED)
        gmm.fit(X_num)
        sampled_num = gmm.sample(n_samples)[0]
        sampled_num_df = pd.DataFrame(sampled_num, columns=synth_numeric_cols)
    else:
        # fallback: sample normals around empirical mean/std
        means = np.nanmean(X_num, axis=0)
        stds = np.nanstd(X_num, axis=0) + 1e-6
        sampled_num = rng.normal(loc=means, scale=stds, size=(n_samples, X_num.shape[1]))
        sampled_num_df = pd.DataFrame(sampled_num, columns=synth_numeric_cols)
else:
    n_samples = max(1, int(SYNTH_FACTOR * n_rows))
    sampled_num_df = pd.DataFrame(index=range(n_samples))

# Categorical sampling (empirical)
sampled_cat_df = pd.DataFrame(index=range(n_samples))
for c in synth_categorical_cols:
    probs_cat = df[c].value_counts(normalize=True, dropna=True)
    values = probs_cat.index.tolist()
    pvals = probs_cat.values
    sampled_cat_df[c] = rng.choice(values, size=n_samples, p=pvals)

# Soft-labels: sample from Dirichlet whose alpha proportional to mean soft-p vector
orig_probs = df[soft_cols].astype(float).to_numpy()
orig_probs = normalize_rows(orig_probs)
alpha = np.clip(orig_probs.mean(axis=0) * 100.0, 1e-3, None)  # concentration
sampled_probs = rng.dirichlet(alpha, size=n_samples)
sampled_probs_df = pd.DataFrame(sampled_probs, columns=soft_cols)

# Combine pieces
synth_parts = [sampled_num_df.reset_index(drop=True),
               sampled_cat_df.reset_index(drop=True),
               sampled_probs_df.reset_index(drop=True)]
synth_df = pd.concat(synth_parts, axis=1)

# Fill any missing original columns (e.g., string identifiers) by sampling empirical values
for c in df.columns:
    if c not in synth_df.columns:
        vals = df[c].dropna().unique()
        synth_df[c] = rng.choice(vals, size=n_samples) if len(vals) > 0 else np.nan

# Reorder and save
synth_df = synth_df[df.columns.tolist()]
synth_df.to_csv(SYNTHETIC_CSV, index=False)
print(f"Synthetic CSV saved: {SYNTHETIC_CSV}  (n_samples={n_samples})")

# -------------------------
# 4) Anonymization (adaptive quantile binning + suppression)
#    We try to avoid total suppression by lowering bin counts iteratively.
# -------------------------
anon_df = df.copy()
qi_candidates = ["age", "sex", "cp", "trestbps", "chol", "thalach"]
qi = [c for c in qi_candidates if c in anon_df.columns]

def anonymize_k_adaptive(df_in, qi_cols, k, initial_bins, min_bins=2):
    """
    Attempt anonymization: iteratively reduce bins until some rows remain or min_bins reached.
    Returns anonymized_df, suppressed_count, used_bins.
    """
    if not qi_cols:
        return df_in.copy(), 0, None
    used_bins = initial_bins
    df_work = df_in.copy()
    while used_bins >= min_bins:
        binned = pd.DataFrame(index=df_work.index)
        # apply per-column robust qcut
        for col in qi_cols:
            try:
                binned[col] = adaptive_qcut_column(df_work[col], used_bins)
            except Exception:
                # fallback to simple rank-based binning
                binned[col] = (pd.Series(df_work[col]).rank(method='average').fillna(0) // (len(df_work) / max(2, used_bins))).astype('Int64')
        # merge binned into df copy and compute group sizes
        merged = df_work.copy()
        for col in qi_cols:
            merged[f"__bin_{col}"] = binned[col]
        group_cols = [f"__bin_{c}" for c in qi_cols]
        group_sizes = merged.groupby(group_cols).size().reset_index(name="cnt")
        small_groups = group_sizes[group_sizes["cnt"] < k].drop(columns=["cnt"])
        if small_groups.empty:
            # no suppression needed
            # replace original qi columns with binned values (as integers)
            for col in qi_cols:
                merged[col] = merged[f"__bin_{col}"].astype('Int64')
                merged.drop(columns=[f"__bin_{col}"], inplace=True)
            return merged, 0, used_bins
        # mark suppressed rows
        mask = merged.merge(small_groups, on=group_cols, how='left', indicator=True)["_merge"] == "both"
        suppressed = mask.sum()
        kept = (~mask).sum()
        if kept > 0:
            # return the kept rows (suppression), with binned QIs
            result = merged[~mask].copy()
            for col in qi_cols:
                result[col] = result[f"__bin_{col}"].astype('Int64')
                result.drop(columns=[f"__bin_{col}"], inplace=True)
            return result.reset_index(drop=True), int(suppressed), used_bins
        # else all rows suppressed -> reduce bins and retry
        used_bins -= 1
    # if we exit loop, return empty DataFrame as last resort
    return pd.DataFrame(columns=df_in.columns), int(n_rows), used_bins

anon_result_df, suppressed_count, used_bins = anonymize_k_adaptive(anon_df, qi, K_ANON, INITIAL_NUM_BINS, MIN_BINS)
anon_result_df.to_csv(ANONYMIZED_CSV, index=False)
print(f"Anonymized CSV saved: {ANONYMIZED_CSV}  (suppressed_rows={suppressed_count}, used_bins={used_bins})")

# -------------------------
# Write README / metadata
# -------------------------
readme = {
    "input": INPUT_CSV,
    "n_rows": int(n_rows),
    "n_cols": int(n_cols),
    "outputs": {
        "encrypted_file": ENCRYPTED_FILE,
        "encryption_key_file": ENCRYPTION_KEY_FILE,
        "noised_csv": NOISED_CSV,
        "synthetic_csv": SYNTHETIC_CSV,
        "anonymized_csv": ANONYMIZED_CSV
    },
    "dp_parameters": {"epsilon": EPSILON, "delta": DELTA, "sigma_used": float(sigma)},
    "synthetic_params": {"gmm_components": int(min(GMM_COMPONENTS, max(1, (df[synth_numeric_cols].shape[0] if synth_numeric_cols else 0)))),
                         "n_samples": int(n_samples)},
    "anonymization": {"k": K_ANON, "initial_bins": INITIAL_NUM_BINS, "used_bins": used_bins, "suppressed_rows": int(suppressed_count)}
}
with open(README_JSON, "w") as f:
    json.dump(readme, f, indent=2)

print("\nDone. Summary:")
print(json.dumps(readme, indent=2))
print(f"\nFiles created in: {OUT_DIR}")
