#!/usr/bin/env python3
# iceid_train_infer_check.py - v21 (Definitive, Scalable)
# -----------------------------------------------------------------------------
# DESCRIPTION:
#   The definitive, gold-standard record linkage pipeline, redesigned for
#   scalability and memory safety on massive datasets.
#
# CHANGES (v21 - Definitive):
# - **New Scalable Workflow:** Replaced the memory-intensive batch processing
#   with a record-by-record, streaming workflow. The script iterates through
#   each source record, generates candidates on the fly using pre-built
#   blocking indexes, and performs inference, preventing memory crashes.
# - **Constant Progress Feedback:** The main inference loop now has a clear
#   progress bar over the ~984k source records, ensuring the script always
#   provides feedback and never appears frozen.
# - **Memory-Safe Outputs:** All outputs, including links for clustering and the
#   final Top-20 matches, are generated in a memory-safe way.
# - **Maintained Gold Standard Features:** This version keeps all advanced
#   features: robust multi-pass blocking, rich comparison-based feature
#   engineering, and the full six-model ensemble.
# -----------------------------------------------------------------------------

import os, gc, json, itertools, logging, random, time
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, log_evaluation
from catboost import CatBoostClassifier
from tqdm import tqdm
import math
from collections import defaultdict
from jellyfish import jaro_winkler_similarity

# ─────────── CONFIG ──────────────────────────────────────────────────────────
ART        = Path("artifacts")
RAW        = Path("raw_data")
MODELDIR   = Path("models_ensemble_advanced"); MODELDIR.mkdir(exist_ok=True)
OUTDIR     = Path("deploy_out"); OUTDIR.mkdir(exist_ok=True)

# --- Main settings ---
RNG         = 42
SAMPLE_FRAC = 1.0
SKIP_TRAINING = False
TOP_K       = 20

# --- Model & Data Params ---
NEG_PER     = 2
TREES       = 250
RF_BATCH_SIZE = 100_000
INFERENCE_CHUNK_SIZE = 10_000 # Process and write results in chunks of this many source records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

class Timer:
    def __init__(self, msg):  self.msg, self.t0 = msg, time.perf_counter()
    def __enter__(self):     logging.info(f"▶ {self.msg}"); return self
    def __exit__(self, *_):  logging.info(f"⏱ {self.msg}: {time.perf_counter()-self.t0:.1f}s")

# ─────────── HELPERS ─────────────────────────────────────────────────────────
def soundex(name):
    if not isinstance(name, str) or not name: return ""
    name = name.upper(); s_code = name[0]
    last_code = {'B': 1, 'F': 1, 'P': 1, 'V': 1, 'C': 2, 'G': 2, 'J': 2, 'K': 2, 'Q': 2, 'S': 2, 'X': 2, 'Z': 2, 'D': 3, 'T': 3, 'L': 4, 'M': 5, 'N': 5, 'R': 6}.get(name[0], 0)
    for char in name[1:]:
        code = {'B': 1, 'F': 1, 'P': 1, 'V': 1, 'C': 2, 'G': 2, 'J': 2, 'K': 2, 'Q': 2, 'S': 2, 'X': 2, 'Z': 2, 'D': 3, 'T': 3, 'L': 4, 'M': 5, 'N': 5, 'R': 6}.get(char, 0)
        if code and code != last_code: s_code += str(code)
        last_code = code
    return (s_code + "000")[:4]

def create_comparison_features(rec1, rec2):
    return np.array([
        jaro_winkler_similarity(rec1['gn'], rec2['gn']),
        jaro_winkler_similarity(rec1['pn_sn'], rec2['pn_sn']),
        abs(rec1['birth'] - rec2['birth']) if rec1['birth'] > 0 and rec2['birth'] > 0 else -1,
        abs(rec1['heim'] - rec2['heim']) if rec1['heim'] > 0 and rec2['heim'] > 0 else -1,
        1 if rec1['sex'] == rec2['sex'] and rec1['sex'] != -1 else 0,
        1 if rec1['gn_initial'] == rec2['gn_initial'] and rec1['gn_initial'] != '' else 0,
        1 if rec1['parish_id'] and rec2['parish_id'] and rec1['parish_id'] == rec2['parish_id'] else 0,
        1 if rec1['district_id'] and rec2['district_id'] and rec1['district_id'] == rec2['district_id'] else 0,
        1 if rec1['county_id'] and rec2['county_id'] and rec1['county_id'] == rec2['county_id'] else 0,
    ])


def filter_pairs_by_mode(pairs, df, mode):
    """Return only within-census or across-census pairs."""
    if mode == "within":
        return [(a,b) for a,b in pairs
                if df.at[a,'manntal'] == df.at[b,'manntal']]
    else:
        return [(a,b) for a,b in pairs
                if df.at[a,'manntal'] != df.at[b,'manntal']]


def load_data(sample_frac):
    with Timer("Load and clean raw data files"):
        rows = pd.read_csv(ART / "row_labels.csv", dtype={"row_id": str})
        if sample_frac < 1.0:
            rng = np.random.RandomState(RNG); k = max(1, int(rows.shape[0] * sample_frac))
            idx = np.sort(rng.choice(rows.shape[0], size=k, replace=False))
            rows_sample = rows.iloc[idx].reset_index(drop=True)
            logging.info(f"→ Sampled {len(rows_sample):,} rows ({sample_frac:.1%})")
        else:
            rows_sample = rows
            logging.info(f"→ Using full dataset ({len(rows_sample):,} rows)")
        
        ids = rows_sample["row_id"].astype(str).values
        labels = rows_sample["person"].fillna(-1).astype(int).values
        
        people_cols = ["id", "heimild", "birthyear", "sex", "first_name", "patronym", "surname"]
        people = pd.read_csv(RAW / "people.csv", usecols=people_cols, dtype={"id": str, "first_name":str, "patronym":str, "surname":str}, low_memory=False).set_index("id").reindex(ids)
        for col in ["first_name", "patronym", "surname"]: people[col] = people[col].fillna('')
        
        mann_cols = ["id","manntal","bi_sokn","bi_hreppur","bi_sysla"]
        mann = pd.read_csv(RAW / "manntol_einstaklingar_new.csv", usecols=mann_cols, dtype={"id": str}, low_memory=False).set_index("id").reindex(ids)
        
        df = pd.DataFrame({
            "id": ids, "label": labels, "heim": people["heimild"].fillna(-1).astype(int).values, 
            "birth": pd.to_numeric(people["birthyear"], errors='coerce').fillna(0).astype(int).values,
            "sex": people["sex"].str.lower().str.strip().map({"karl": 1, "kona": 0, "karl.":1, "kona.":0}).fillna(-1).astype(int).values,
            "parish_id": mann["bi_sokn"].fillna('').values,
            "district_id": mann["bi_hreppur"].fillna('').values,
            "county_id": mann["bi_sysla"].fillna('').values,
            "pn": people["patronym"].values, "sn": people["surname"].values, "gn": people["first_name"].values
        })
        df['pn_sn'] = df['pn'].where(df['pn'] != '', df['sn'])
        df['gn_initial'] = df['gn'].str[0:1].fillna('')
        df['pn_sn_soundex'] = df['pn_sn'].apply(soundex)
        df['birth_decade'] = (df['birth'] // 10) * 10
    return df

def predict_ensemble_proba(ensemble, X_data):
    all_probs = []
    for model in ensemble:
        if isinstance(model, list):
            all_probs.append(np.array([m.predict_proba(X_data)[:, 1] for m in model]).mean(axis=0))
        else:
            all_probs.append(model.predict_proba(X_data)[:, 1])
    return np.array(all_probs).mean(axis=0)

def connected_components(n, links):
    parent = list(range(n))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i]); return parent[i]
    for a,b,_ in tqdm(links, desc="Building clusters"):
        ra, rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra
    root2cid = {r: i for i, r in enumerate(pd.Series(parent).unique())}
    return [root2cid.get(find(i), -1) for i in range(n)]

def evaluate_clusters(labels, clusters, ids):
    df = pd.DataFrame({'label': labels, 'cluster': clusters, 'id': ids})
    df = df[df.label != -1].copy()
    if df.empty: logging.warning("No labeled data to evaluate clusters against."); return
    
    with Timer("Calculating pairwise evaluation metrics"):
        predicted_pairs = set(itertools.chain.from_iterable(itertools.combinations(sorted(g['id'].tolist()), 2) for _, g in tqdm(df.groupby('cluster'), desc="Generating predicted pairs") if len(g) > 1))
        actual_pairs = set(itertools.chain.from_iterable(itertools.combinations(sorted(g['id'].tolist()), 2) for _, g in tqdm(df.groupby('label'), desc="Generating actual pairs") if len(g) > 1))
        
        tp = len(predicted_pairs.intersection(actual_pairs)); fp = len(predicted_pairs.difference(actual_pairs)); fn = len(actual_pairs.difference(predicted_pairs))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0; recall = tp / (tp + fn) if (tp + fn) > 0 else 0; f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    logging.info(f"--- Cluster Evaluation ---\nPairwise Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\nTP: {tp:,}, FP: {fp:,}, FN: {fn:,}")

# ─────────── MAIN ────────────────────────────────────────────────────────────
def main():
    logging.info("Starting Gold Standard Record Linkage Pipeline.")
    df = load_data(SAMPLE_FRAC)
    N = len(df)
    mdir = MODELDIR / "combined"; mdir.mkdir(exist_ok=True)
    all_model_names = ["rf", "catboost", "xgboost", "lightgbm", "mlp", "logistic"]

    if not SKIP_TRAINING:
        with Timer("Generate candidate pairs for training"):
            candidate_pairs = set()
            blocking_passes = { "Soundex_Initial_Decade_Sex": ["pn_sn_soundex", "gn_initial", "birth_decade", "sex"], "Parish_Initial_Decade_Sex": ["parish_id", "gn_initial", "birth_decade", "sex"] }
            for pass_name, block_cols in blocking_passes.items():
                logging.info(f"Running blocking pass for training: {pass_name}")
                pass_df = df.copy(); [pass_df.drop(pass_df[pass_df[c] == ''].index, inplace=True) for c in block_cols if pass_df[c].dtype == 'object']; [pass_df.drop(pass_df[pass_df[c] <= 0].index, inplace=True) for c in block_cols if pass_df[c].dtype in ['int64','int32']]
                if pass_df.empty: continue
                groups = pass_df.groupby(block_cols)
                for _, group in tqdm(groups, desc=f"Blocking on {pass_name}", leave=False):
                    if len(group) < 2: continue
                    for a, b in itertools.combinations(group.index, 2): candidate_pairs.add(tuple(sorted((a, b))))
            logging.info(f"Generated {len(candidate_pairs):,} unique candidate pairs for training.")
        
        with Timer("Create Training Data with Rich Features"):
            pos_pairs, neg_pairs_all = [], []
            logging.info("Filtering candidate pairs into positive/negative sets...")
            for p in tqdm(candidate_pairs, desc="Filtering pairs"):
                label1, label2 = df.at[p[0],'label'], df.at[p[1],'label']
                if label1 != -1 and label2 != -1:
                    if label1 == label2: pos_pairs.append(p)
                    else: neg_pairs_all.append(p)
            
            if not pos_pairs: logging.error("FATAL: No positive pairs found. Cannot train."); return
            neg_pairs = random.sample(neg_pairs_all, min(len(neg_pairs_all), len(pos_pairs) * NEG_PER))
            training_pairs = pos_pairs + neg_pairs
            if not training_pairs: logging.error("FATAL: No training pairs created."); return
            y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
            feature_vectors = [create_comparison_features(df.iloc[a], df.iloc[b]) for a,b in tqdm(training_pairs, "Generating training features")]
            X_train = np.array(feature_vectors)
        
        with Timer("Full Training Pipeline"):
            X_tr, X_te, y_tr, y_te = train_test_split(X_train, y, test_size=0.2, stratify=y, random_state=RNG)
            
            logging.info("Training RandomForest..."); rf_models = []
            with tqdm(total=X_tr.shape[0], desc=f"Training RF", unit="pairs") as pbar:
                for i in range(0, X_tr.shape[0], RF_BATCH_SIZE):
                    rf = RandomForestClassifier(n_estimators=TREES, n_jobs=4, random_state=RNG+i).fit(X_tr[i:i+RF_BATCH_SIZE], y_tr[i:i+RF_BATCH_SIZE])
                    rf_models.append(rf); pbar.update(X_tr[i:i+RF_BATCH_SIZE].shape[0])
            dump(rf_models, mdir / "rf.joblib"); logging.info("✓ RandomForest training complete.")

            logging.info("Training CatBoost..."); cat = CatBoostClassifier(iterations=TREES, task_type="GPU", devices="0", gpu_ram_part=0.5, random_state=RNG).fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=50); dump(cat, mdir / "catboost.joblib"); del cat; gc.collect(); logging.info("✓ CatBoost training complete.")
            logging.info("Training XGBoost..."); xgbc = XGBClassifier(n_estimators=TREES, tree_method='hist', device='cuda', random_state=RNG).fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50); dump(xgbc, mdir / "xgboost.joblib"); del xgbc; gc.collect(); logging.info("✓ XGBoost training complete.")
            logging.info("Training LightGBM..."); lgbm = LGBMClassifier(n_estimators=TREES, device='gpu', verbosity=-1, random_state=RNG).fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[log_evaluation(50)]); dump(lgbm, mdir / "lightgbm.joblib"); del lgbm; gc.collect(); logging.info("✓ LightGBM training complete.")
            logging.info("Training MLP..."); mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RNG).fit(X_tr, y_tr); dump(mlp, mdir / "mlp.joblib"); del mlp; gc.collect(); logging.info("✓ MLP training complete.")
            logging.info("Training Logistic Regression..."); lr = LogisticRegression(max_iter=1000, random_state=RNG, n_jobs=-1).fit(X_tr, y_tr); dump(lr, mdir / "logistic.joblib"); del lr; gc.collect(); logging.info("✓ Logistic Regression training complete.")

            logging.info("Evaluating final ensemble...");
            ensemble = [load(mdir / f"{name}.joblib") for name in all_model_names]
            probs = predict_ensemble_proba(ensemble, X_te)
            thr = max(np.linspace(0.01, 0.99, 99), key=lambda t: float(f1_score(y_te, probs >= t)))
            f1_val = f1_score(y_te, probs >= thr); auc = roc_auc_score(y_te, probs)
            logging.info(f"Final Validation: F1={f1_val:.4f} AUROC={auc:.4f} @ Threshold={thr:.2f}")
            meta = {"thr": thr, "f1": f1_val, "auroc": auc}; open(mdir / "meta.json", "w").write(json.dumps(meta, indent=2))

    with Timer("Run Scalable Inference, Clustering, and Top-K Generation"):
        logging.info("Building blocking indexes for fast lookup...")
        blocking_indexes = {}
        blocking_passes = { "Soundex_Initial_Decade_Sex": ["pn_sn_soundex", "gn_initial", "birth_decade", "sex"], "Parish_Initial_Decade_Sex": ["parish_id", "gn_initial", "birth_decade", "sex"], "Parish_BirthYear": ["parish_id", "birth"], "Soundex_BirthYear": ["pn_sn_soundex", "birth"] }
        for pass_name, block_cols in blocking_passes.items():
            index = defaultdict(list)
            pass_df = df.copy(); [pass_df.drop(pass_df[pass_df[c] == ''].index, inplace=True) for c in block_cols if pass_df[c].dtype == 'object']; [pass_df.drop(pass_df[pass_df[c] <= 0].index, inplace=True) for c in block_cols if pass_df[c].dtype in ['int64','int32']]
            for i, row in tqdm(pass_df.iterrows(), total=len(pass_df), desc=f"Building index for {pass_name}"):
                key = tuple(row[block_cols])
                index[key].append(i)
            blocking_indexes[pass_name] = index

        if not mdir.exists(): logging.error(f"FATAL: Models not found in {mdir}."); return
        ensemble = [load(mdir / f"{name}.joblib") for name in all_model_names]
        with open(mdir / "meta.json", "r") as f: meta = json.load(f)
        threshold = meta['thr']
        
        links = []
        output_path = OUTDIR / f"top_{TOP_K}_matches_advanced.csv"
        pd.DataFrame(columns=["id", f"top_{TOP_K}_matches"]).to_csv(output_path, index=False) 

        for chunk_start in tqdm(range(0, N, INFERENCE_CHUNK_SIZE), desc="Main Record-by-Record Inference"):
            chunk_end = min(chunk_start + INFERENCE_CHUNK_SIZE, N)
            chunk_df = df.iloc[chunk_start:chunk_end]
            chunk_top_k_results = {}
            chunk_links = []

            for i, rec1 in chunk_df.iterrows():
                candidate_indices = set()
                for pass_name, block_cols in blocking_passes.items():
                    key = tuple(rec1[block_cols])
                    if key in blocking_indexes[pass_name]:
                        candidate_indices.update(blocking_indexes[pass_name][key])
                candidate_indices.discard(i)
                if not candidate_indices: continue

                sub_cands = [(i, j) for j in candidate_indices if i < j]
                if not sub_cands: continue

                feature_vectors = [create_comparison_features(rec1, df.iloc[j]) for _, j in sub_cands]
                probs = predict_ensemble_proba(ensemble, np.array(feature_vectors))
                
                # Store all predictions for this record to find its Top-K
                all_rec1_preds = [(sub_cands[k][1], p) for k,p in enumerate(probs)]
                all_rec1_preds.sort(key=lambda x: x[1], reverse=True)

                # Save links for clustering
                for match_idx, prob in all_rec1_preds:
                    if prob >= threshold: chunk_links.append((i, match_idx, prob))

                # Save Top-K results
                match_list = [[df.at[m_idx, 'id'], round(p, 4)] for m_idx, p in all_rec1_preds[:TOP_K]]
                chunk_top_k_results[rec1['id']] = json.dumps(match_list)

            # --- Append chunk results to disk ---
            if chunk_top_k_results:
                chunk_output_df = pd.DataFrame.from_dict(chunk_top_k_results, orient='index', columns=[f'top_{TOP_K}_matches'])
                chunk_output_df.index.name = 'id'
                chunk_output_df.to_csv(output_path, mode='a', header=False)
            links.extend(chunk_links)
            gc.collect()

        # --- Final Clustering and Evaluation ---
        logging.info(f"Generated {len(links):,} total links above threshold {threshold:.2f} for clustering.")
        pd.DataFrame(links, columns=["row1_idx", "row2_idx", "prob"]).to_csv(OUTDIR / "links_advanced.csv", index=False)
        clusters = connected_components(N, links)
        pd.DataFrame({"id": df['id'], "cluster_id": clusters}).to_csv(OUTDIR / "clusters_advanced.csv", index=False)
        evaluate_clusters(df['label'].values, clusters, df['id'].values)

    logging.info("✓ All steps complete.")

if __name__ == "__main__":
    main()
