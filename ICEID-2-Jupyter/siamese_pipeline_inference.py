#!/usr/bin/env python3
"""
Fast, robust, and evaluatable inference script for ICE-ID.
Supports safe, sharded execution and resumes from checkpoints.

CORRECT EXECUTION FLOW:
-----------------------
This script has two modes: processing and merging. You must run them separately.

1. Run all processing shards in the background.
   (Optional: Add --percentage for a dry run on a fraction of the data)

   # Example on 10% of the data with 2 workers:
   python siamese_pipeline_inference.py --percentage 10 --num-shards 2 --shard-index 1 &
   python siamese_pipeline_inference.py --percentage 10 --num-shards 2 --shard-index 2 &

2. Wait for all background jobs to complete using the shell's `wait` command.

   wait

3. Run the final merge and evaluation step.

   python siamese_pipeline_inference.py --percentage 10 --merge-only
"""
import os
import gc
import json
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict
import glob

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm
from sklearn import metrics

try:
    import cupy as cp
    CUPY_OK = True
except ModuleNotFoundError:
    cp = None
    CUPY_OK = False

import torch
from rapidfuzz.distance import JaroWinkler
from xgboost import XGBClassifier, DMatrix
from scipy import sparse
import faiss
import heapq

class Timer:
    def __init__(self, msg): self.msg, self.t0 = msg, time.perf_counter()
    def __enter__(self): logging.info(f"▶ {self.msg}"); return self
    def __exit__(self, *_): logging.info(f"⏱ {self.msg}: {time.perf_counter()-self.t0:.1f}s")


# ─── paths & constants ─────────────────────────────────────────────────────
RNG      = 42
RAW      = Path("raw_data")
ART      = Path("artifacts"); ART.mkdir(exist_ok=True)
MODELDIR = Path("models_ensemble_siamese")
OUTDIR   = Path("deploy_out"); OUTDIR.mkdir(exist_ok=True)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_INTERVAL = 1000          # flush outputs every N source rows

# ─── small helpers ─────────────────────────────────────────────────────────
load_json = lambda p: json.load(open(p, "r"))

def _merge_topk(heap_list, qid, sim, dst, k):
    h = heap_list[qid]
    if len(h) < k: heapq.heappush(h, (sim, dst))
    elif sim > h[0][0]: heapq.heapreplace(h, (sim, dst))

def soundex(name):
    if not name: return ""
    name = name.upper(); out = name[0]
    enc = {'B':1,'F':1,'P':1,'V':1,'C':2,'G':2,'J':2,'K':2,'Q':2,'S':2,'X':2,'Z':2,
           'D':3,'T':3,'L':4,'M':5,'N':5,'R':6}
    last = enc.get(out,0)
    for ch in name[1:]:
        c = enc.get(ch,0)
        if c and c != last: out += str(c)
        last = c
    return (out + "000")[:4]

def load_all_data():
    with Timer("Load and clean all data files"):
        rows = pd.read_csv(ART / "row_labels.csv", dtype={"row_id": str})
        
        # Load other files but do not set index yet
        people_raw = pd.read_csv(
            RAW / "people.csv", usecols=["id", "heimild", "birthyear", "sex", "first_name", "patronym", "surname"],
            dtype=str, low_memory=False
        )
        mann_raw = pd.read_csv(
            RAW / "manntol_einstaklingar_new.csv", usecols=["id", "bi_sokn", "bi_hreppur", "bi_sysla"],
            dtype=str, low_memory=False
        )

        # Merge dataframes based on the 'id' column
        df = rows.rename(columns={"row_id": "id"})
        df = pd.merge(df, people_raw, on="id", how="left")
        df = pd.merge(df, mann_raw, on="id", how="left")
        
        # Clean and create features
        df[["first_name", "patronym", "surname"]] = df[["first_name", "patronym", "surname"]].fillna('')
        df[["bi_sokn", "bi_hreppur", "bi_sysla"]] = df[["bi_sokn", "bi_hreppur", "bi_sysla"]].fillna('')
        
        df["label"] = df.person.fillna(-1).astype(int)
        df["census"] = df.heimild.fillna(-1).astype(int)
        df["heim"] = df.heimild.fillna(-1).astype(int)
        df["birth"] = pd.to_numeric(df.birthyear, errors='coerce').fillna(0).astype(int)
        df["sex"] = df.sex.str.lower().str.strip().map({"karl": 1, "karl.": 1, "kona": 0, "kona.": 0}).fillna(-1).astype(int)
        df = df.rename(columns={
            "bi_sokn": "parish_id", "bi_hreppur": "district_id", "bi_sysla": "county_id",
            "patronym": "pn", "surname": "sn", "first_name": "gn"
        })
        
        df["pn_sn"] = df.pn.where(df.pn != '', df.sn)
        df["pn_sn_soundex"] = df.pn_sn.apply(soundex)
        df["birth_decade"] = (df.birth // 10) * 10
        
        # Select and reorder final columns, drop intermediates
        final_cols = ["id", "label", "census", "heim", "birth", "sex", "parish_id", "district_id",
                      "county_id", "pn", "sn", "gn", "pn_sn", "pn_sn_soundex", "birth_decade"]
        df = df[final_cols]

    return df

def create_comparison_features(r1, r2):
    sim = JaroWinkler.normalized_similarity
    return np.array([
        sim(r1.gn, r2.gn), sim(r1.pn_sn, r2.pn_sn),
        abs(r1.birth - r2.birth) if r1.birth and r2.birth else -1,
        abs(r1.heim - r2.heim) if r1.heim and r2.heim else -1,
        int(r1.sex != -1 and r1.sex == r2.sex),
        int(bool(r1.parish_id) and r1.parish_id == r2.parish_id),
        int(bool(r1.district_id) and r1.district_id == r2.district_id),
        int(bool(r1.county_id) and r1.county_id == r2.county_id),
    ], dtype=float)

def score_pairs(models, feats):
    out = []
    for m in models:
        if isinstance(m, XGBClassifier):
            booster = m.get_booster()
            if CUPY_OK and DEVICE == "cuda" and cp is not None:
                booster.set_param('predictor', 'gpu_predictor')
            else:
                booster.set_param('predictor', 'cpu_predictor')
            dmatrix_feats = DMatrix(feats)
            probs = booster.predict(dmatrix_feats)
            out.append(probs)
        elif isinstance(m, list):
            out.append(np.mean([rf.predict_proba(feats)[:, 1] for rf in m], 0))
        else:
            out.append(m.predict_proba(feats)[:, 1])
    return np.mean(out, axis=0)

# ─── ENSEMBLE (shardable) ──────────────────────────────────────────────────
def run_ensemble(df, tag, start_idx, end_idx, shard_index, resume=False):
    logging.info(f"Starting ensemble stage for shard {shard_index}")
    mdir = MODELDIR / f"combined_{tag}"
    thr = load_json(mdir/"meta.json")["thr"]
    models = [load(mdir/fn) for fn in ("rf.joblib","catboost.joblib","xgboost.joblib","lightgbm.joblib","mlp.joblib","logistic.joblib")]
    
    with Timer("Build blocking index"):
        block_defs = {"soundex_decade_sex": ["pn_sn_soundex", "birth_decade", "sex"],
                      "parish_decade_sex": ["parish_id", "birth_decade", "sex"],
                      "parish_year": ["parish_id", "birth"],
                      "soundex_year": ["pn_sn_soundex", "birth"]}
        bidx = {}
        temp_df = df.reset_index()
        for cols in block_defs.values():
            mask = pd.Series(True, index=temp_df.index)
            for c in cols: mask &= temp_df[c].ne('') if temp_df[c].dtype == 'object' else temp_df[c].gt(0)
            bidx[tuple(cols)] = temp_df.loc[mask].groupby(cols)['index'].apply(list).to_dict()

    ckpt = OUTDIR/f"ensemble_links_{tag}_shard_{shard_index}.ckpt"
    links_file = OUTDIR / f"ensemble_links_{tag}_shard_{shard_index}.csv"
    if resume and links_file.exists():
        logging.info(f"Links file for shard {shard_index} found, skipping.")
        return

    if resume and ckpt.exists():
        last = int(ckpt.read_text())
        start_idx = max(start_idx, last + 1)
        logging.info(f"Resuming shard {shard_index} from index {start_idx}")

    found_links = []
    pbar = tqdm(total=end_idx - start_idx, desc=f"Shard {shard_index} ({tag})", unit="rec")
    for i in range(start_idx, end_idx):
        cand = set()
        for cols, idx_map in bidx.items():
            key = tuple(df.at[i, c] for c in cols)
            cand.update(idx_map.get(key, []))
        cand.discard(i)
        if cand:
            feats = np.vstack([create_comparison_features(df.loc[i], df.loc[j]) for j in cand])
            probs = score_pairs(models, feats)
            for j, p in zip(cand, probs):
                same = (df.census.iloc[i] == df.census.iloc[j])
                if p >= thr and ((tag=="within") == same):
                    found_links.append((i, j) if i < j else (j, i))
        pbar.update()
        if i % 1000 == 0: ckpt.write_text(str(i))
    pbar.close()
    pd.DataFrame(list(set(found_links)), columns=['src_idx', 'dst_idx']).to_csv(links_file, index=False)
    logging.info(f"Shard {shard_index} finished and saved {len(found_links)} links.")
    if ckpt.exists(): os.remove(ckpt)

# ─── SIAMESE (shardable) ───────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, d_in: int, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, d_out))
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=1)

def run_siamese(df, tag, start_idx, end_idx, shard_index):
    logging.info(f"Starting Siamese stage for shard {shard_index}")
    meta = load_json(MODELDIR / f"meta_{tag}.json"); thr = meta["thr"]; emb_dim = meta.get("emb_dim", 256)
    
    # The dataframe passed here has its original index from before sampling/sharding
    original_indices = df.index.to_numpy()
    
    full_X_sparse = sparse.load_npz(ART / "iceid_ml_ready.npz")
    X_sparse = full_X_sparse[original_indices]
    
    enc = Encoder(X_sparse.shape[1], emb_dim).to(DEVICE)
    enc.load_state_dict(torch.load(MODELDIR / f"enc_{tag}.pt", map_location=DEVICE, weights_only=True), strict=False)
    enc.eval()
    
    df = df.reset_index(drop=True)

    with Timer("Compute embeddings"):
        embed = np.zeros((len(df), emb_dim), dtype=np.float32)
        for i in tqdm(range(0, len(df), 1024), leave=False, desc="embed"):
            batch = torch.from_numpy(X_sparse[i:i+1024].toarray()).float().to(DEVICE)
            with torch.no_grad(): embed[i:i+1024] = enc(batch).cpu().numpy()
    del X_sparse, full_X_sparse; gc.collect()

    cpu_index = faiss.IndexFlatIP(emb_dim); cpu_index.add(embed)
    FAISS_GPU_AVAILABLE = hasattr(faiss, "GpuIndexFlatIP")
    gpu_flat = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), emb_dim) if DEVICE == "cuda" and FAISS_GPU_AVAILABLE else None
            
    tops = [list() for _ in range(len(df))]
    with Timer("FAISS Search"):
        DB_BATCH_SIZE = 4096
        GPU_QUERY_BATCH_SIZE = 256
        CPU_QUERY_BATCH_SIZE = 60_000

        for db0 in range(0, len(embed), DB_BATCH_SIZE):
            db1 = min(db0 + DB_BATCH_SIZE, len(embed))
            use_gpu = DEVICE == "cuda" and (db1-db0) <= DB_BATCH_SIZE and gpu_flat is not None
            
            if use_gpu:
                faiss_index, max_q = gpu_flat, GPU_QUERY_BATCH_SIZE
                faiss_index.reset()
                faiss_index.add(embed[db0:db1])
            else:
                faiss_index, max_q = cpu_index, CPU_QUERY_BATCH_SIZE

            for off in tqdm(range(0, len(embed), max_q), leave=False, desc=f"search shard"):
                q_slice = slice(off, min(off + max_q, len(embed)))
                D, I = faiss_index.search(embed[q_slice], 64)
                for qi, src in enumerate(range(*q_slice.indices(len(embed)))):
                    for sim, dst in zip(D[qi], I[qi]):
                        if dst == -1 or sim < thr: continue
                        dst_global = (dst + db0) if use_gpu else dst
                        if dst_global >= len(df): continue
                        if dst_global == src: continue
                        if (tag == "within" and df.census.iloc[src] != df.census.iloc[dst_global]) or \
                           (tag == "across" and df.census.iloc[src] == df.census.iloc[dst_global]): continue
                        _merge_topk(tops, src, float(sim), dst_global, 64)
    
    top_path = OUTDIR / f"siamese_top_{tag}_shard_{shard_index}.csv"
    link_path = OUTDIR / f"siamese_links_{tag}_shard_{shard_index}.csv"
    top_path.write_text("id,top_matches\n")
    link_path.write_text("src_id,dst_id,sim\n")
    top_buf, link_buf = [], []

    def flush():
        nonlocal top_buf, link_buf
        if top_buf:
            with top_path.open("a") as f: f.write("\n".join(top_buf) + "\n")
            top_buf = []
        if link_buf:
            with link_path.open("a") as f: f.write("\n".join(link_buf) + "\n")
            link_buf = []

    with Timer(f"Write Siamese outputs for shard {shard_index}"):
        for i in range(start_idx, end_idx):
            src_id = i
            if not tops[src_id]: continue
            
            row_id = df.id.iloc[src_id]
            best = sorted(tops[src_id], reverse=True)[:20]
            match_list = json.dumps([[df.id.iloc[d], round(s, 4)] for s, d in best])
            top_buf.append(f"{row_id},{match_list}")
            
            for sim, dst in best:
                if src_id < dst:
                    dst_row_id = df.id.iloc[dst]
                    link_buf.append(f"{row_id},{dst_row_id},{sim:.6f}")
            
            if i > 0 and i % SAVE_INTERVAL == 0: flush()
    flush()
    logging.info(f"Shard {shard_index} finished Siamese stage.")

# ─── MERGE & EVALUATE ───────────────────────────────────────────────────────
def run_final_merge_and_eval(df, tag):
    with Timer(f"Final Merge & Eval for '{tag}'"):
        link_files = glob.glob(str(OUTDIR / f"ensemble_links_{tag}_shard_*.csv"))
        if not link_files:
            logging.warning(f"No link files found for tag '{tag}'. Skipping final steps.")
            return

        logging.info(f"Merging {len(link_files)} link files...")
        all_links = pd.concat([pd.read_csv(f) for f in link_files]).drop_duplicates()

        logging.info(f"Found {len(all_links)} unique links. Building final clusters...")
        parent = list(range(len(df)))
        rank = [0] * len(df)
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                if rank[ra] < rank[rb]: parent[ra] = rb
                elif rank[ra] > rank[rb]: parent[rb] = ra
                else: parent[rb] = ra; rank[ra] += 1
        
        for _, row in tqdm(all_links.iterrows(), total=len(all_links), desc="Clustering"):
            union(row.src_idx, row.dst_idx)

        final_clusters = np.array([find(i) for i in range(len(df))])
        
        out_df = pd.DataFrame({"id": df.id, "predicted_cluster": final_clusters})
        out_path = OUTDIR / f"ensemble_clusters_final_{tag}.csv"
        out_df.to_csv(out_path, index=False)
        logging.info(f"Saved final clusters to {out_path}")

        labeled_mask = df['label'] != -1
        if not labeled_mask.any():
            logging.warning("No ground truth labels found (all are -1). Skipping evaluation.")
            return

        true_labels = df.loc[labeled_mask, 'label'].values
        pred_labels = final_clusters[labeled_mask]
        
        report = {
            "tag": tag, "num_true_clusters": len(np.unique(true_labels)), "num_pred_clusters": len(np.unique(pred_labels)),
            "adjusted_rand_index": metrics.adjusted_rand_score(true_labels, pred_labels),
            "adjusted_mutual_info": metrics.adjusted_mutual_info_score(true_labels, pred_labels),
            "homogeneity": metrics.homogeneity_score(true_labels, pred_labels),
            "completeness": metrics.completeness_score(true_labels, pred_labels),
            "v_measure": metrics.v_measure_score(true_labels, pred_labels),
        }
        
        logging.info(f"--- Evaluation Report for '{tag}' ---")
        for key, val in report.items():
            logging.info(f"{key:<25}: {val:.4f}" if isinstance(val, float) else f"{key:<25}: {val}")
        
        report_path = OUTDIR / f"evaluation_report_{tag}.json"
        with open(report_path, 'w') as f: json.dump(report, f, indent=2)
        logging.info(f"Saved evaluation report to {report_path}")

# ─── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-index", type=int, default=1, help="1-based worker index")
    parser.add_argument("--num-shards",  type=int, default=1, help="Total number of shards")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--merge-only", action="store_true", help="Run only the final merge and evaluation step.")
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Run on a random percentage of the data (e.g., 10.0 for 10%%). Default: 100%%."
    )
    args = parser.parse_args()

    # Simplified data handling flow
    full_df = load_all_data()

    if args.merge_only:
        logging.info("--- Running in MERGE-ONLY mode ---")
        # To evaluate correctly on a sample, we must recreate the exact same sample
        df_for_merge = full_df
        if args.percentage < 100.0:
            logging.info(f"Evaluating on a {args.percentage}%% sample.")
            df_for_merge = full_df.sample(frac=args.percentage / 100.0, random_state=RNG).reset_index(drop=True)

        for tag in ("within", "across"):
            run_final_merge_and_eval(df_for_merge, tag)
        logging.info("🎉 All tasks complete.")

    else:
        # --- Run a processing shard ---
        processing_df = full_df
        if args.percentage < 100.0:
            logging.info(f"Running on a {args.percentage}%% random sample of the data.")
            processing_df = full_df.sample(frac=args.percentage / 100.0, random_state=RNG)
        
        # All functions will now operate on a dataframe with a simple 0..N-1 index
        processing_df = processing_df.reset_index(drop=True)

        if not (1 <= args.shard_index <= args.num_shards): raise ValueError("shard-index must be 1..num-shards")
        N  = len(processing_df)
        start = (args.shard_index-1)*N // args.num_shards
        end   = (args.shard_index  )*N // args.num_shards

        for tag in ("within","across"):
            logging.info(f"=== {tag.upper()} | Shard {args.shard_index}/{args.num_shards} | Processing indices {start}-{end} ===")
            # The full processing_df is passed, and the function internally handles the start:end slice
            run_ensemble(processing_df, tag, start, end, args.shard_index, resume=args.resume)
            if DEVICE == "cuda":
                run_siamese(processing_df, tag, start, end, args.shard_index)
        logging.info(f"Shard {args.shard_index} has completed its processing tasks.")