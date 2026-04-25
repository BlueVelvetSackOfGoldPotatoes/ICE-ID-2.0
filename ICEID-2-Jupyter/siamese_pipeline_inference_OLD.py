#!/usr/bin/env python3
"""
Fast inference-only script for ICE-ID, with automatic sharding:
  python siamese_pipeline_inference.py --shard-index 1 --num-shards 2
"""
import os
import gc
import itertools
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
import time

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

try:
    import cupy as cp
    CUPY_OK = True
except ModuleNotFoundError:
    cp = None
    CUPY_OK = False

import torch
from torch import nn
from rapidfuzz.distance import JaroWinkler
from xgboost import XGBClassifier
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
ART      = Path("artifacts")
MODELDIR = Path("models_ensemble_siamese")
OUTDIR   = Path("deploy_out"); OUTDIR.mkdir(exist_ok=True)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CPU_JOBS = max(1, (os.cpu_count() or 2) // 2)
BATCH, MICRO = 10_000, 2_000
SAMPLE_FRAC = 1
SAVE_INTERVAL = 1000          # flush outputs every N source rows
MAX_Q         = 60_000        # ≤ 65 535 queries per FAISS GPU call

# ─── small helpers ─────────────────────────────────────────────────────────
load_json = lambda p: json.load(open(p, "r"))

def _merge_topk(heap_list, qid, sim, dst, k):
    """
    Maintain a min‑heap of length ≤ k with (similarity, dst) for each query id.
    """
    h = heap_list[qid]
    if len(h) < k:
        heapq.heappush(h, (sim, dst))
    elif sim > h[0][0]:
        heapq.heapreplace(h, (sim, dst))

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

        # ── Pair-preserving subsample ────────────────────────────────────────
        if SAMPLE_FRAC < 1.0:
            rng = np.random.default_rng(RNG)

            labelled   = rows[rows.person != -1]
            unlabelled = rows[rows.person == -1]

            # persons that actually have ≥2 rows  → ensures positives exist
            dup_persons = (
                labelled.groupby("person").size()
                .loc[lambda s: s >= 2]
                .index.values
            )
            if len(dup_persons) == 0:
                raise RuntimeError(
                    "Dataset has no duplicate person-ids; cannot build positive pairs."
                )

            k_dup  = max(1, int(len(dup_persons) * SAMPLE_FRAC))
            keep_p = rng.choice(dup_persons, k_dup, replace=False)

            labelled_sample = labelled[labelled.person.isin(keep_p)]

            # some negatives: take a fraction of the unlabelled set (if it exists)
            if len(unlabelled):
                n_neg       = max(1, int(len(unlabelled) * SAMPLE_FRAC))
                unlab_sample = unlabelled.sample(n=n_neg, random_state=RNG)
                rows = pd.concat([labelled_sample, unlab_sample]).reset_index(drop=True)
            else:
                rows = labelled_sample.reset_index(drop=True)

        ids    = rows.row_id.astype(str).values
        labels = rows.person.fillna(-1).astype(int).values

        people = pd.read_csv(
            RAW / "people.csv",
            usecols=["id", "heimild", "birthyear", "sex",
                     "first_name", "patronym", "surname"],
            dtype=str, low_memory=False
        ).set_index("id").reindex(ids)
        people[["first_name", "patronym", "surname"]] = (
            people[["first_name", "patronym", "surname"]].fillna('')
        )

        mann = pd.read_csv(
            RAW / "manntol_einstaklingar_new.csv",
            usecols=["id", "bi_sokn", "bi_hreppur", "bi_sysla"],
            dtype=str, low_memory=False
        ).set_index("id").reindex(ids)

        # census = source census number  (we’ll use it for within/ across logic)
        census = people.heimild.fillna(-1).astype(int)

        df = pd.DataFrame({
            "id": ids,
            "label": labels,
            "census": census,                       # ← NEW column
            "heim":   census,                       # keep original name for legacy code
            "birth": pd.to_numeric(people.birthyear, errors='coerce').fillna(0).astype(int),
            "sex": people.sex.str.lower().str.strip()
                     .map({"karl": 1, "karl.": 1, "kona": 0, "kona.": 0})
                     .fillna(-1).astype(int),
            "parish_id":   mann.bi_sokn.fillna(''),
            "district_id": mann.bi_hreppur.fillna(''),
            "county_id":   mann.bi_sysla.fillna(''),
            "pn": people.patronym,
            "sn": people.surname,
            "gn": people.first_name
        })

        df["pn_sn"]          = df.pn.where(df.pn != '', df.sn)
        df["pn_sn_soundex"]  = df.pn_sn.apply(soundex)
        df["birth_decade"]   = (df.birth // 10) * 10
        df["cluster_gt"] = df.label.where(df.label != -1, df.id).astype(str)

        df.reset_index(drop=True, inplace=True)

    return df


def create_comparison_features(r1, r2):
    sim = JaroWinkler.normalized_similarity
    return np.array([
        sim(r1.gn,    r2.gn),
        sim(r1.pn_sn, r2.pn_sn),
        abs(r1.birth - r2.birth) if r1.birth and r2.birth else -1,
        abs(r1.heim  - r2.heim)  if r1.heim  and r2.heim  else -1,
        int(r1.sex != -1 and r1.sex == r2.sex),
        int(bool(r1.parish_id)   and r1.parish_id   == r2.parish_id),
        int(bool(r1.district_id) and r1.district_id == r2.district_id),
        int(bool(r1.county_id)   and r1.county_id   == r2.county_id),
    ], dtype=float)

def score_with_xgb(m, feats):
    if m.get_params().get("device","")=="cuda" and CUPY_OK and DEVICE=="cuda" and cp is not None:
        return m.predict_proba(cp.asarray(feats))[:,1]
    m.set_params(predictor="cpu_predictor", device=None)
    return m.predict_proba(feats)[:,1]

def score_pairs(models, feats):
    out=[]
    for m in models:
        if isinstance(m, XGBClassifier):
            out.append(score_with_xgb(m, feats))
        elif isinstance(m, list):
            out.append(np.mean([rf.predict_proba(feats)[:,1] for rf in m],0))
        else:
            out.append(m.predict_proba(feats)[:,1])
    return np.mean(out, axis=0)

def connected_components(n, links):
    parent=list(range(n))
    rank=[0]*n
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for a,b,_ in links:
        ra,rb=find(a),find(b)
        if ra==rb: continue
        if rank[ra]<rank[rb]:
            parent[ra]=rb
        elif rank[ra]>rank[rb]:
            parent[rb]=ra
        else:
            parent[rb]=ra; rank[ra]+=1
    root={}
    return [root.setdefault(find(i), len(root)) for i in range(n)]


# ─── ENSEMBLE (shardable) ──────────────────────────────────────────────────
def run_ensemble(df, tag, start_idx, end_idx, resume=False):
    """
    If resume=True and a checkpoint exists, pick up at last i+1.
    Writes a tiny checkpoint file after every record.
    """

    out_csv = OUTDIR / f"ensemble_clusters_{tag}.csv"
    if resume and out_csv.exists():
        logging.info(f"[Ensemble-{tag}] clusters CSV found – skipping.")
        return

    mdir    = MODELDIR / f"combined_{tag}"
    models  = [load(mdir/fn) for fn in
               ("rf.joblib","catboost.joblib","xgboost.joblib",
                "lightgbm.joblib","mlp.joblib","logistic.joblib")]
    thr     = load_json(mdir/"meta.json")["thr"]

    # build blocks in-place (unchanged)
    block_defs = {
      "soundex_decade_sex": ["pn_sn_soundex","birth_decade","sex"],
      "parish_decade_sex":  ["parish_id","birth_decade","sex"],
      "parish_year":        ["parish_id","birth"],
      "soundex_year":       ["pn_sn_soundex","birth"],
    }
    bidx = {}
    for cols in block_defs.values():
        mask = np.ones(len(df), bool)
        for c in cols:
            if df[c].dtype == object:
                mask &= df[c].ne('')
            else:
                mask &= df[c].gt(0)
        idx = defaultdict(list)
        for ii in df.index[mask]:
            key = tuple(df.at[ii, c] for c in cols)
            idx[key].append(ii)
        bidx[tuple(cols)] = idx

    # union-find state
    parent = list(range(len(df)))
    rank   = [0] * len(df)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # checkpoint file
    ckpt = OUTDIR/f"ensemble_{tag}.ckpt"
    if resume and not ckpt.exists():
        ckpt.write_text(str(start_idx-1))
    if resume and ckpt.exists():
        last = int(ckpt.read_text())
        start_idx = max(start_idx, last+1)

    mode = "a" if resume else "w"

    # open sample log
    sf = open(OUTDIR/f"samples_{tag}.txt", mode)
    positives = 0
    total = end_idx - start_idx
    pbar = tqdm(total=total, desc=f"{tag} recs", unit="rec")

    for i in range(start_idx, end_idx):
        cand = set()
        for cols, idx in bidx.items():
            key = tuple(df.at[i, c] for c in cols)
            cand.update(idx.get(key, []))
        cand.discard(i)
        if not cand:
            pbar.update()
            ckpt.write_text(str(i))
            continue

        feats = np.vstack([create_comparison_features(df.loc[i], df.loc[j])
                           for j in cand])
        probs = score_pairs(models, feats)

        for j, p in zip(cand, probs):
            same = (df.census[i] == df.census[j])
            if p >= thr and ((tag=="within") == same):
                a, b = (i, j) if i<j else (j, i)
                union(a, b)
                positives += 1
                if positives % 1000 == 0:
                    ra, rb = df.loc[a], df.loc[b]
                    entry = (
                      f"[SAMPLE #{positives}] p={p:.4f}\n\n"
                      f"**Q** idx={a},id={ra.id}\n"
                      f"{ra.to_frame().T.to_markdown(index=False)}\n\n"
                      f"**M** idx={b},id={rb.id}\n"
                      f"{rb.to_frame().T.to_markdown(index=False)}\n\n—\n"
                    )
                    tqdm.write(entry)
                    sf.write(entry + "\n"); sf.flush()

        del feats, probs
        gc.collect()
        pbar.update()
        ckpt.write_text(str(i))

    pbar.close()
    sf.close()

    # write final clusters
    clusters = [find(i) for i in range(len(df))]
    pd.DataFrame({"id": df.id, "cluster": clusters}) \
        .to_csv(OUTDIR / f"ensemble_clusters_{tag}.csv", index=False)

    # keep checkpoint so --resume knows we already finished
    del bidx, parent, rank
    gc.collect()


class Encoder(nn.Module):
    """
    *Exact* architecture that was trained (3 linear layers + BatchNorm + GELU).
    Keeping it global means we can load weights before run_siamese().
    """
    def __init__(self, d_in: int, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, d_out)
        )

    def forward(self, x):
        # return L2-normalised embeddings (cosine distance ≡ dot-product)
        return F.normalize(self.net(x), p=2, dim=1)

# ─── SIAMESE ───────────────────────────────────────────────────
SAVE_INTERVAL = 1000         # flush outputs every N source rows

def run_siamese(df: pd.DataFrame, tag: str) -> None:
    logging.info(f"[Siamese-{tag}] starting")

    # 1. metadata ---------------------------------------------------------
    meta         = load_json(MODELDIR / f"meta_{tag}.json")
    thr          = meta["thr"]
    emb_dim      = meta.get("emb_dim", 256)
    weights_file = MODELDIR / f"enc_{tag}.pt"

    # 2. data -------------------------------------------------------------
    X_sparse = sparse.load_npz(ART / "iceid_ml_ready.npz")[df.index]

    # 3. model ------------------------------------------------------------
    enc = Encoder(X_sparse.shape[1], emb_dim).to(DEVICE)
    enc.load_state_dict(torch.load(weights_file, map_location=DEVICE),
                        strict=False)
    enc.eval()

    # 4. embed ------------------------------------------------------------
    with Timer("Compute embeddings"):
        embed = np.zeros((len(df), emb_dim), dtype=np.float32)
        for i in tqdm(range(0, len(df), 1024), leave=False, desc="embed"):
            batch = torch.from_numpy(
                X_sparse[i:i+1024].toarray()).float().to(DEVICE)
            with torch.no_grad():
                embed[i:i+1024] = enc(batch).cpu().numpy()
    del X_sparse; gc.collect()

    # 5. FAISS ------------------------------------------------------------
    cpu_index = faiss.IndexFlatIP(emb_dim)   # master index on CPU (never crashes)
    cpu_index.add(embed)

    if DEVICE == "cuda":
        gpu_res  = faiss.StandardGpuResources()
        gpu_flat = faiss.GpuIndexFlatIP(gpu_res, emb_dim)
    else:
        gpu_flat = None                      # just a sentinel

    MAX_DB_GPU = 100_000    # < 131 072  → always under CuBLAS limit
    K_NEIGH    = 64
    MAX_Q_GPU  = 4_096      # send tiny query batches to GPU
        
    # 6. blocking (same as before) ---------------------------------------
    block_defs = {
        "soundex_decade_sex": ["pn_sn_soundex", "birth_decade", "sex"],
        "parish_decade_sex":  ["parish_id", "birth_decade", "sex"],
    }
    bidx = {}
    for cols in block_defs.values():
        idx = defaultdict(list)
        tmp = df.copy()
        for c in cols:
            tmp = tmp[tmp[c] != ''] if tmp[c].dtype == 'object' else tmp[tmp[c] > 0]
        for ii, row in tmp.iterrows():
            idx[tuple(row[c] for c in cols)].append(ii)
        bidx[tuple(cols)] = idx
    del tmp

    def make_block_index(vec_ids):
        """
        Build a *CPU‑only* FAISS index for this block.
        (The full‑dataset index can still live on GPU for ≤131 072 vecs,
        but per‑block searches stay on CPU so we never call the buggy
        GEMM path again.)
        """
        sub = faiss.IndexFlatIP(emb_dim)   # always CPU → no CuBLAS
        sub.add(embed[vec_ids])
        return sub

    # 7. outputs ----------------------------------------------------------
    top_path  = OUTDIR / f"siamese_top_{tag}_20.csv"
    link_path = OUTDIR / f"siamese_links_{tag}.csv"
    top_path.write_text("id,top_matches\n")
    link_path.write_text("src_id,dst_id,sim\n")

    total_processed = 0
    top_buf, link_buf = [], []

    def flush():
        if top_buf:
            with top_path.open("a") as f:
                for rid, js in top_buf:
                    f.write(f"{rid},{js}\n")
            top_buf.clear()
        if link_buf:
            with link_path.open("a") as f:
                for a, b, s in link_buf:
                    f.write(f"{df.id[a]},{df.id[b]},{s:.6f}\n")
            link_buf.clear()

    # 8. main loop --------------------------------------------------------
    K_GLOBAL = 512

    tops = [list() for _ in range(len(df))]

    for db0 in range(0, len(embed), MAX_DB_GPU):
        db1 = min(db0 + MAX_DB_GPU, len(embed))

        # ---- put *just this shard* on GPU if safe, else keep it on CPU ----
        if DEVICE == "cuda" and (db1 - db0) <= MAX_DB_GPU and gpu_flat is not None:
            gpu_flat.reset()
            gpu_flat.add(embed[db0:db1])      # ≤ 100 000 vectors → safe
            shard_index = gpu_flat
            this_max_q  = MAX_Q_GPU           # 4 096‑query micro‑batches
        else:
            shard_index = cpu_index           # CPU fallback
            this_max_q  = MAX_Q               # 60 000 as before

        # ---- search the whole query set against this shard -----------------
        for off in range(0, len(embed), this_max_q):
            q_slice = slice(off, min(off + this_max_q, len(embed)))
            D, I = shard_index.search(embed[q_slice], K_NEIGH)


            for qi, src in enumerate(range(*q_slice.indices(len(embed)))):
                for sim, dst in zip(D[qi], I[qi]):
                    if dst == -1 or sim < thr:                  # FAISS pad / low sim
                        continue
                    dst_global = dst + db0                     # re‑map to whole set
                    if dst_global == src:
                        continue
                    if tag == "within" and df.census[src] != df.census[dst_global]:
                        continue
                    if tag == "across" and df.census[src] == df.census[dst_global]:
                        continue
                    _merge_topk(tops, src, float(sim), dst_global, K_NEIGH)

    for src_id, heap in enumerate(tops):
        if not heap:
            continue
        best = sorted(heap, reverse=True)[:20]          # highest sim first
        top_buf.append((df.id[src_id],
                        json.dumps([[df.id[d], round(s, 4)] for s, d in best])))
        for sim, dst in best:
            if src_id < dst:
                link_buf.append((src_id, dst, sim))


    # 9. final flush ------------------------------------------------------
    flush()
    logging.info(f"[Siamese-{tag}] completed – all results written")


# ─── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=1,
                        help="1-based worker index")
    parser.add_argument("--num-shards",  type=int, default=1,
                        help="total number of shards")
    parser.add_argument("--resume", action="store_true",
                        help="resume ensemble from last checkpoint")
    args = parser.parse_args()

    df = load_all_data()
    N  = len(df)
    if not (1 <= args.shard_index <= args.num_shards):
        raise ValueError("shard-index must be 1..num-shards")

    start = (args.shard_index-1)*N // args.num_shards
    end   = (args.shard_index  )*N // args.num_shards

    for tag in ("within","across"):
        logging.info(f"=== Ensemble {tag} shard {args.shard_index}/{args.num_shards}"
                     f" idx[{start}:{end}) ===")
        run_ensemble(df, tag, start, end, resume=args.resume)

        if DEVICE=="cuda":
            logging.info(f"=== Siamese {tag} (full) ===")
            run_siamese(df, tag)
        else:
            logging.info("CUDA not available → skipping Siamese.")
