import os, gc, json, itertools, logging, random, time, math, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, log_evaluation
from catboost import CatBoostClassifier
from jellyfish import jaro_winkler_similarity

import argparse


# ─────────── CONFIG ──────────────────────────────────────────────────────────
PIPELINE_NAME = "Ensemble"
ART   = Path("artifacts")
RAW   = Path("raw_data")
MODELDIR_ENSEMBLE = Path("models_ensemble_siamese"); MODELDIR_ENSEMBLE.mkdir(exist_ok=True)
MODELDIR_SIAMESE  = Path("models_ensemble_siamese");  MODELDIR_SIAMESE.mkdir(exist_ok=True)
OUTDIR = Path("deploy_out"); OUTDIR.mkdir(exist_ok=True)

SAMPLE_FRAC = 1

RNG  = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Flags
RUN_CLASSICAL_ENSEMBLE = True
RUN_SIAMESE_PIPELINE   = True
SKIP_TRAINING          = False

# Shared hyper-params
# ── Things you’ll want to tune ────────────────────────────────────────────
#   ▸ NEG_PER            –  class-balance for training pairs
#   ▸ TREES              –  #estimators for tree models
#   ▸ EMBEDDING_DIM      –  Siamese latent size
#   ▸ SIAMESE_LR         –  Adam learning-rate
#   ▸ SIAMESE_BATCH      –  GPU RAM vs gradient quality
#   ▸ MARGIN             –  contrastive-loss margin

# Shared hyper-params
NEG_PER  = 4          # was 3
TREES    = 400        # was 250  (works better for <100 feature vectors)
TOP_K    = 20
INFERENCE_CHUNK_SIZE = 8_000     # reduces temp-RAM peak
PREVIEW_INTERVAL = 1000 # Print a preview every N records
# Ensemble-specific
RF_BATCH_SIZE = 100_000

# Siamese-specific
EMBEDDING_DIM   = 256     # gives the network more capacity
SIAMESE_LR      = 3e-5    # lower LR for smaller batch
SIAMESE_BATCH   = 64      # works if you have ≥12 GB VRAM; else keep 32
SIAMESE_EPOCHS  = 30
EARLY_STOP_PAT  = 4
MARGIN          = 0.5     #

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=SIAMESE_LR)
parser.add_argument("--dim", type=int,   default=EMBEDDING_DIM)
args = parser.parse_args()
SIAMESE_LR   = args.lr
EMBEDDING_DIM= args.dim

# ─────────── UTILITIES ───────────────────────────────────────────────────────
class Timer:
    def __init__(self, msg): self.msg, self.t0 = msg, time.perf_counter()
    def __enter__(self): logging.info(f"▶ {self.msg}"); return self
    def __exit__(self, *_): logging.info(f"⏱ {self.msg}: {time.perf_counter()-self.t0:.1f}s")

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

# ─────────── DATA LOADING ────────────────────────────────────────────────────
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

# ─────────── FEATURE ENGINEERING ────────────────────────────────────────────
def create_comparison_features(r1, r2):

    """Return a fixed-length numeric vector comparing two records."""
    return np.array([
        # string similarity
        jaro_winkler_similarity(r1.gn,    r2.gn),
        jaro_winkler_similarity(r1.pn_sn, r2.pn_sn),

        # numeric distance (-1 → missing)
        abs(r1.birth - r2.birth) if r1.birth and r2.birth else -1,
        abs(r1.heim  - r2.heim)  if r1.heim  and r2.heim  else -1,

        # simple binary tests
        int(r1.sex != -1 and r1.sex == r2.sex),
        int(r1.parish_id    != '' and r1.parish_id    == r2.parish_id),
        int(r1.district_id  != '' and r1.district_id  == r2.district_id),
        int(r1.county_id    != '' and r1.county_id    == r2.county_id),
    ], dtype=float)

def predict_ensemble_proba(models, X):
    """Return the mean positive-class probability over all sub-models."""
    probs = []
    for m in models:
        # --- XGBoost (CPU or GPU) -----------------------------------------
        if isinstance(m, XGBClassifier):
            # Give XGBoost ordinary NumPy.  It will return NumPy as well,
            # even when the model was trained with GPU support.
            probs.append(m.predict_proba(X)[:, 1])        # ← CHANGED
        # --- Bagged random-forest ('rf_list') ------------------------------
        elif isinstance(m, list):
            probs.append(np.mean([rf.predict_proba(X)[:, 1] for rf in m], 0))
        # --- Everything else (CatBoost, LightGBM, MLP, LR) -----------------
        else:
            probs.append(m.predict_proba(X)[:, 1])

    return np.mean(probs, axis=0)

# ─────────── CLASSICAL ENSEMBLE PIPELINE ────────────────────────────────────
def run_classical_pipeline(df, tag):
    logging.info("="*80 + "\n--- Classical Ensemble Pipeline ---\n" + "="*80)
    mdir = MODELDIR_ENSEMBLE / f"combined_{tag}"
    mdir.mkdir(exist_ok=True)
    model_files = ["rf","catboost","xgboost","lightgbm","mlp","logistic"]

    # ── Generate labeled training pairs ────────────────────────────────────
    cand_pairs = set()
    block_defs = {
        "Soundex_Initial_Decade_Sex": ["pn_sn_soundex","birth_decade","sex"],
        "Parish_Initial_Decade_Sex":  ["parish_id","birth_decade","sex"],
    }
    with Timer("Blocking for candidate pairs"):
        for pname, cols in block_defs.items():
            df_pass = df.copy()
            for c in cols:
                mask = (df_pass[c] != '') if df_pass[c].dtype == 'object' else (df_pass[c] > 0)
                df_pass = df_pass[mask]
            for _, grp in tqdm(df_pass.groupby(cols), desc=pname, leave=False):
                if len(grp) < 2: continue
                for a, b in itertools.combinations(grp.index, 2):
                    if tag == "within" and df.census[a] != df.census[b]:
                        continue
                    if tag == "across"  and df.census[a] == df.census[b]:
                        continue
                    cand_pairs.add(tuple(sorted((a, b))))
    logging.info(f"{len(cand_pairs):,} candidate pairs")

    pos, neg_all = [], []
    for a, b in cand_pairs:
        la, lb = df.label[a], df.label[b]
        if la != -1 and lb != -1:
            if la == lb:
                pos.append((a, b))
            else:
                neg_all.append((a, b))
    if not pos:
        return
    neg   = random.sample(neg_all, min(len(neg_all), len(pos) * NEG_PER))
    pairs = pos + neg
    y     = np.array([1]*len(pos) + [0]*len(neg))

    X = np.vstack([create_comparison_features(df.loc[a], df.loc[b]) for a, b in pairs])

    # -- train/validation split ---------------------------------------------
    cls_cnt = np.array(np.bincount(y, minlength=2), dtype=int)
    strat   = y if not np.any(cls_cnt < 2) else None
    if strat is None:
        logging.warning(f"[{tag}] too few labeled examples, skipping stratification")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.20, stratify=strat, random_state=RNG
    )

    # ── Train sub-models ───────────────────────────────────────────────────
    with Timer("Training ensemble"):
        rf_list = []
        for i in range(0, len(X_tr), RF_BATCH_SIZE):
            rf = RandomForestClassifier(
                n_estimators=TREES,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RNG + i
            ).fit(X_tr[i:i+RF_BATCH_SIZE], y_tr[i:i+RF_BATCH_SIZE])
            rf_list.append(rf)
        dump(rf_list, mdir/"rf.joblib")

        cat = CatBoostClassifier(
            iterations=TREES, task_type="GPU", devices="0",
            verbose=False, random_state=RNG
        ).fit(X_tr, y_tr)
        dump(cat, mdir/"catboost.joblib")

        xgb = XGBClassifier(
            device="cuda", tree_method="hist",
            n_estimators=1200, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bynode=0.8,
            reg_lambda=1, random_state=RNG
        )
        xgb.fit(X_tr, y_tr)
        dump(xgb, mdir/"xgboost.joblib")

        lgbm = LGBMClassifier(
            n_estimators=1600, learning_rate=0.04,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            device_type="gpu", random_state=RNG, verbosity=-1
        )
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[log_evaluation(50)])
        dump(lgbm, mdir/"lightgbm.joblib")

        mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=RNG)
        mlp.fit(X_tr, y_tr)
        dump(mlp, mdir/"mlp.joblib")

        lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RNG)
        lr.fit(X_tr, y_tr)
        dump(lr, mdir/"logistic.joblib")

    # ── Validate & choose threshold ─────────────────────────────────────────
    ensemble = [load(mdir/f"{n}.joblib") for n in model_files]
    probs    = predict_ensemble_proba(ensemble, X_va)
    if len(np.unique(y_va)) < 2:
        thr, f1_val, auc = 0.5, 0.0, float("nan")
        logging.warning(f"[{tag}] only one class in validation → using thr=0.5")
    else:
        thr    = max(np.linspace(0.05,0.95,19), key=lambda t: float(f1_score(y_va, probs >= t)))
        f1_val = f1_score(y_va, probs >= thr)
        auc    = roc_auc_score(y_va, probs)
    json.dump({"thr":float(thr), "f1":float(f1_val), "auc":float(auc)},
              open(mdir/"meta.json","w"), indent=2)
    logging.info(f"Validation F1={f1_val:.4f} AUROC={auc:.4f} @thr={thr:.2f}")

    # ── Scalable inference + clustering ─────────────────────────────────────
    with Timer("Ensemble inference / clustering"):
        # build blocking indices
        blocks, bidx = {
            "Soundex_Initial_Decade_Sex": ["pn_sn_soundex","birth_decade","sex"],
            "Parish_Initial_Decade_Sex":  ["parish_id","birth_decade","sex"],
            "Parish_BirthYear":           ["parish_id","birth"],
            "Soundex_BirthYear":          ["pn_sn_soundex","birth"],
        }, {}
        for name, cols in blocks.items():
            idx = defaultdict(list)
            df_pass = df.copy()
            for c in cols:
                mask = (df_pass[c] != '') if df_pass[c].dtype == 'object' else (df_pass[c] > 0)
                df_pass = df_pass[mask]
            for i, row in tqdm(df_pass.iterrows(), total=len(df_pass),
                              desc=f"build {name}", leave=False):
                idx[tuple(row[cols])].append(i)
            bidx[name] = idx

        thr      = json.load(open(mdir/"meta.json"))["thr"]
        ensemble = [load(mdir/f"{n}.joblib") for n in model_files]

        out_top = OUTDIR / f"ensemble_top_{tag}_{TOP_K}.csv"
        if out_top.exists(): out_top.unlink()
        pd.DataFrame(columns=["id", f"top_{TOP_K}_matches"]).to_csv(out_top, index=False)

        links = []
        N     = len(df)

        # 1️⃣ initialize processed_persons
        processed_persons: set[int] = set()

        for start in tqdm(range(0, N, INFERENCE_CHUNK_SIZE),
                          desc="Inference", leave=False):
            chunk_top = {}
            for i in range(start, min(start + INFERENCE_CHUNK_SIZE, N)):
                # 2️⃣ short-circuit rows whose label we've already used
                pers_i = df.label[i]
                if pers_i != -1 and pers_i in processed_persons:
                    continue

                # generate candidates via blocks
                cand = set()
                for name, cols in blocks.items():
                    key = tuple(df.loc[i, cols])
                    cand.update(bidx[name].get(key, []))
                cand.discard(i)
                if not cand:
                    continue
                cand = list(cand)

                # build features, score
                X     = np.vstack([create_comparison_features(df.loc[i], df.loc[j]) for j in cand])
                p     = predict_ensemble_proba(ensemble, X)
                preds = sorted(zip(cand, p), key=lambda x: x[1], reverse=True)

                # link & collect new_links
                new_links: list[int] = []
                for j, prob in preds:
                    if tag == "within" and df.census[i] != df.census[j]:
                        continue
                    if tag == "across"  and df.census[i] == df.census[j]:
                        continue
                    if prob >= thr and i < j:
                        links.append((i, j, prob))
                        new_links.append(j)

                # 3️⃣ mark any hand-labeled persons we just touched
                if pers_i != -1:
                    processed_persons.add(pers_i)
                for j in new_links:
                    pers_j = df.label[j]
                    if pers_j != -1:
                        processed_persons.add(pers_j)

                # preview + write out top-K
                preview_top(
                    df, df.id[i],
                    [df.id[j] for j, _ in preds],
                    [sc for _, sc in preds],
                    f"{PIPELINE_NAME}-{tag}", i
                )
                chunk_top[df.id[i]] = json.dumps(
                    [[df.id[j], round(float(prob),4)] for j, prob in preds[:TOP_K]]
                )

            if chunk_top:
                pd.Series(chunk_top).to_csv(out_top, mode='a', header=False)
            gc.collect()

        logging.info(f"{len(links):,} links ≥ thr")
        clusters = connected_components(N, links)
        pd.DataFrame({"id":df.id, "cluster":clusters})\
          .to_csv(OUTDIR / f"ensemble_clusters_{tag}.csv", index=False)
        evaluate_clusters(df.label.values, clusters, df.id.values, "Ensemble")

    show_sanity_sample(
        df,
        OUTDIR / f"ensemble_top_{tag}_{TOP_K}.csv",
        f"Ensemble-{tag}"
    )

# ── live preview for every processed query row ────────────────────────────
def preview_top(df, query_id, cands, scores, label, q_no):
    """
    Print a preview for a query row + its top 5 suggestions.
    This is throttled to only print periodically to avoid spamming the console.
    """
    # Only print every Nth item (and the very first one)
    if q_no > 0 and q_no % PREVIEW_INTERVAL != 0:
        return

    # Use tqdm.write to avoid breaking the progress bar
    tqdm.write(f"\n {label} | row #{q_no}")
    tqdm.write(df.loc[df.id == query_id]
               .drop(columns=["id"])
               .to_markdown(index=False))

    for rank, (cid, sc) in enumerate(zip(cands[:5], scores[:5]), 1):
        tqdm.write(f"\n #{rank:02d} (p̂={sc:.4f})")
        tqdm.write(df.loc[df.id == cid]
                   .drop(columns=["id"])
                   .to_markdown(index=False))


# ─────────── SIAMESE NETWORK PIPELINE ───────────────────────────────────────
class RecordPairDataset(Dataset):
    def __init__(self, X, pairs, y):
        self.X = X.tocsr(); self.pairs=pairs; self.y=y
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        a,b = self.pairs[i]
        return (torch.from_numpy(self.X[a].toarray()[0]).float(),
                torch.from_numpy(self.X[b].toarray()[0]).float(),
                torch.tensor(self.y[i], dtype=torch.float32))

class EmbeddingNet(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_in,1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(1024,512),  nn.BatchNorm1d(512),  nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512,d_out))
    def forward(self,x): return F.normalize(self.net(x),p=2,dim=1)

class SiameseNet(nn.Module):
    def __init__(self, enc): super().__init__(); self.enc=enc
    def forward(self,x1,x2): return self.enc(x1), self.enc(x2)

class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__(); self.m=margin
    def forward(self,z1,z2,y):
        sim=F.cosine_similarity(z1,z2)
        return torch.mean((1-y)*sim**2 + y*torch.clamp(self.m-sim,min=0)**2)

def gen_hard_pairs(df, idxs, mult):
    pos,neg=[],[]
    block={"Soundex_Initial_Decade_Sex":
              ["pn_sn_soundex","birth_decade","sex"],
           "Parish_Initial_Decade_Sex":
              ["parish_id","birth_decade","sex"]}
    sub=df.loc[idxs]
    for cols in block.values():
        tmp=sub.copy()
        for c in cols:
            tmp=tmp[tmp[c]!=''] if tmp[c].dtype=='object' else tmp[tmp[c]>0]
        for _,grp in tmp.groupby(cols):
            if len(grp)<2: continue
            for a,b in itertools.combinations(grp.index,2):
                la,lb=df.label[a],df.label[b]
                if la!=-1 and lb!=-1:
                    if la==lb: pos.append((a,b))
                    else: neg.append((a,b))
    neg=random.sample(neg, min(len(neg), len(pos)*mult))
    y=[1]*len(pos)+[0]*len(neg)
    return pos+neg, y

def find_threshold(model, dl):
    sims,lab=[],[]
    with torch.no_grad():
        for v1,v2,l in dl:
            z1,z2=model(v1.to(DEVICE),v2.to(DEVICE))
            sims.extend(F.cosine_similarity(z1,z2).cpu().numpy())
            lab.extend(l.numpy())
    sims=np.array(sims); lab=np.array(lab)
    thr=max(np.linspace(0,1,401), key=lambda t: float(f1_score(lab, sims>=t)))
    return float(thr)


def evaluate_clusters(labels, clusters, ids, model_name):
    df = pd.DataFrame({'label': labels, 'cluster': clusters, 'id': ids})
    df = df[df.label != -1].copy()
    if df.empty: logging.warning(f"[{model_name}] No labeled data to evaluate clusters against."); return
    
    with Timer(f"[{model_name}] Calculating pairwise evaluation metrics"):
        predicted_pairs = set(itertools.chain.from_iterable(itertools.combinations(sorted(g['id'].tolist()), 2) for _, g in tqdm(df.groupby('cluster'), desc="Generating predicted pairs", leave=False) if len(g) > 1))
        actual_pairs = set(itertools.chain.from_iterable(itertools.combinations(sorted(g['id'].tolist()), 2) for _, g in tqdm(df.groupby('label'), desc="Generating actual pairs", leave=False) if len(g) > 1))
        
        tp = len(predicted_pairs.intersection(actual_pairs)); fp = len(predicted_pairs.difference(actual_pairs)); fn = len(actual_pairs.difference(predicted_pairs))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0; recall = tp / (tp + fn) if (tp + fn) > 0 else 0; f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    logging.info(f"--- [{model_name}] Cluster Evaluation ---\nPairwise Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\nTP: {tp:,}, FP: {fp:,}, FN: {fn:,}")


def connected_components(n, links):
    parent = list(range(n))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i]); return parent[i]
    for a,b,_ in tqdm(links, desc="Building clusters", leave=False):
        ra, rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra
    root2cid = {r: i for i, r in enumerate(pd.Series(parent).unique())}
    return [root2cid.get(find(i), -1) for i in range(n)]


def run_siamese_pipeline(df, tag):
    logging.info("="*80 + "\n--- Siamese Pipeline ---\n" + "="*80)

    X_full   = sparse.load_npz(ART / "iceid_ml_ready.npz")
    row_idx  = df.index.to_numpy()
    X_sparse = X_full[row_idx]
    N        = X_sparse.shape[0]

    # ── Training (skip if SKIP_TRAINING) ──────────────────────────────────
    if not SKIP_TRAINING:
        lbl      = df[df.label != -1].index
        tr, va   = train_test_split(lbl, test_size=0.2, random_state=RNG)
        tr_pairs, y_tr = gen_hard_pairs(df, tr, NEG_PER)
        va_pairs, y_va = gen_hard_pairs(df, va, NEG_PER)

        dl_tr = DataLoader(RecordPairDataset(X_sparse, tr_pairs, y_tr),
                           batch_size=SIAMESE_BATCH, shuffle=True, num_workers=0)
        dl_va = DataLoader(RecordPairDataset(X_sparse, va_pairs, y_va),
                           batch_size=SIAMESE_BATCH, shuffle=False, num_workers=0)

        enc = EmbeddingNet(X_sparse.shape[1], EMBEDDING_DIM).to(DEVICE)
        net = SiameseNet(enc).to(DEVICE)
        opt = optim.Adam(net.parameters(), lr=SIAMESE_LR)
        crit = CosineContrastiveLoss().to(DEVICE)

        best, patience = float('inf'), 0
        with Timer("Siamese training"):
            for e in range(1, SIAMESE_EPOCHS+1):
                net.train(); tl = 0
                for v1, v2, l in tqdm(dl_tr, desc=f"Epoch{e}[TR]", leave=False):
                    v1, v2, l = v1.to(DEVICE), v2.to(DEVICE), l.to(DEVICE)
                    opt.zero_grad()
                    z1, z2 = net(v1, v2)
                    loss = crit(z1, z2, l)
                    loss.backward()
                    opt.step()
                    tl += loss.item()
                net.eval(); vl = 0
                with torch.no_grad():
                    for v1, v2, l in dl_va:
                        v1, v2, l = v1.to(DEVICE), v2.to(DEVICE), l.to(DEVICE)
                        vl += crit(*net(v1, v2), l).item()
                logging.info(f"Epoch {e}: TL={tl/len(dl_tr):.4f} VL={vl/len(dl_va):.4f}")
                if vl < best:
                    best, patience = vl, 0
                    torch.save(enc.state_dict(), MODELDIR_SIAMESE / f"enc_{tag}.pt")
                else:
                    patience += 1
                if patience >= EARLY_STOP_PAT:
                    break

        thr = find_threshold(net, dl_va)
        json.dump({"thr":thr},
                  open(MODELDIR_SIAMESE / f"meta_{tag}.json", "w"),
                  indent=2)

    # ── Embed all records ────────────────────────────────────────────────
    enc = EmbeddingNet(X_sparse.shape[1], EMBEDDING_DIM)
    enc.load_state_dict(torch.load(MODELDIR_SIAMESE / f"enc_{tag}.pt", weights_only=True))
    enc.to(DEVICE).eval()
    thr = json.load(open(MODELDIR_SIAMESE / f"meta_{tag}.json"))["thr"]

    with Timer("Embedding all records"):
        emb = np.zeros((N, EMBEDDING_DIM), dtype=np.float32)
        for i in tqdm(range(0, N, SIAMESE_BATCH), leave=False):
            idx   = range(i, min(i+SIAMESE_BATCH, N))
            dense = torch.from_numpy(X_sparse[idx].toarray()).float().to(DEVICE)
            with torch.no_grad():
                emb[idx] = enc(dense).cpu().numpy()
    del X_sparse; gc.collect()

    # ── Build block indices ────────────────────────────────────────────
    blocks = {
        "Soundex_Initial_Decade_Sex": ["pn_sn_soundex","birth_decade","sex"],
        "Parish_Initial_Decade_Sex":  ["parish_id","birth_decade","sex"],
    }
    bidx = {}
    for cols in blocks.values():
        idx = defaultdict(list)
        tmp = df.copy()
        for c in cols:
            tmp = tmp[tmp[c] != ''] if tmp[c].dtype == 'object' else tmp[tmp[c] > 0]
        for i, row in tqdm(tmp.iterrows(), total=len(tmp),
                          desc=f"build {cols[0]}", leave=False):
            idx[tuple(row[cols])].append(i)
        bidx[tuple(cols)] = idx

    out_top = OUTDIR / f"siamese_top_{tag}_{TOP_K}.csv"
    pd.DataFrame(columns=["id", f"top_{TOP_K}_matches"]).to_csv(out_top, index=False)
    links = []
    PIPELINE_NAME = "Siamese"

    # 1️⃣ initialize processed_persons
    processed_persons: set[int] = set()

    for s in tqdm(range(0, N, INFERENCE_CHUNK_SIZE),
                  desc="Siamese inference", leave=False):
        chunk_top = {}
        for i in range(s, min(s + INFERENCE_CHUNK_SIZE, N)):
            # 2️⃣ short-circuit rows whose label we've already used
            pers_i = df.label[i]
            if pers_i != -1 and pers_i in processed_persons:
                continue

            # generate candidates via blocks
            cand = set()
            for cols, idx in bidx.items():
                row_key = tuple(df.loc[i, list(cols)])
                cand.update(idx.get(row_key, []))
            cand.discard(i)
            if not cand:
                continue
            cand = list(cand)

            # score
            sim   = np.sum(emb[i] * emb[cand], axis=1)
            preds = sorted(zip(cand, sim), key=lambda x: x[1], reverse=True)

            # link & collect new_links
            new_links: list[int] = []
            for j, score in preds:
                if tag == "within" and df.census[i] != df.census[j]:
                    continue
                if tag == "across"  and df.census[i] == df.census[j]:
                    continue
                if score >= thr and i < j:
                    links.append((i, j, score))
                    new_links.append(j)

            # 3️⃣ mark any hand-labeled persons we just touched
            if pers_i != -1:
                processed_persons.add(pers_i)
            for j in new_links:
                pers_j = df.label[j]
                if pers_j != -1:
                    processed_persons.add(pers_j)

            # preview + top-K
            preview_top(
                df, df.id[i],
                [df.id[j] for j, _ in preds],
                [sc for _, sc in preds],
                f"{PIPELINE_NAME}-{tag}", i
            )
            chunk_top[df.id[i]] = json.dumps(
                [[df.id[j], round(float(score),4)] for j, score in preds[:TOP_K]]
            )

        if chunk_top:
            pd.Series(chunk_top).to_csv(out_top, mode='a', header=False)
        gc.collect()

    clusters = connected_components(N, links)
    pd.DataFrame({"id":df.id, "cluster":clusters})\
      .to_csv(OUTDIR / f"siamese_clusters_{tag}.csv", index=False)
    evaluate_clusters(df.label.values, clusters, df.id.values, "Siamese")

    show_sanity_sample(
        df,
        OUTDIR / f"siamese_top_{tag}_{TOP_K}.csv",
        f"Siamese-{tag}"
    )
# ════════════════════════════════════════════════════════════════════════════
# Sanity-check utility  – print 10 random rows + their top-20 candidates
# ════════════════════════════════════════════════════════════════════════════
def show_sanity_sample(df: pd.DataFrame, top_csv: Path, label: str,
                       k: int = TOP_K, n: int = 10, rnd: int = RNG) -> None:
    """
    Pretty-print `n` random entries from the prediction top-k file together
    with the *entire* dataframe rows of each of their k recommendations.

    Parameters
    ----------
    df       : full person dataframe (after load_all_data)
    top_csv  : path of the *_top_*.csv produced by the pipeline
    label    : string shown in the header    (e.g. 'Ensemble-within')
    k, n, rnd: TOP-K to display, #queries to sample, and RNG seed
    """
    if not top_csv.exists() or top_csv.stat().st_size == 0:
        logging.warning(f"[sanity] {label}: '{top_csv.name}' is empty – skip"); return

    tops = pd.read_csv(top_csv)
    if tops.empty:
        logging.warning(f"[sanity] {label}: no rows in csv"); return

    samp = tops.sample(n=min(n, len(tops)), random_state=rnd)

    wide_line = "═" * 120
    for query_num, (idx, row) in enumerate(samp.iterrows(), 1):
        q_id = row["id"]
        try:
            cand = json.loads(row[f"top_{k}_matches"])
        except json.JSONDecodeError:
            logging.error(f"[sanity] bad JSON for id={q_id} – skip"); continue

        print(f"\n{wide_line}\n🌟  {label}   |   QUERY #{query_num}")
        print(df.loc[df.id == q_id]
              .drop(columns=["id"])                # hide raw id if desired
              .to_markdown(index=False))

        print(f"\nTop {k} suggestions:")
        for rank, (cid, score) in enumerate(cand, 1):
            print(f"\n#{rank:02d} (p̂ = {score:.4f})")
            print(df.loc[df.id == cid]
                  .drop(columns=["id"])
                  .to_markdown(index=False))



# ─────────── ENTRY POINT ────────────────────────────────────────────────────
def main():
    logging.info("═"*80+"\nUnified Record-Linkage Pipeline\n"+"═"*80)
    torch.manual_seed(RNG); np.random.seed(RNG); random.seed(RNG)
    df=load_all_data()

    for tag in ("within", "across"):
        if RUN_CLASSICAL_ENSEMBLE:
            run_classical_pipeline(df, tag)
        if RUN_SIAMESE_PIPELINE and DEVICE=="cuda":
            run_siamese_pipeline(df, tag)

if __name__=="__main__":
    main()