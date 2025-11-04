from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .er_bench import (
    prepare_pair_splits as _prepare_pair_splits,
    _record_text,
)
from .edits import rebuild_clusters_from_edges, canonicalize_edges


def _load_text_maps(dataset_dir: str) -> Tuple[Dict, Dict]:
    """Loads left and right text mappings from a dataset directory.

    Reads left.csv and right.csv files and creates dictionaries mapping record
    IDs to their text representations.

    Args:
        dataset_dir (str): Path to the directory containing left.csv and right.csv.

    Returns:
        Tuple[Dict, Dict]: A tuple containing (left_map, right_map) dictionaries.
    """
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    return left.set_index("id")["text"].to_dict(), right.set_index("id")["text"].to_dict()


def _load_pairs_tfidf(dataset_dir: str, lmap: Dict, rmap: Dict, split: str) -> Tuple[pd.Series, pd.Series]:
    """Loads and concatenates text pairs for TF-IDF-based models.

    Creates concatenated text features from left and right record texts using
    [SEP] as a separator, along with their labels.

    Args:
        dataset_dir (str): Path to the dataset directory containing split CSVs.
        lmap (Dict): Dictionary mapping left table IDs to text.
        rmap (Dict): Dictionary mapping right table IDs to text.
        split (str): Name of the split to load (e.g., 'train', 'validation', 'test').

    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing (X, y) where X is concatenated texts and y is labels.
    """
    p = pd.read_csv(os.path.join(dataset_dir, f"{split}.csv"))
    X = (p["ltable_id"].map(lmap).fillna("") + " [SEP] " + p["rtable_id"].map(rmap).fillna(""))
    y = p["label"].astype(int)
    return X, y


def _prepare_texts_list(df: pd.DataFrame, lmap: Dict, rmap: Dict) -> List[str]:
    """Prepares a list of concatenated text pairs from a dataframe.

    Creates text features suitable for TF-IDF vectorization by concatenating
    left and right record texts with [SEP] separator.

    Args:
        df (pd.DataFrame): DataFrame containing ltable_id and rtable_id columns.
        lmap (Dict): Dictionary mapping left table IDs to text.
        rmap (Dict): Dictionary mapping right table IDs to text.

    Returns:
        List[str]: A list of concatenated text strings for each pair.
    """
    return [str(lmap.get(r["ltable_id"], "") + " [SEP] " + rmap.get(r["rtable_id"], "")) for _, r in df.iterrows()]


@dataclass
class ExternalTrainConfig:
    """Configuration for training external pair classifiers on ICE-ID data.

    Args:
        people_csv (str): Path to people.csv.
        labels_csv (str): Path to labels csv.
        out_root (str): Output directory root where artifacts and model outputs are stored.
        sample_frac (float): Fraction of people rows to use for training data generation.
        seed (int): Random seed.

    Returns:
        ExternalTrainConfig: Immutable training configuration.
    """
    people_csv: str
    labels_csv: str
    out_root: str
    sample_frac: float = 1.0
    seed: int = 42


def prepare_training_data(cfg: ExternalTrainConfig) -> str:
    """Generates pair classification splits for external models.

    Args:
        cfg (ExternalTrainConfig): Training configuration.

    Returns:
        str: Path to the dataset directory containing left.csv, right.csv and split CSVs.
    """
    dm_data = os.path.join(cfg.out_root, "pair_splits")
    _prepare_pair_splits(
        people_csv=cfg.people_csv,
        labels_csv=cfg.labels_csv,
        out_dir=dm_data,
        max_pos=1_000_000 if cfg.sample_frac >= 1.0 else 1_000,
        neg_ratio=1.0,
        seed=cfg.seed,
        sample_frac=cfg.sample_frac,
        sample_seed=cfg.seed,
    )
    return dm_data


def _build_blocks(people: pd.DataFrame) -> Dict[Tuple[str, int], List[int]]:
    """Builds lightweight blocks using 2-char name prefix and 5-year birth buckets.

    Args:
        people (pd.DataFrame): Canonical people dataframe.

    Returns:
        Dict[Tuple[str,int], List[int]]: Mapping of (prefix,bucket) to record ids.
    """
    def prefix(row) -> str:
        for c in ("nafn_norm", "first_name", "surname"):
            v = str(row.get(c, "") or "").strip().lower()
            if v:
                return v[:2]
        return ""
    def by_bucket(by: Optional[float]) -> int:
        try:
            i = int(by)
        except Exception:
            return -1
        return i // 5
    blocks: Dict[Tuple[str, int], List[int]] = {}
    for rid, row in people.iterrows():
        pfx = prefix(row)
        byb = by_bucket(row.get("birthyear"))
        key = (pfx, byb)
        blocks.setdefault(key, []).append(int(row["id"]))
    return blocks


def _candidate_pairs(people: pd.DataFrame, max_pairs_per_node: int = 50) -> List[Tuple[int, int]]:
    """Generates candidate pairs within simple blocks with a per-node cap.

    Args:
        people (pd.DataFrame): Canonical people dataframe.
        max_pairs_per_node (int): Maximum candidate partners per record.

    Returns:
        List[Tuple[int,int]]: Candidate pairs.
    """
    blocks = _build_blocks(people)
    partners: Dict[int, int] = {}
    pairs: List[Tuple[int, int]] = []
    for key, ids in blocks.items():
        ids = list(sorted(set(int(i) for i in ids)))
        for i, a in enumerate(ids):
            used = 0
            for b in ids[i + 1:]:
                pa, pb = partners.get(a, 0), partners.get(b, 0)
                if pa >= max_pairs_per_node or pb >= max_pairs_per_node:
                    continue
                pairs.append((a, b))
                partners[a] = pa + 1
                partners[b] = pb + 1
    return pairs


def predict_edges_with_hf(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50, device: Optional[str] = None) -> Tuple[str, str]:
    """Predicts edges using a HuggingFace classifier trained on the pair splits.

    Args:
        dataset_dir (str): Directory with left/right and split CSVs produced by prepare_training_data.
        people_csv (str): Path to people.csv for candidate generation.
        out_dir (str): Destination directory for edges.csv and clusters.csv.
        threshold (float): Decision threshold on the positive probability.
        max_pairs_per_node (int): Cap candidate pairs per node within blocks.
        device (Optional[str]): 'cpu' or 'cuda' to force device. If None, auto.

    Returns:
        Tuple[str,str]: Paths to saved edges.csv and clusters.csv.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    os.makedirs(out_dir, exist_ok=True)
    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)
    for c in ["birthyear"]:
        people[c] = pd.to_numeric(people.get(c, pd.Series([], dtype=str)), errors="coerce").astype("Int64")
    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    model_dir = os.path.join(out_dir, "model")
    if os.path.isdir(model_dir):
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    dev = torch.device("cuda" if (device == "cuda" or (device is None and torch.cuda.is_available())) else "cpu")
    model.to(dev)
    model.eval()

    batch_size = 256
    scores: List[Tuple[int, int, float]] = []
    for i in range(0, len(cands), batch_size):
        chunk = cands[i:i + batch_size]
        texts = [(lmap.get(a, ""), rmap.get(b, "")) for a, b in chunk]
        enc = tok([f"{x} [SEP] {y}" for x, y in texts], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        for (a, b), p in zip(chunk, probs):
            scores.append((a, b, float(p)))

    pred = [(a, b) for (a, b, p) in scores if p >= threshold]
    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)
    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)

    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def predict_edges_with_sbert(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.7, max_pairs_per_node: int = 50, device: Optional[str] = None) -> Tuple[str, str]:
    """Predicts edges using SBERT cosine similarity within light blocks.

    Args:
        dataset_dir (str): Directory with left/right for text mapping.
        people_csv (str): Path to people.csv.
        out_dir (str): Destination for edges/clusters.
        threshold (float): Cosine similarity threshold.
        max_pairs_per_node (int): Cap on candidate pairs per node.
        device (Optional[str]): 'cpu' or 'cuda'.

    Returns:
        Tuple[str,str]: Paths to edges.csv and clusters.csv.
    """
    from sentence_transformers import SentenceTransformer, util
    import torch

    os.makedirs(out_dir, exist_ok=True)
    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)
    for c in ["birthyear"]:
        people[c] = pd.to_numeric(people.get(c, pd.Series([], dtype=str)), errors="coerce").astype("Int64")
    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    if device is None:
        device = "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    a_texts = [lmap.get(a, "") for a, _ in cands]
    b_texts = [rmap.get(b, "") for _, b in cands]
    a_emb = model.encode(a_texts, convert_to_tensor=True, show_progress_bar=False)
    b_emb = model.encode(b_texts, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(a_emb, b_emb).diagonal().cpu().numpy()
    pred = [pair for pair, s in zip(cands, sims) if s >= threshold]
    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)
    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_hf_classifier(dataset_dir: str, out_dir: str, epochs: int = 1, device: Optional[str] = None) -> str:
    """Fine-tunes DistilBERT on the prepared pair splits and saves the model.

    Args:
        dataset_dir (str): Directory with left/right and train/validation/test CSVs.
        out_dir (str): Output directory; model saved under out_dir/model.
        epochs (int): Number of training epochs.
        device (Optional[str]): 'cpu' or 'cuda'. If None, auto.

    Returns:
        str: Path to the saved model directory.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import f1_score
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "model")
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    def to_pairs(df: pd.DataFrame):
        texts = (df["ltable_id"].map(lmap).fillna("") + " [SEP] " + df["rtable_id"].map(rmap).fillna(""))
        labels = df["label"].astype(int).tolist()
        return texts.tolist(), labels
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(dataset_dir, "validation.csv"))
    
    if len(train) == 0:
        raise ValueError("Training set is empty. Increase sample_frac.")
    
    tr_texts, tr_labels = to_pairs(train)
    va_texts, va_labels = to_pairs(valid)
    
    if len(tr_texts) == 0 or not any(tr_labels):
        raise ValueError("Training data has no positive examples. Increase sample_frac.")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize(texts):
        return tok(texts, padding=True, truncation=True, max_length=128)
    tr_enc = tokenize(tr_texts)
    va_enc = tokenize(va_texts)
    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=max(1, int(epochs)),
        learning_rate=3e-5,
        logging_steps=50,
        lr_scheduler_type="linear",
        report_to=[],
        save_strategy="no",
        eval_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=SimpleDS(tr_enc, tr_labels))
    trainer.train()
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    return model_dir


def train_tfidf_classifier(dataset_dir: str, out_dir: str, max_features: int = 200000, lr_C: float = 1.0) -> str:
    """Trains a TF-IDF + LogisticRegression classifier and saves it.

    Args:
        dataset_dir (str): Directory with pair splits.
        out_dir (str): Output directory; model saved under out_dir/model_tfidf.joblib.
        max_features (int): Max TF-IDF features.
        lr_C (float): Inverse regularization strength for Logistic Regression.

    Returns:
        str: Path to the saved model file.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    def load_pairs(fp):
        p = pd.read_csv(fp)
        X = (p["ltable_id"].map(lmap).fillna("") + " [SEP] " + p["rtable_id"].map(rmap).fillna(""))
        y = p["label"].astype(int)
        return X, y
    Xtr, ytr = load_pairs(os.path.join(dataset_dir, "train.csv"))
    vect = TfidfVectorizer(min_df=1, max_features=max_features)
    Xtrv = vect.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=200, C=float(lr_C))
    clf.fit(Xtrv, ytr)
    path = os.path.join(out_dir, "model_tfidf.joblib")
    joblib.dump({"vectorizer": vect, "classifier": clf}, path)
    return path


def predict_edges_with_tfidf(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using the saved TF-IDF + LogisticRegression model.

    Args:
        dataset_dir (str): Directory with left/right text maps.
        people_csv (str): Path to people.csv.
        out_dir (str): Output directory that contains model_tfidf.joblib.
        threshold (float): Decision threshold on positive class.
        max_pairs_per_node (int): Cap candidate pairs per node.

    Returns:
        Tuple[str,str]: Paths to edges.csv and clusters.csv.
    """
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model_tfidf.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing TF-IDF model: {model_path}")
    bundle = joblib.load(model_path)
    vect = bundle["vectorizer"]
    clf = bundle["classifier"]
    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)
    for c in ["birthyear"]:
        people[c] = pd.to_numeric(people.get(c, pd.Series([], dtype=str)), errors="coerce").astype("Int64")
    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    texts = [f"{lmap.get(a, '')} [SEP] {rmap.get(b, '')}" for a, b in cands]
    X = vect.transform(texts)
    probs = clf.predict_proba(X)[:, 1]
    pred = [pair for pair, p in zip(cands, probs) if p >= threshold]
    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)
    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_mlp_classifier(
    dataset_dir: str,
    out_dir: str,
    epochs: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_features: int = 20000,
    device: Optional[str] = None,
) -> str:
    """Trains a TF-IDF + MLP classifier and saves it.

    Args:
        dataset_dir (str): Directory with pair splits.
        out_dir (str): Output directory; model saved under out_dir/model_mlp.
        epochs (int): Number of training epochs.
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout rate.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        max_features (int): Max TF-IDF features.
        device (Optional[str]): 'cpu' or 'cuda'. If None, auto.

    Returns:
        str: Path to the saved model directory.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "model_mlp")
    os.makedirs(model_dir, exist_ok=True)

    lmap, rmap = _load_text_maps(dataset_dir)
    Xtr, ytr = _load_pairs_tfidf(dataset_dir, lmap, rmap, "train")
    Xval, yval = _load_pairs_tfidf(dataset_dir, lmap, rmap, "validation")

    vect = TfidfVectorizer(min_df=1, max_features=max_features)
    Xtrv = vect.fit_transform(Xtr)
    Xvalv = vect.transform(Xval)

    input_dim = Xtrv.shape[1]

    class MLPClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
            super().__init__()
            layers = []
            in_dim = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 2))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    dev = torch.device("cuda" if (device == "cuda" or (device is None and torch.cuda.is_available())) else "cpu")
    model = MLPClassifier(input_dim, hidden_dim, num_layers, dropout).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def sparse_to_tensor(sparse_mat):
        dense = sparse_mat.toarray()
        return torch.FloatTensor(dense).to(dev)

    Xtr_t = sparse_to_tensor(Xtrv)
    ytr_t = torch.LongTensor(ytr.values).to(dev)
    Xval_t = sparse_to_tensor(Xvalv)
    yval_t = torch.LongTensor(yval.values).to(dev)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(Xtr_t), batch_size):
            batch_X = Xtr_t[i:i + batch_size]
            batch_y = ytr_t[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(Xval_t)
            val_loss = criterion(val_outputs, yval_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    best_model_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    model.eval()
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    joblib.dump(vect, os.path.join(model_dir, "vectorizer.joblib"))
    joblib.dump({"input_dim": input_dim, "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout}, 
                os.path.join(model_dir, "config.joblib"))
    return model_dir


def predict_edges_with_mlp(
    dataset_dir: str,
    people_csv: str,
    out_dir: str,
    threshold: float = 0.5,
    max_pairs_per_node: int = 50,
    device: Optional[str] = None,
) -> Tuple[str, str]:
    """Predicts edges using trained MLP classifier.

    Args:
        dataset_dir (str): Directory with left/right CSV for text mapping.
        people_csv (str): Path to people.csv.
        out_dir (str): Output directory with trained model.
        threshold (float): Classification threshold.
        max_pairs_per_node (int): Maximum candidate pairs per node.
        device (Optional[str]): 'cpu' or 'cuda'.

    Returns:
        Tuple[str, str]: Paths to edges.csv and clusters.csv.
    """
    import torch
    import torch.nn as nn
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "model_mlp")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Missing MLP model: {model_dir}")

    config = joblib.load(os.path.join(model_dir, "config.joblib"))
    vect = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    model_state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")

    class MLPClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
            super().__init__()
            layers = []
            in_dim = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 2))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = MLPClassifier(
        config["input_dim"],
        config["hidden_dim"],
        config["num_layers"],
        config["dropout"]
    )
    model.load_state_dict(model_state)
    dev = torch.device("cuda" if (device == "cuda" or (device is None and torch.cuda.is_available())) else "cpu")
    model.to(dev)
    model.eval()

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)
    for c in ["birthyear"]:
        people[c] = pd.to_numeric(people.get(c, pd.Series([], dtype=str)), errors="coerce").astype("Int64")
    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    texts = [f"{lmap.get(a, '')} [SEP] {rmap.get(b, '')}" for a, b in cands]

    batch_size = 256
    scores: List[Tuple[int, int, float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk_texts = texts[i:i + batch_size]
            chunk_cands = cands[i:i + batch_size]
            X = vect.transform(chunk_texts)
            X_t = torch.FloatTensor(X.toarray()).to(dev)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            for (a, b), p in zip(chunk_cands, probs):
                scores.append((a, b, float(p)))

    pred = [(a, b) for (a, b, p) in scores if p >= threshold]
    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)
    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp




def train_splink_model(dataset_dir: str, out_dir: str, threshold: float = 0.5) -> str:
    """Trains a Splink probabilistic record linkage model.

    Args:
        dataset_dir (str): Directory with pair splits.
        out_dir (str): Output directory.
        threshold (float): Probability threshold for matches.

    Returns:
        str: Path to saved model.
    """
    from splink.duckdb.linker import DuckDBLinker
    import json

    os.makedirs(out_dir, exist_ok=True)

    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))

    people = pd.concat([left, right]).drop_duplicates(subset=["id"])
    people = people.rename(columns={"id": "unique_id"})

    settings = {
        "link_type": "dedupe_only",
        "unique_id_column_name": "unique_id",
        "blocking_rules": [],
        "comparisons": [
            {
                "output_column_name": "text",
                "comparison_levels": [
                    {"sql_condition": "text_l = text_r", "label_for_charts": "Exact match"},
                    {"sql_condition": "text_l != text_r", "label_for_charts": "Different"},
                ],
            },
        ],
    }

    linker = DuckDBLinker(people[["unique_id", "text"]], settings)
    linker.estimate_u_using_random_sampling(max_pairs=min(10000, len(train)))
    model_path = os.path.join(out_dir, "splink_model.json")
    with open(model_path, "w") as f:
        json.dump(linker._settings_obj._settings_dict, f, indent=2)
    return model_path

def predict_edges_with_splink(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using Splink model."""
    from splink.duckdb.linker import DuckDBLinker
    import json
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "splink_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Splink model: {model_path}")

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    df = pd.DataFrame([{"unique_id": a, "text": str(lmap.get(a, ""))} for a, _ in cands] + 
                       [{"unique_id": b, "text": str(rmap.get(b, ""))} for _, b in cands]).drop_duplicates(subset=["unique_id"])

    with open(model_path, "r") as f:
        settings = json.load(f)

    linker = DuckDBLinker(df, settings)
    predictions = linker.predict(threshold_match_probability=threshold)
    
    if predictions.empty:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        matches = predictions[predictions["match_probability"] >= threshold]
        if matches.empty:
            edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
        else:
            edges_list = []
            for _, row in matches.iterrows():
                edges_list.append({"id1": int(row["unique_id_l"]), "id2": int(row["unique_id_r"])})
            edges = pd.DataFrame(edges_list)
            edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_xgboost_model(dataset_dir: str, out_dir: str, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1, min_samples_split: int = 2, subsample: float = 1.0) -> str:
    """Trains an XGBoost model for pair classification.

    Args:
        dataset_dir (str): Directory with pair splits.
        out_dir (str): Output directory.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum tree depth.
        learning_rate (float): Learning rate.
        min_samples_split (int): Minimum samples to split a node.
        subsample (float): Fraction of samples for each tree.

    Returns:
        str: Path to saved model.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed: pip install xgboost")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(dataset_dir, "validation.csv"))

    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    def prepare_texts(df):
        return [str(lmap.get(r["ltable_id"], "") + " [SEP] " + rmap.get(r["rtable_id"], "")) for _, r in df.iterrows()]

    vect = TfidfVectorizer(min_df=2, max_features=20000)
    Xtr = vect.fit_transform(prepare_texts(train))
    ytr = train["label"].values
    Xval = vect.transform(prepare_texts(valid))
    yval = valid["label"].values

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_samples_split,
        subsample=subsample,
        random_state=42
    )
    clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=10, verbose=False)

    model_path = os.path.join(out_dir, "xgboost_model.joblib")
    joblib.dump({"classifier": clf, "vectorizer": vect}, model_path)
    return model_path


def predict_edges_with_xgboost(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using XGBoost model."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed: pip install xgboost")
    import joblib
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "xgboost_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing XGBoost model: {model_path}")

    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    vect = bundle["vectorizer"]

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    texts = [f"{lmap.get(a, '')} [SEP] {rmap.get(b, '')}" for a, b in cands]

    X = vect.transform(texts)
    probs = clf.predict_proba(X)[:, 1]
    pred = [pair for pair, p in zip(cands, probs) if p >= threshold]

    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_lightgbm_model(dataset_dir: str, out_dir: str, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1, min_samples_split: int = 2, subsample: float = 1.0) -> str:
    """Trains a LightGBM model for pair classification.

    Args:
        dataset_dir (str): Directory with pair splits.
        out_dir (str): Output directory.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum tree depth.
        learning_rate (float): Learning rate.
        min_samples_split (int): Minimum samples to split a node.
        subsample (float): Fraction of samples for each tree.

    Returns:
        str: Path to saved model.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed: pip install lightgbm")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(dataset_dir, "validation.csv"))

    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    def prepare_texts(df):
        return [str(lmap.get(r["ltable_id"], "") + " [SEP] " + rmap.get(r["rtable_id"], "")) for _, r in df.iterrows()]

    vect = TfidfVectorizer(min_df=2, max_features=20000)
    Xtr = vect.fit_transform(prepare_texts(train))
    ytr = train["label"].values
    Xval = vect.transform(prepare_texts(valid))
    yval = valid["label"].values

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_samples=min_samples_split,
        subsample=subsample,
        random_state=42,
        verbose=-1
    )
    clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])

    model_path = os.path.join(out_dir, "lightgbm_model.joblib")
    joblib.dump({"classifier": clf, "vectorizer": vect}, model_path)
    return model_path


def predict_edges_with_lightgbm(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed: pip install lightgbm")
    import joblib
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "lightgbm_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing LightGBM model: {model_path}")

    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    vect = bundle["vectorizer"]

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    texts = [f"{lmap.get(a, '')} [SEP] {rmap.get(b, '')}" for a, b in cands]

    X = vect.transform(texts)
    probs = clf.predict_proba(X)[:, 1]
    pred = [pair for pair, p in zip(cands, probs) if p >= threshold]

    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_cross_encoder(dataset_dir: str, out_dir: str, epochs: int = 3, device: Optional[str] = None) -> str:
    """Trains a Cross-Encoder BERT model for pair classification."""
    from sentence_transformers import CrossEncoder

    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "model_crossencoder")

    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(dataset_dir, "validation.csv"))

    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    def prepare_data(df):
        texts = [(str(lmap.get(r["ltable_id"], "")), str(rmap.get(r["rtable_id"], ""))) for _, r in df.iterrows()]
        labels = df["label"].astype(int).tolist()
        return texts, labels

    tr_texts, tr_labels = prepare_data(train)
    va_texts, va_labels = prepare_data(valid)

    model = CrossEncoder("distilbert-base-uncased", num_labels=2)
    train_samples = [[t[0], t[1], float(l)] for t, l in zip(tr_texts, tr_labels)]
    eval_samples = [[t[0], t[1], float(l)] for t, l in zip(va_texts, va_labels)]

    model.fit(
        train_dataloader=train_samples,
        epochs=epochs,
        output_path=model_dir,
        warmup_steps=min(100, len(train_samples) // 10),
    )

    return model_dir


def predict_edges_with_crossencoder(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50, device: Optional[str] = None) -> Tuple[str, str]:
    """Predicts edges using Cross-Encoder model."""
    from sentence_transformers import CrossEncoder
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "model_crossencoder")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Missing CrossEncoder model: {model_dir}")

    model = CrossEncoder(model_dir)

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    pairs = [(str(lmap.get(a, "")), str(rmap.get(b, ""))) for a, b in cands]
    scores = model.predict(pairs)

    pred = [(a, b) for (a, b), s in zip(cands, scores) if float(s) >= threshold]
    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp

def train_dedupe_model(dataset_dir: str, out_dir: str, sample_size: int = 15000) -> str:
    """Trains a Dedupe model using active learning."""
    import dedupe
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))

    try:
        import dedupe.variables as v
        fields = [v.String("text")]
    except (ImportError, AttributeError):
        fields = [{"field": "text", "type": "String"}]
    linker = dedupe.RecordLink(fields)

    left_dict = {str(row["id"]): {"text": str(row["text"])} for _, row in left.iterrows()}
    right_dict = {str(row["id"]): {"text": str(row["text"])} for _, row in right.iterrows()}

    labeled_pairs = {}
    for _, row in train.head(min(1000, len(train))).iterrows():
        l_id, r_id, label = str(row["ltable_id"]), str(row["rtable_id"]), row["label"]
        if l_id in left_dict and r_id in right_dict:
            labeled_pairs[(left_dict[l_id], right_dict[r_id])] = (label == 1)

    if labeled_pairs:
        linker.markPairs(labeled_pairs)
        linker.train()

    model_path = os.path.join(out_dir, "dedupe_model.pkl")
    with open(model_path, "wb") as f:
        linker.writeSettings(f)
    return model_path


def predict_edges_with_dedupe(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using Dedupe model."""
    import dedupe
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "dedupe_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Dedupe model: {model_path}")

    with open(model_path, "rb") as f:
        linker = dedupe.StaticRecordLink(f)

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    records_a = [{"id": a, "text": str(lmap.get(a, ""))} for a, _ in cands]
    records_b = [{"id": b, "text": str(rmap.get(b, ""))} for _, b in cands]

    matches = linker.match(records_a, records_b, threshold=threshold)

    if not matches:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges_list = []
        for match in matches:
            edges_list.append({"id1": int(match[0]["id"]), "id2": int(match[1]["id"])})
        edges = pd.DataFrame(edges_list)
        edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_recordlinkage_model(dataset_dir: str, out_dir: str) -> str:
    """Trains a RecordLinkage model."""
    import recordlinkage as rl
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))

    df_a = left.set_index("id")
    df_b = right.set_index("id")

    if len(df_a) == 0 or len(df_b) == 0:
        raise ValueError("Empty datasets. Increase sample_frac.")
    
    indexer = rl.Index()
    indexer.block("text")
    candidate_pairs = indexer.index(df_a, df_b)
    
    if len(candidate_pairs) == 0:
        raise ValueError("No candidate pairs found. Increase sample_frac.")

    compare = rl.Compare()
    compare.string("text", "text", method="jarowinkler", threshold=0.85, label="text_sim")
    compare.exact("text", "text", label="text_exact")

    features = compare.compute(candidate_pairs, df_a, df_b)

    train_index = pd.MultiIndex.from_tuples([(r["ltable_id"], r["rtable_id"]) for _, r in train.iterrows()])
    train_labels = train.set_index(["ltable_id", "rtable_id"])["label"]
    
    common_pairs = candidate_pairs.intersection(train_index)
    if len(common_pairs) > 0:
        X_train = features.loc[common_pairs].values
        y_train = train_labels.loc[common_pairs].values
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
    else:
        if len(features) > 0:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(features.values, [0] * len(features))
        else:
            raise ValueError("No features computed. Increase sample_frac.")

    model_path = os.path.join(out_dir, "recordlinkage_model.joblib")
    joblib.dump({"classifier": clf, "indexer": indexer, "compare": compare}, model_path)
    return model_path


def predict_edges_with_recordlinkage(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using RecordLinkage model."""
    import recordlinkage as rl
    import joblib
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "recordlinkage_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing RecordLinkage model: {model_path}")

    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    indexer = bundle["indexer"]
    compare = bundle["compare"]

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    df_a = pd.DataFrame([{"id": a, "text": str(lmap.get(a, ""))} for a, _ in cands]).set_index("id")
    df_b = pd.DataFrame([{"id": b, "text": str(rmap.get(b, ""))} for _, b in cands]).set_index("id")

    candidate_pairs = indexer.index(df_a, df_b)
    if len(candidate_pairs) > 0:
        features = compare.compute(candidate_pairs, df_a, df_b)
        probs = clf.predict_proba(features.values)[:, 1]
        pred_pairs = candidate_pairs[probs >= threshold]
    else:
        pred_pairs = candidate_pairs

    if pred_pairs.empty:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges_list = [{"id1": int(pair[0]), "id2": int(pair[1])} for pair in pred_pairs]
        edges = pd.DataFrame(edges_list)
        edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_random_forest_model(dataset_dir: str, out_dir: str, n_estimators: int = 200, max_depth: int = 10, min_samples_split: int = 5, min_samples_leaf: int = 2, max_features: str = "sqrt") -> str:
    """Trains a Random Forest model for pair classification.

    Args:
        dataset_dir (str): Directory with pair splits.
        out_dir (str): Output directory.
        n_estimators (int): Number of trees.
        max_depth (int): Maximum tree depth.
        min_samples_split (int): Minimum samples to split a node.
        min_samples_leaf (int): Minimum samples in a leaf.
        max_features (str): Number of features to consider.

    Returns:
        str: Path to saved model.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(dataset_dir, "validation.csv"))

    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    def prepare_texts(df):
        return [str(lmap.get(r["ltable_id"], "") + " [SEP] " + rmap.get(r["rtable_id"], "")) for _, r in df.iterrows()]

    vect = TfidfVectorizer(min_df=2, max_features=20000)
    Xtr = vect.fit_transform(prepare_texts(train))
    ytr = train["label"].values
    Xval = vect.transform(prepare_texts(valid))
    yval = valid["label"].values

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)

    model_path = os.path.join(out_dir, "random_forest_model.joblib")
    joblib.dump({"classifier": clf, "vectorizer": vect}, model_path)
    return model_path


def predict_edges_with_random_forest(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using Random Forest model.

    Args:
        dataset_dir (str): Directory with left/right text maps.
        people_csv (str): Path to people.csv.
        out_dir (str): Output directory.
        threshold (float): Match probability threshold.
        max_pairs_per_node (int): Cap candidate pairs per node.

    Returns:
        Tuple[str,str]: Paths to edges.csv and clusters.csv.
    """
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "random_forest_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Random Forest model: {model_path}")

    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    vect = bundle["vectorizer"]

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    texts = [f"{lmap.get(a, '')} [SEP] {rmap.get(b, '')}" for a, b in cands]

    X = vect.transform(texts)
    probs = clf.predict_proba(X)[:, 1]
    pred = [pair for pair, p in zip(cands, probs) if p >= threshold]

    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp


def train_gradient_boosting_model(dataset_dir: str, out_dir: str, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6, min_samples_split: int = 10, min_samples_leaf: int = 4, subsample: float = 0.8) -> str:
    """Trains a Gradient Boosting model for pair classification.

    Args:
        dataset_dir (str): Directory with pair splits.
        out_dir (str): Output directory.
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate.
        max_depth (int): Maximum tree depth.
        min_samples_split (int): Minimum samples to split a node.
        min_samples_leaf (int): Minimum samples in a leaf.
        subsample (float): Fraction of samples for each tree.

    Returns:
        str: Path to saved model.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(dataset_dir, "validation.csv"))

    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()

    def prepare_texts(df):
        return [str(lmap.get(r["ltable_id"], "") + " [SEP] " + rmap.get(r["rtable_id"], "")) for _, r in df.iterrows()]

    vect = TfidfVectorizer(min_df=2, max_features=20000)
    Xtr = vect.fit_transform(prepare_texts(train))
    ytr = train["label"].values
    Xval = vect.transform(prepare_texts(valid))
    yval = valid["label"].values

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=42
    )
    clf.fit(Xtr, ytr)

    model_path = os.path.join(out_dir, "gradient_boosting_model.joblib")
    joblib.dump({"classifier": clf, "vectorizer": vect}, model_path)
    return model_path


def predict_edges_with_gradient_boosting(dataset_dir: str, people_csv: str, out_dir: str, threshold: float = 0.5, max_pairs_per_node: int = 50) -> Tuple[str, str]:
    """Predicts edges using Gradient Boosting model.

    Args:
        dataset_dir (str): Directory with left/right text maps.
        people_csv (str): Path to people.csv.
        out_dir (str): Output directory.
        threshold (float): Match probability threshold.
        max_pairs_per_node (int): Cap candidate pairs per node.

    Returns:
        Tuple[str,str]: Paths to edges.csv and clusters.csv.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    import joblib
    from .edits import rebuild_clusters_from_edges, canonicalize_edges

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "gradient_boosting_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Gradient Boosting model: {model_path}")

    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    vect = bundle["vectorizer"]

    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if "id" not in people.columns:
        raise ValueError("people.csv missing id column")
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)

    cands = _candidate_pairs(people, max_pairs_per_node=max_pairs_per_node)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    texts = [f"{lmap.get(a, '')} [SEP] {rmap.get(b, '')}" for a, b in cands]

    X = vect.transform(texts)
    probs = clf.predict_proba(X)[:, 1]
    pred = [pair for pair, p in zip(cands, probs) if p >= threshold]

    if not pred:
        edges = pd.DataFrame(columns=["id1", "id2"], dtype=int)
    else:
        edges = pd.DataFrame(pred, columns=["id1", "id2"], dtype=int)
        edges = canonicalize_edges(edges)

    edges_fp = os.path.join(out_dir, "edges.csv")
    edges.to_csv(edges_fp, index=False)
    clusters = rebuild_clusters_from_edges(edges, people["id"].astype(int).tolist())
    cl_fp = os.path.join(out_dir, "clusters.csv")
    clusters.to_csv(cl_fp, index=False)
    return edges_fp, cl_fp
