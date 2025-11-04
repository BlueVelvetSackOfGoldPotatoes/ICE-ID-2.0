"""
Thin, importable wrapper around the standalone 'analysis' script so Streamlit
can call evaluation programmatically and display results inline.

Usage:
    from dashboard.eval_api import analyze_backends
    results = analyze_backends(run_dir, people_csv, labels_csv, backends=None)
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional

import pandas as pd
from .schemas import LabelsSchema


def _load_gt(people_csv: str, labels_csv: str, schema: LabelsSchema | None = None) -> pd.DataFrame:
    """Loads and merges people and labels CSVs to create a ground truth mapping.

    It creates a DataFrame mapping each record 'id' to a 'gt_cluster' ID,
    prioritizing the 'person' column from people_csv if available.

    Args:
        people_csv (str): Path to the people.csv file.
        labels_csv (str): Path to the labels CSV file (e.g., ground truth).

    Returns:
        pd.DataFrame: A DataFrame with 'id' and 'gt_cluster' columns.
    """
    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False, low_memory=False)

    if schema is None:
        for c in ["id", "person"]:
            if c not in people.columns:
                people[c] = ""
        for c in ["id", "bi_einstaklingur"]:
            if c not in labels.columns:
                labels[c] = ""
    else:
        if "id" not in people.columns:
            people["id"] = ""
        if "person" not in people.columns:
            people["person"] = ""
        if schema.id_col not in labels.columns:
            labels[schema.id_col] = ""
        if schema.cluster_col not in labels.columns:
            labels[schema.cluster_col] = ""

    people["id"] = pd.to_numeric(people["id"], errors="coerce")
    people["person"] = pd.to_numeric(people["person"], errors="coerce")
    if schema is None:
        labels_id_col = "id"
        labels_cluster_col = "bi_einstaklingur"
    else:
        labels_id_col = schema.id_col
        labels_cluster_col = schema.cluster_col
    labels[labels_id_col] = pd.to_numeric(labels[labels_id_col], errors="coerce")
    labels[labels_cluster_col] = pd.to_numeric(labels[labels_cluster_col], errors="coerce")

    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype(int)

    # Prefer 'person' if present, else labels
    gt = people[["id", "person"]].copy()
    have_person = gt["person"].notna().sum() > 0 and (gt["person"] >= 0).sum() > 0
    if have_person:
        gt["gt_cluster"] = gt["person"]
    else:
        if schema is None:
            gt = gt.merge(labels[["id", "bi_einstaklingur"]], on="id", how="left")
            gt["gt_cluster"] = gt["bi_einstaklingur"]
        else:
            tmp = labels[[labels_id_col, labels_cluster_col]].copy()
            tmp = tmp.rename(columns={labels_id_col: "id", labels_cluster_col: "gt_cluster"})
            gt = gt.merge(tmp, on="id", how="left")
    gt = gt[["id", "gt_cluster"]].copy()
    return gt


def _wilson_interval(p_hat: float, n: int, z: float = 1.96):
    """Calculates the Wilson score confidence interval for a binomial proportion.

    Args:
        p_hat (float): The observed proportion of successes.
        n (int): The total number of trials.
        z (float, optional): The z-score for the desired confidence level. Defaults to 1.96 (95%).

    Returns:
        tuple[Optional[float], Optional[float]]: A tuple containing the lower and upper bounds of the confidence interval.
    """
    if n == 0:
        return (None, None)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _parse_members(s):
    """Parses a cluster members column into lists of integers.

    Handles both formats:
    - New format: '1;2;3' (semicolon-delimited)
    - Old format: '[1, 2, 3]' (Python list representation)

    Args:
        s (str): A string representing cluster members.

    Returns:
        list[int]: A list of integer member IDs.
    """
    if isinstance(s, str) and s:
        if s.startswith('[') and s.endswith(']'):
            import ast
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                s_clean = s.strip('[]').replace(' ', '')
                return [int(x) for x in s_clean.split(',') if x]
        else:
            return [int(x) for x in s.split(";") if x]
    return []


def _analyze_backend(bname: str, run_dir: str, gt: pd.DataFrame) -> Dict:
    """Performs a detailed analysis for a single backend's output.

    It calculates precision on labeled data, identifies novel edges, and computes
    cluster purity and size statistics if cluster data is available.

    Args:
        bname (str): The name of the backend (subfolder name).
        run_dir (str): The parent directory containing the backend folder.
        gt (pd.DataFrame): The ground truth DataFrame with 'id' and 'gt_cluster' columns.

    Returns:
        Dict: A dictionary containing a comprehensive set of evaluation metrics.
    """
    method_dir = run_dir if bname in (".", "", None) else os.path.join(run_dir, bname)
    edges_path = os.path.join(method_dir, "edges.csv")
    clusters_path = os.path.join(method_dir, "clusters.csv")
    out = {"backend": bname if bname not in (".", "", None) else os.path.basename(method_dir) or "auto"}

    if not os.path.exists(edges_path):
        out["error"] = f"missing {edges_path}"
        return out

    edges = pd.read_csv(edges_path)
    edges["id1"] = edges["id1"].astype(int)
    edges["id2"] = edges["id2"].astype(int)

    gt_map = dict(zip(gt["id"].astype(int), gt["gt_cluster"]))

    def is_lab(i):
        v = gt_map.get(int(i))
        return (v is not None) and (not pd.isna(v))

    def same_ent(a, b):
        va, vb = gt_map.get(int(a)), gt_map.get(int(b))
        return (not pd.isna(va)) and (not pd.isna(vb)) and (va == vb)

    n_total = len(edges)
    
    id1_lab = edges["id1"].apply(is_lab)
    id2_lab = edges["id2"].apply(is_lab)
    
    both_lab_mask = id1_lab & id2_lab
    exactly_one_lab_mask = id1_lab ^ id2_lab
    none_lab_mask = ~(both_lab_mask | exactly_one_lab_mask)

    pred_pairs_labeled = int(both_lab_mask.sum())
    if pred_pairs_labeled > 0:
        labeled_edges = edges[both_lab_mask]
        id1_gt = labeled_edges["id1"].astype(int).map(gt_map)
        id2_gt = labeled_edges["id2"].astype(int).map(gt_map)
        tp_labeled = int(((id1_gt == id2_gt) & id1_gt.notna() & id2_gt.notna()).sum())
    else:
        tp_labeled = 0
    fp_labeled = int(pred_pairs_labeled - tp_labeled)
    prec_labeled = tp_labeled / pred_pairs_labeled if pred_pairs_labeled else None
    llo, lhi = _wilson_interval(prec_labeled, pred_pairs_labeled) if prec_labeled is not None else (None, None)

    semi_novel = int(exactly_one_lab_mask.sum())
    novel = int(none_lab_mask.sum())
    total_unlabeled_side = semi_novel + novel

    est_tp_unlabeled = prec_labeled * total_unlabeled_side if prec_labeled is not None else None
    est_tp_unlabeled_lo = llo * total_unlabeled_side if llo is not None else None
    est_tp_unlabeled_hi = lhi * total_unlabeled_side if lhi is not None else None

    touched_nodes = pd.concat([edges["id1"], edges["id2"]]).unique()
    n_nodes = len(touched_nodes)
    touched_nodes_lab = pd.Series(touched_nodes).apply(is_lab)
    n_nodes_labeled = int(touched_nodes_lab.sum())
    n_nodes_unlabeled = n_nodes - n_nodes_labeled

    purity_weighted = None
    pct_clusters_pure90 = None
    clusters_all_unlabeled = None
    size_stats = {}
    if os.path.exists(clusters_path):
        cl = pd.read_csv(clusters_path)
        cl["members"] = cl["members"].apply(_parse_members)
        sizes = cl["size"].astype(int).tolist()
        if sizes:
            size_stats = {
                "n_clusters": int(len(cl)),
                "singleton_rate": float(sum(1 for s in sizes if s == 1) / len(sizes)),
                "mean_size": float(sum(sizes) / len(sizes)),
                "p90_size": int(pd.Series(sizes).quantile(0.9)),
                "p99_size": int(pd.Series(sizes).quantile(0.99)),
                "max_size": int(max(sizes)),
            }
        purities = []
        all_unlab = 0
        for members in cl["members"]:
            labs = [gt_map.get(int(i)) for i in members if is_lab(i)]
            if len(labs) == 0:
                all_unlab += 1
                continue
            try:
                counts = Counter([int(float(x)) for x in labs if pd.notna(x)])
                if not counts:
                    continue
                maj = counts.most_common(1)[0][1]
                purities.append(maj / len(labs))
            except (ValueError, TypeError):
                continue
        if purities:
            weights = []
            vals = []
            for members in cl["members"]:
                labs = [gt_map.get(int(i)) for i in members if is_lab(i)]
                if len(labs) == 0:
                    continue
                try:
                    counts = Counter([int(float(x)) for x in labs if pd.notna(x)])
                    if not counts:
                        continue
                    maj = counts.most_common(1)[0][1]
                    vals.append(maj / len(labs))
                    weights.append(len(labs))
                except (ValueError, TypeError):
                    continue
            if weights:
                purity_weighted = float((sum(v * w for v, w in zip(vals, weights))) / sum(weights))
            pct_clusters_pure90 = float(sum(1 for v in purities if v >= 0.9) / len(purities))
        clusters_all_unlabeled = int(all_unlab)

    out.update({
        "edges_total": n_total,
        "pred_pairs_labeled": pred_pairs_labeled,
        "tp_labeled": tp_labeled,
        "fp_labeled": fp_labeled,
        "prec_labeled": round(prec_labeled, 6) if prec_labeled is not None else None,
        "prec_labeled_wilson95": (round(llo, 6), round(lhi, 6)) if llo is not None else None,
        "semi_novel_edges": semi_novel,
        "novel_edges": novel,
        "est_true_new_edges": None if est_tp_unlabeled is None else int(round(est_tp_unlabeled)),
        "est_true_new_edges_95ci": None if est_tp_unlabeled_lo is None else (int(round(est_tp_unlabeled_lo)), int(round(est_tp_unlabeled_hi))),
        "nodes_with_any_match": n_nodes,
        "nodes_with_any_match_labeled": n_nodes_labeled,
        "nodes_with_any_match_unlabeled": n_nodes_unlabeled,
        "cluster_stats": {
            **size_stats,
            "weighted_purity_on_labeled": purity_weighted,
            "pct_clusters_pure90_on_labeled": pct_clusters_pure90,
            "clusters_all_unlabeled": clusters_all_unlabeled
        }
    })
    return out


def _detect_backends(run_dir: str) -> List[str]:
    """Automatically detects backend subfolders within a run directory.

    A folder is considered a backend if it contains 'edges.csv' or 'clusters.csv'.
    If no subfolders are found, it checks if the run_dir itself is a backend.

    Args:
        run_dir (str): The path to the run directory to scan.

    Returns:
        List[str]: A sorted list of detected backend names.
    """
    cand: List[str] = []
    if not os.path.isdir(run_dir):
        return cand
    for name in os.listdir(run_dir):
        p = os.path.join(run_dir, name)
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "clusters.csv")) or os.path.exists(os.path.join(p, "edges.csv")):
                cand.append(name)
    if not cand:
        if os.path.exists(os.path.join(run_dir, "clusters.csv")) or os.path.exists(os.path.join(run_dir, "edges.csv")):
            cand = ["."]
    return sorted(cand)


def analyze_backends(run_dir: str, people_csv: str, labels_csv: str, backends: Optional[Iterable[str]] = None, labels_schema: LabelsSchema | None = None) -> Dict:
    """Runs evaluation for multiple backends and returns results as a dictionary.

    This is the main entry point for programmatically evaluating backend outputs.
    It orchestrates loading ground truth, finding backends, and analyzing each one.

    Args:
        run_dir (str): The directory containing backend subfolders.
        people_csv (str): Path to the main people.csv file.
        labels_csv (str): Path to the ground truth labels file.
        backends (Optional[Iterable[str]], optional): A specific list of backend names to
            analyze. If None, backends are auto-detected. Defaults to None.

    Returns:
        Dict: A dictionary where keys are backend names and values are their analysis results.
    """
    if backends is None:
        backends = _detect_backends(run_dir)

    gt = _load_gt(people_csv, labels_csv, schema=labels_schema)
    results: Dict[str, Dict] = {}
    for b in backends:
        results[b] = _analyze_backend(b, run_dir, gt)
    # If nothing found, include a sentinel to help the UI show a message
    if not results:
        results["_debug"] = {"error": f"No backends found in {run_dir}. Expect subdirs like gbdt/, logreg/ with edges.csv (and optionally clusters.csv)."}
    return results
