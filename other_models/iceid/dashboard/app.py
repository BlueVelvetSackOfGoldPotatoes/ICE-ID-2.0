"""Streamlit dashboard with tabs for Training, Evaluation, and Inspect & Edit.

Run: streamlit run other_models/iceid/dashboard/app.py
"""
from __future__ import annotations

import os
import sys
import inspect
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Path setup so imports & defaults work no matter where you launch streamlit run
# --------------------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../other_models/iceid/dashboard
ICEID_ROOT = os.path.dirname(THIS_DIR)                 # .../other_models/iceid  (← this is your project root now)
REPO_ROOT = os.path.dirname(os.path.dirname(ICEID_ROOT))  # .../ICE-ID-2.0 (top-level repo that contains other_models/)
if ICEID_ROOT not in sys.path:
    sys.path.insert(0, ICEID_ROOT)

# Local imports
from dashboard.train_api import (  # type: ignore
    build_command,
    run_pipeline_streaming,
    scan_artifacts,
    prime_out_dir,
)
from dashboard.eval_api import analyze_backends  # type: ignore
from dashboard.inspector_tab import show_inspector_tab  # type: ignore

# --------------------------------------------------------------------------------------
# Small helpers for the Evaluation tab
# --------------------------------------------------------------------------------------
def _metric_explanations() -> Dict[str, str]:
    """Provides short definitions for each metric key shown in the UI.

    Returns:
        Dict[str, str]: A dictionary mapping metric names to their explanations.
    """
    return {
        "prec_labeled": "Precision on labeled pairs (true positives / (true positives + false positives))",
        "rec_labeled": "Recall on labeled pairs (true positives / (true positives + false negatives))",
        "f1_labeled": "F1 score on labeled pairs (harmonic mean of precision and recall)",
        "prec_labeled_wilson95": "Wilson 95% confidence interval for precision on labeled pairs",
        "edges_total": "Total number of predicted edges/pairs",
        "semi_novel_edges": "Edges where one node is labeled, one is unlabeled",
        "novel_edges": "Edges where both nodes are unlabeled",
        "est_true_new_edges": "Estimated number of true edges among novel/semi-novel predictions",
        "nodes_with_any_match": "Number of nodes that have at least one predicted match",
        "cluster_stats": "Statistics about the clustering results (purity, number of clusters, etc.)",
    }

def _train_param_explanations() -> Dict[str, str]:
    """Provides short definitions for training parameters shown in the UI.

    Returns:
        Dict[str, str]: A dictionary mapping parameter names to their explanations.
    """
    return {
        "people_csv": "Path to the people.csv file containing individual records",
        "labels_csv": "Path to the labels.csv file containing ground truth cluster labels",
        "out_dir": "Output directory where results will be saved",
        "backends": "Comma-separated list of backends to run (e.g., 'gbdt,logreg')",
        "num_shards": "Number of shards to split the data into for parallel processing",
        "shard_id": "ID of the current shard (0-based)",
        "n_workers": "Number of worker processes for parallel processing",
        "max_block_size": "Maximum size of a blocking group",
        "max_pairs_per_block": "Maximum number of pairs to generate per block",
        "gbdt_neg_ratio": "Ratio of negative to positive samples for GBDT training",
        "thresh_grid": "Number of threshold values to test for optimal threshold selection",
        "seed": "Random seed for reproducibility",
        "sample_frac": "Fraction of data to use (1.0 = all data, 0.1 = 10%)",
        "fn_prefix": "Length of first name prefix for blocking",
        "pat_prefix": "Length of patronymic prefix for blocking",
        "year_bucket_width": "Width of year buckets for blocking",
        "dual_blocking": "Whether to use dual blocking (both forward and reverse)",
        "soft_filter_max_year_diff": "Maximum year difference for soft filtering",
        "soft_filter_sex": "Whether to apply sex-based soft filtering",
        "preview_window": "Fraction of data to use for preview/validation",
        "preview_limit": "Maximum number of samples for preview",
    }

def _bytes_to_human(n: int) -> str:
    """Converts bytes to human-readable format.

    Args:
        n (int): Number of bytes.

    Returns:
        str: Human-readable string (e.g., "1.2 MB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def _grade(value: Optional[float], bands: List[Tuple[float, str]]) -> str:
    """Grades a numerical value based on threshold bands.

    Args:
        value (Optional[float]): The numerical value to grade. Can be None.
        bands (List[Tuple[float, str]]): A list of (threshold, label) tuples, sorted in descending order of threshold.

    Returns:
        str: The corresponding label ('poor' if below all thresholds, 'unknown' if value is None).
    """
    if value is None:
        return "unknown"
    for th, lab in bands:
        if value >= th:
            return lab
    return "poor"

def _fmt_pct(x: Optional[float]) -> str:
    """Formats a float as a percentage string.

    Args:
        x (Optional[float]): The float to format (e.g., 0.951). Returns '—' if None.

    Returns:
        str: The formatted percentage string (e.g., "95.1%").
    """
    return "—" if x is None else f"{100.0 * x:.1f}%"

def _auto_conclusions(r: Dict) -> Dict[str, str]:
    """Generates heuristic, human-readable conclusions for a backend's metrics.

    Args:
        r (Dict): A dictionary of evaluation results for a single backend.

    Returns:
        Dict[str, str]: A dictionary mapping metric names to their automated interpretation.
    """
    out: Dict[str, str] = {}
    # Pull fields
    prec = r.get("prec_labeled")
    ci = r.get("prec_labeled_wilson95") or (None, None)
    ci_lo, ci_hi = ci
    ci_width = None if (ci_lo is None or ci_hi is None) else (ci_hi - ci_lo)

    edges_total = r.get("edges_total")
    semi_novel = r.get("semi_novel_edges")
    novel = r.get("novel_edges")
    est_new = r.get("est_true_new_edges")
    nodes_any = r.get("nodes_with_any_match")

    cl = (r.get("cluster_stats") or {})
    n_clusters = cl.get("n_clusters")
    purity_w = cl.get("weighted_purity_on_labeled")

    # Precision
    grade = _grade(prec, [(0.95, "excellent"), (0.90, "good"), (0.80, "ok")])
    if prec is None:
        out["prec_labeled"] = "Precision unknown on labeled pairs (likely zero labeled overlap). Consider checking label file alignment."
    else:
        if grade in ("excellent", "good"):
            msg = f"{grade.capitalize()} precision {_fmt_pct(prec)} on labeled pairs."
        elif grade == "ok":
            msg = f"Moderate precision {_fmt_pct(prec)}; expect some false positives—consider stricter thresholding or feature tweaks."
        else:
            msg = f"Low precision {_fmt_pct(prec)}; many predicted pairs among labeled items are wrong."
        if ci_width is not None:
            if ci_width <= 0.1:
                msg += f" CI [{_fmt_pct(ci_lo)}, {_fmt_pct(ci_hi)}] is tight → estimate is stable."
            else:
                msg += f" CI [{_fmt_pct(ci_lo)}, {_fmt_pct(ci_hi)}] is wide → limited labeled volume or unstable thresholds."
        out["prec_labeled"] = msg

    # Edges volume
    if isinstance(edges_total, int):
        if edges_total == 0:
            out["edges_total"] = "No predicted pairs. Check thresholds or model outputs."
        elif edges_total < 100:
            out["edges_total"] = f"Low volume ({edges_total}). Might be too conservative or data is small."
        else:
            out["edges_total"] = f"{edges_total} predicted pairs. Reasonable volume for downstream clustering."
    else:
        out["edges_total"] = "Predicted pair count unavailable."

    # Novel vs semi-novel
    if isinstance(semi_novel, int) and isinstance(novel, int):
        total_unlab = semi_novel + novel
        if total_unlab == 0:
            out["novel_edges"] = "No unlabeled-side discoveries. Model may be over-conservative on new regions."
        else:
            if est_new is None or prec is None:
                out["novel_edges"] = f"{total_unlab} potential new edges (semi-novel + novel). Precision unknown—review samples."
            else:
                exp_true = int(est_new)
                if prec >= 0.9:
                    out["novel_edges"] = f"{total_unlab} potential new edges; ~{exp_true} expected true → strong discovery potential."
                elif prec >= 0.8:
                    out["novel_edges"] = f"{total_unlab} potential new edges; ~{exp_true} expected true → decent, but verify borderline cases."
                else:
                    out["novel_edges"] = f"{total_unlab} potential new edges; ~{exp_true} expected true → many may be spurious."
    else:
        out["novel_edges"] = "Novel/semi-novel edge counts unavailable."

    # Clusters
    if isinstance(n_clusters, int):
        if n_clusters == 0:
            out["cluster_stats"] = "No clusters formed. Check if edges were generated and clustering logic."
        elif n_clusters == 1:
            out["cluster_stats"] = f"Single cluster ({n_clusters}). May indicate over-clustering or insufficient data."
        else:
            out["cluster_stats"] = f"{n_clusters} clusters formed."
            if purity_w is not None:
                grade_p = _grade(purity_w, [(0.95, "excellent"), (0.90, "good"), (0.80, "ok")])
                out["cluster_stats"] += f" Weighted purity: {_fmt_pct(purity_w)} ({grade_p})."
    else:
        out["cluster_stats"] = "Cluster statistics unavailable."

    return out

@st.cache_data(ttl=60)
def _get_available_run_directories() -> List[str]:
    """Scans the runs directory to find available run directories.

    Returns:
        List[str]: A list of available run directory paths, sorted by modification time (newest first).
    """
    runs_dir = os.path.join(ICEID_ROOT, "runs")
    if not os.path.exists(runs_dir):
        return []

    run_dirs = []
    for item in os.listdir(runs_dir):
        item_path = os.path.join(runs_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains any backend subdirectories or is a shard directory
            has_backends = False
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    # Check if it's a backend directory (has edges.csv or clusters.csv)
                    if (os.path.exists(os.path.join(subitem_path, "edges.csv")) or
                        os.path.exists(os.path.join(subitem_path, "clusters.csv"))):
                        has_backends = True
                        break
                    # Or if it's a shard directory
                    if subitem.startswith("shard_"):
                        has_backends = True
                        break

            if has_backends:
                run_dirs.append(item_path)

    # Sort by modification time (newest first)
    run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return run_dirs

@st.cache_data(ttl=60)
def _get_available_backends(run_dir: str) -> List[str]:
    """Scans a run directory to find available backend subdirectories.

    Args:
        run_dir (str): Path to the run directory to scan.

    Returns:
        List[str]: A list of available backend names, sorted alphabetically.
    """
    if not os.path.exists(run_dir):
        return []

    backends = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a backend directory (has edges.csv or clusters.csv)
            if (os.path.exists(os.path.join(item_path, "edges.csv")) or
                os.path.exists(os.path.join(item_path, "clusters.csv"))):
                backends.append(item)

    backends.sort()
    return backends

# --------------------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="ICE-ID: Clustering Dashboard", layout="wide")
st.title("ICE-ID Dashboard")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab_train, tab_eval, tab_inspect, tab_data_editor = st.tabs(["Training", "Evaluation", "Inspect & Edit", "Data Editor"])

# --------------------------------------------------------------------------------------
# TRAINING TAB
# --------------------------------------------------------------------------------------
with tab_train:
    st.subheader("Run Pipeline")
    explain_train = _train_param_explanations()

    # Model type selector
    model_type = st.selectbox(
        "Model Type",
        ["ICE-ID Pipeline", "External Models"],
        help="Select the type of model to train"
    )

    if model_type == "ICE-ID Pipeline":
        # ---- Reuse precomputed artifacts (NEW) ----
        with st.expander("Reuse precomputed artifacts (blocks/candidates/features)", expanded=False):
            src_shard_dir = st.text_input(
                "Existing shard directory to reuse (source)",
                value=os.path.join(ICEID_ROOT, "runs", "hundred_loose_dual", "shard_s0_of_1"),
                help="Point to a previous shard directory that already contains heavy pre-stage files."
            )
            colp1, colp2 = st.columns(2)
            with colp1:
                if st.button("Preview reusable artifacts", help="View the artifacts available for reuse from the selected shard directory"):
                    dirs, files = scan_artifacts(src_shard_dir)
                    if not dirs and not files:
                        st.warning("No reusable artifacts detected in that folder.")
                    else:
                        if dirs:
                            st.markdown("**Reusable directories**")
                            st.dataframe(pd.DataFrame(
                                [{"path": d["path"], "size": d["size"]} for d in dirs]
                            ).assign(size_readable=lambda df: df["size"].map(lambda x: _bytes_to_human(x)))[["path","size_readable"]])
                        if files:
                            st.markdown("**Reusable files**")
                            st.dataframe(pd.DataFrame(
                                [{"path": f["path"], "size": f["size"]} for f in files]
                            ).assign(size_readable=lambda df: df["size"].map(lambda x: _bytes_to_human(x)))[["path","size_readable"]])
            with colp2:
                st.caption("Artifacts copied only when you click **Prime output shard** below, after setting your run parameters.")

        with st.form("train_form", clear_on_submit=False):
            # Paths (match your repo layout)
            default_people = os.path.join(ICEID_ROOT, "raw_data", "people.csv")
            default_labels = os.path.join(ICEID_ROOT, "raw_data", "manntol_einstaklingar_new.csv")
            # Use an existing runs root by default
            default_out = os.path.join(ICEID_ROOT, "runs", "hundred_loose_dual")

            # Training entry module (so you can point at a real entry point in this repo)
            entry_module = st.text_input(
                "Training entry module (python -m …)",
                value="other_models.main",
                help="Change this to the actual module that runs training, e.g. 'iceid.train_main' if that exists."
            )

            people_csv = st.text_input(
                "people.csv path", value=default_people, help=explain_train["people_csv"]
            )
            labels_csv = st.text_input(
                "labels.csv path (bi_einstaklingur)", value=default_labels, help=explain_train["labels_csv"]
            )
            out_dir = st.text_input(
                "Output directory (parent of shard_* folders)", value=default_out, help=explain_train["out_dir"]
            )

            # Core params
            backends = st.text_input(
                "Backends (comma-separated)", value="gbdt,logreg", help=explain_train["backends"]
            )
            device_choice = st.selectbox(
                "Compute device",
                ["Auto (GPU if available)", "CPU only", "GPU:0"],
                index=0,
                help="Also sets CUDA_VISIBLE_DEVICES for you."
            )
            num_shards = st.number_input(
                "num_shards", min_value=1, value=1, step=1, help=explain_train["num_shards"]
            )
            shard_id = st.number_input(
                "shard_id", min_value=0, value=0, step=1, help=explain_train["shard_id"]
            )
            n_workers = st.number_input(
                "n_workers", min_value=1, value=4, step=1, help=explain_train["n_workers"]
            )
            max_block_size = st.number_input(
                "max_block_size", min_value=1, value=5000, step=1, help=explain_train["max_block_size"]
            )
            max_pairs_per_block = st.number_input(
                "max_pairs_per_block", min_value=1, value=250000, step=1000, help=explain_train["max_pairs_per_block"]
            )
            gbdt_neg_ratio = st.number_input(
                "gbdt_neg_ratio", min_value=0.0, value=2.0, step=0.1, help=explain_train["gbdt_neg_ratio"]
            )
            thresh_grid = st.number_input(
                "thresh_grid", min_value=1, value=101, step=2, help=explain_train["thresh_grid"]
            )
            seed = st.number_input(
                "seed", min_value=0, value=42, step=1, help=explain_train["seed"]
            )
            sample_frac = st.number_input(
                "sample_frac", min_value=0.0, max_value=1.0, value=1.0, step=0.05, help=explain_train["sample_frac"]
            )
            fn_prefix = st.number_input(
                "fn_prefix", min_value=0, value=2, step=1, help=explain_train["fn_prefix"]
            )
            pat_prefix = st.number_input(
                "pat_prefix", min_value=0, value=3, step=1, help=explain_train["pat_prefix"]
            )
            year_bucket_width = st.number_input(
                "year_bucket_width", min_value=1, value=5, step=1, help=explain_train["year_bucket_width"]
            )
            soft_filter_max_year_diff = st.number_input(
                "soft_filter_max_year_diff", min_value=0, value=15, step=1, help=explain_train["soft_filter_max_year_diff"]
            )

            # Booleans
            dual_blocking = st.checkbox(
                "dual_blocking", value=False, help=explain_train["dual_blocking"]
            )
            soft_filter_sex = st.checkbox(
                "soft_filter_sex", value=True, help=explain_train["soft_filter_sex"]
            )

            # Preview params
            preview_window = st.number_input(
                "preview_window", min_value=0.0, max_value=1.0, value=0.05, step=0.01, help=explain_train["preview_window"]
            )
            preview_limit = st.number_input(
                "preview_limit", min_value=0, value=500, step=50, help=explain_train["preview_limit"]
            )

            # Priming control (does the copy BEFORE clicking Run)
            prime_btn = st.form_submit_button("Prime output shard with precomputed artifacts", help="Copy artifacts from the source directory to the target shard before training")
            run_btn = st.form_submit_button("Run pipeline", help="Start the ICE-ID entity resolution pipeline with the specified parameters")

            # Compute derived shard path (where we prime)
            target_shard_dir = os.path.join(out_dir, f"shard_s{int(shard_id)}_of_{int(num_shards)}")

            if prime_btn:
                try:
                    if not os.path.isdir(src_shard_dir):
                        st.error("Source shard directory does not exist.")
                    else:
                        os.makedirs(target_shard_dir, exist_ok=True)
                        report = prime_out_dir(src_shard_dir, target_shard_dir)
                        n_dir = sum(1 for x in report["copied"] if x["kind"] == "dir")
                        n_file = sum(1 for x in report["copied"] if x["kind"] == "file")
                        st.success(f"Primed {target_shard_dir} with {n_dir} directories and {n_file} files.")
                        with st.expander("What was copied"):
                            st.write(report["copied"])
                        if report["skipped"]:
                            with st.expander("Skipped items (with reasons)"):
                                st.write(report["skipped"])
                except Exception as e:
                    st.error(f"Priming failed: {e}")

        if run_btn:
            try:
                os.makedirs(out_dir, exist_ok=True)

                # Map device selection to flag + environment
                device_flag = None
                cuda_env = None
                if device_choice == "CPU only":
                    device_flag = "cpu"
                    cuda_env = "-1"
                elif device_choice == "GPU:0":
                    device_flag = "cuda"
                    cuda_env = "0"

                # Build command; pass new args only if build_command supports them
                sig = inspect.signature(build_command)
                extra_kwargs = {}
                if "entry_module" in sig.parameters:
                    extra_kwargs["entry_module"] = entry_module
                if "device" in sig.parameters:
                    extra_kwargs["device"] = device_flag

                cmd = build_command(
                    people_csv=people_csv,
                    labels_csv=labels_csv,
                    out_dir=out_dir,
                    num_shards=int(num_shards),
                    shard_id=int(shard_id),
                    n_workers=int(n_workers),
                    max_block_size=int(max_block_size),
                    max_pairs_per_block=int(max_pairs_per_block),
                    backends=backends,
                    gbdt_neg_ratio=float(gbdt_neg_ratio),
                    thresh_grid=int(thresh_grid),
                    seed=int(seed),
                    sample_frac=float(sample_frac),
                    fn_prefix=int(fn_prefix),
                    pat_prefix=int(pat_prefix),
                    year_bucket_width=int(year_bucket_width),
                    dual_blocking=bool(dual_blocking),
                    soft_filter_max_year_diff=int(soft_filter_max_year_diff),
                    soft_filter_sex=bool(soft_filter_sex),
                    preview_window=float(preview_window),
                    preview_limit=int(preview_limit),
                    **extra_kwargs,
                )

                st.code(" ".join(cmd), language="bash")
                st.info("Pipeline started. Check the logs below for progress.")

                # Prepare env for CUDA visibility
                env = os.environ.copy()
                if cuda_env is not None:
                    env["CUDA_VISIBLE_DEVICES"] = cuda_env

                # Run the pipeline
                try:
                    for ev in run_pipeline_streaming(cmd, cwd=ICEID_ROOT, env=env):
                        if ev["type"] == "line":
                            st.write(ev["text"])
                        elif ev["type"] == "done":
                            st.success("Pipeline completed!")
                            break
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
            except Exception as e:
                st.error(f"Training setup error: {e}")

    elif model_type == "External Models":
        # Settings management for external models
        st.markdown("#### ⚙️ Model Settings")
        from dashboard.settings_manager import settings_manager

        col_settings_load, col_settings_save = st.columns(2)

        with col_settings_load:
            st.markdown("**Load Settings**")
            if "available_settings" not in st.session_state or st.button("🔄 Refresh Settings", key="refresh_settings", help="Reload the list of saved model configurations"):
                st.session_state["available_settings"] = settings_manager.list_configs()
            
            available_settings = st.session_state.get("available_settings", [])

            if available_settings:
                selected_settings = st.selectbox(
                    "Load saved settings:",
                    ["None"] + available_settings,
                    key="load_training_settings",
                    help="Select a previously saved configuration to load its parameters"
                )

                if selected_settings != "None":
                    try:
                        config = settings_manager.load_config(selected_settings)
                        st.success(f"✅ Loaded: {config.model_name}")

                        # Show key settings
                        summary = settings_manager.get_config_summary(config)
                        st.json({k: v for k, v in list(summary.items())[:5]})  # Show first 5 settings
                    except Exception as e:
                        st.error(f"❌ Error loading settings: {e}")
                else:
                    st.info("No saved settings available")

        with col_settings_save:
            st.markdown("**Quick Save**")
            if st.button("💾 Save Current Settings", key="quick_save_settings", help="Save current model parameters for future use"):
                st.info("Go to Data Editor tab to save detailed settings")

        st.markdown("---")

        # Paths and model selection (outside form so parameters update immediately)
        default_people = os.path.join(ICEID_ROOT, "raw_data", "people.csv")
        default_labels = os.path.join(ICEID_ROOT, "raw_data", "manntol_einstaklingar_new.csv")
        default_out = os.path.join(ICEID_ROOT, "runs", "external_models")

        people_csv = st.text_input("people.csv path", value=default_people, help="Path to the CSV file containing people records")
        labels_csv = st.text_input("labels.csv path", value=default_labels, help="Path to the CSV file containing ground truth labels for entity matching")
        out_dir = st.text_input("Output directory", value=default_out, help="Directory where model outputs and artifacts will be saved")

        # Model selection (outside form so UI updates immediately)
        selected_models = st.multiselect(
            "Select models to train",
            [
                "Ditto (HF)", "ZeroER (SBERT)", "TF-IDF", "MLP",
                "Splink", "XGBoost", "LightGBM", "Cross-Encoder",
                "Dedupe", "RecordLinkage", "Random Forest", "Gradient Boosting"
            ],
            default=[],
            help="Choose one or more entity resolution models to train. Each model will be trained independently with the parameters specified below."
        )

        sample_frac = st.number_input("Sample fraction", min_value=0.0, max_value=1.0, value=0.01, step=0.01, help="Fraction of data to use for training")
        random_state = st.number_input("Random state", min_value=0, value=42, step=1, help="Random seed for reproducibility")

        has_mlp = "MLP" in selected_models
        has_ditto = "Ditto (HF)" in selected_models
        has_crossencoder = "Cross-Encoder" in selected_models
        has_xgboost = "XGBoost" in selected_models
        has_lightgbm = "LightGBM" in selected_models
        has_gradient_boosting = "Gradient Boosting" in selected_models
        has_random_forest = "Random Forest" in selected_models
        has_tfidf = "TF-IDF" in selected_models
        has_recordlinkage = "RecordLinkage" in selected_models
        has_probabilistic = any(m in selected_models for m in ["Splink", "Dedupe"])

        mlp_epochs = 10
        mlp_learning_rate = 1e-3
        mlp_batch_size = 32
        mlp_hidden_dim = 128
        mlp_num_layers = 2
        mlp_dropout = 0.1
        mlp_weight_decay = 1e-4

        if has_mlp:
            st.markdown("### MLP Parameters")
            col_mlp1, col_mlp2 = st.columns(2)
            with col_mlp1:
                mlp_epochs = st.number_input("Training epochs", min_value=1, value=10, step=1, help="Number of training epochs", key="mlp_epochs")
                mlp_learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, step=1e-6, format="%.2e", help="Learning rate for optimization", key="mlp_lr")
                mlp_batch_size = st.number_input("Batch size", min_value=1, value=32, step=1, help="Batch size for training", key="mlp_batch")
            with col_mlp2:
                mlp_hidden_dim = st.number_input("Hidden dimension", min_value=32, value=128, step=32, help="Hidden layer dimension", key="mlp_hidden")
                mlp_num_layers = st.number_input("Number of layers", min_value=1, value=2, step=1, help="Number of hidden layers", key="mlp_layers")
            with st.expander("🔧 Advanced MLP Parameters", expanded=False):
                col_mlp_adv1, col_mlp_adv2 = st.columns(2)
                with col_mlp_adv1:
                    mlp_dropout = st.slider("Dropout rate", 0.0, 0.5, 0.1, 0.01, help="Dropout probability", key="mlp_dropout")
                    mlp_weight_decay = st.number_input("Weight decay", min_value=0.0, value=1e-4, step=1e-5, format="%.2e", help="L2 regularization", key="mlp_wd")

        ditto_epochs = 3
        ditto_learning_rate = 2e-5
        ditto_batch_size = 16
        ditto_max_length = 512
        ditto_dropout = 0.1
        ditto_weight_decay = 1e-4
        ditto_optimizer = "adamw"
        ditto_scheduler = "linear"
        ditto_early_stopping = True

        if has_ditto:
            st.markdown("### Ditto (HF) Parameters")
            col_ditto1, col_ditto2 = st.columns(2)
            with col_ditto1:
                ditto_epochs = st.number_input("Training epochs", min_value=1, value=3, step=1, help="Number of training epochs", key="ditto_epochs")
                ditto_learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=2e-5, step=1e-6, format="%.2e", help="Learning rate for optimization", key="ditto_lr")
                ditto_batch_size = st.number_input("Batch size", min_value=1, value=16, step=1, help="Batch size for training", key="ditto_batch")
            with col_ditto2:
                ditto_max_length = st.number_input("Max text length", min_value=64, value=512, step=64, help="Maximum sequence length", key="ditto_maxlen")
            with st.expander("🔧 Advanced Ditto Parameters", expanded=False):
                col_ditto_adv1, col_ditto_adv2 = st.columns(2)
                with col_ditto_adv1:
                    ditto_dropout = st.slider("Dropout rate", 0.0, 0.5, 0.1, 0.01, help="Dropout probability", key="ditto_dropout")
                    ditto_weight_decay = st.number_input("Weight decay", min_value=0.0, value=1e-4, step=1e-5, format="%.2e", help="L2 regularization", key="ditto_wd")
                with col_ditto_adv2:
                    ditto_optimizer = st.selectbox("Optimizer", ["adam", "adamw", "sgd", "rmsprop"], index=1, help="Optimization algorithm", key="ditto_opt")
                    ditto_scheduler = st.selectbox("Scheduler", ["none", "linear", "cosine", "step"], index=1, help="Learning rate scheduler", key="ditto_sched")
                    ditto_early_stopping = st.checkbox("Early stopping", value=True, help="Stop training if validation loss doesn't improve", key="ditto_es")

        crossencoder_epochs = 3
        crossencoder_learning_rate = 2e-5
        crossencoder_batch_size = 16
        crossencoder_max_length = 512
        crossencoder_dropout = 0.1
        crossencoder_weight_decay = 1e-4
        crossencoder_optimizer = "adamw"
        crossencoder_scheduler = "linear"
        crossencoder_early_stopping = True

        if has_crossencoder:
            st.markdown("### Cross-Encoder Parameters")
            col_ce1, col_ce2 = st.columns(2)
            with col_ce1:
                crossencoder_epochs = st.number_input("Training epochs", min_value=1, value=3, step=1, help="Number of training epochs", key="ce_epochs")
                crossencoder_learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=2e-5, step=1e-6, format="%.2e", help="Learning rate for optimization", key="ce_lr")
                crossencoder_batch_size = st.number_input("Batch size", min_value=1, value=16, step=1, help="Batch size for training", key="ce_batch")
            with col_ce2:
                crossencoder_max_length = st.number_input("Max text length", min_value=64, value=512, step=64, help="Maximum sequence length", key="ce_maxlen")
            with st.expander("🔧 Advanced Cross-Encoder Parameters", expanded=False):
                col_ce_adv1, col_ce_adv2 = st.columns(2)
                with col_ce_adv1:
                    crossencoder_dropout = st.slider("Dropout rate", 0.0, 0.5, 0.1, 0.01, help="Dropout probability", key="ce_dropout")
                    crossencoder_weight_decay = st.number_input("Weight decay", min_value=0.0, value=1e-4, step=1e-5, format="%.2e", help="L2 regularization", key="ce_wd")
                with col_ce_adv2:
                    crossencoder_optimizer = st.selectbox("Optimizer", ["adam", "adamw", "sgd", "rmsprop"], index=1, help="Optimization algorithm", key="ce_opt")
                    crossencoder_scheduler = st.selectbox("Scheduler", ["none", "linear", "cosine", "step"], index=1, help="Learning rate scheduler", key="ce_sched")
                    crossencoder_early_stopping = st.checkbox("Early stopping", value=True, help="Stop training if validation loss doesn't improve", key="ce_es")

        xgb_n_estimators = 100
        xgb_max_depth = 5
        xgb_learning_rate = 0.1
        xgb_min_samples_split = 2
        xgb_subsample = 1.0

        if has_xgboost:
            st.markdown("### XGBoost Parameters")
            col_xgb1, col_xgb2 = st.columns(2)
            with col_xgb1:
                xgb_n_estimators = st.number_input("Number of estimators", min_value=10, value=100, step=10, help="Number of boosting rounds", key="xgb_n_est")
                xgb_max_depth = st.number_input("Max depth", min_value=1, value=5, step=1, help="Maximum tree depth", key="xgb_max_d")
                xgb_learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f", help="Learning rate for boosting", key="xgb_lr")
            with col_xgb2:
                xgb_min_samples_split = st.number_input("Min samples split", min_value=2, value=2, step=1, help="Minimum samples to split a node", key="xgb_min_split")
                xgb_subsample = st.number_input("Subsample ratio", min_value=0.1, max_value=1.0, value=1.0, step=0.1, help="Fraction of samples for each tree", key="xgb_subsample")

        lgbm_n_estimators = 100
        lgbm_max_depth = 5
        lgbm_learning_rate = 0.1
        lgbm_min_samples_split = 2
        lgbm_subsample = 1.0

        if has_lightgbm:
            st.markdown("### LightGBM Parameters")
            col_lgbm1, col_lgbm2 = st.columns(2)
            with col_lgbm1:
                lgbm_n_estimators = st.number_input("Number of estimators", min_value=10, value=100, step=10, help="Number of boosting rounds", key="lgbm_n_est")
                lgbm_max_depth = st.number_input("Max depth", min_value=1, value=5, step=1, help="Maximum tree depth", key="lgbm_max_d")
                lgbm_learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f", help="Learning rate for boosting", key="lgbm_lr")
            with col_lgbm2:
                lgbm_min_samples_split = st.number_input("Min samples split", min_value=2, value=2, step=1, help="Minimum samples to split a node", key="lgbm_min_split")
                lgbm_subsample = st.number_input("Subsample ratio", min_value=0.1, max_value=1.0, value=1.0, step=0.1, help="Fraction of samples for each tree", key="lgbm_subsample")

        gb_n_estimators = 200
        gb_max_depth = 6
        gb_learning_rate = 0.05
        gb_min_samples_split = 10
        gb_min_samples_leaf = 4
        gb_subsample = 0.8

        if has_gradient_boosting:
            st.markdown("### Gradient Boosting (sklearn) Parameters")
            col_gb1, col_gb2 = st.columns(2)
            with col_gb1:
                gb_n_estimators = st.number_input("Number of estimators", min_value=10, value=200, step=10, help="Number of boosting stages", key="gb_n_est")
                gb_max_depth = st.number_input("Max depth", min_value=1, value=6, step=1, help="Maximum tree depth", key="gb_max_d")
                gb_learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f", help="Learning rate", key="gb_lr")
            with col_gb2:
                gb_min_samples_split = st.number_input("Min samples split", min_value=2, value=10, step=1, help="Minimum samples to split a node", key="gb_min_split")
                gb_min_samples_leaf = st.number_input("Min samples leaf", min_value=1, value=4, step=1, help="Minimum samples in a leaf", key="gb_min_leaf")
                gb_subsample = st.number_input("Subsample ratio", min_value=0.1, max_value=1.0, value=0.8, step=0.1, help="Fraction of samples for each tree", key="gb_subsample")

        rf_n_estimators = 200
        rf_max_depth = 10
        rf_min_samples_split = 5
        rf_min_samples_leaf = 2
        rf_max_features = "sqrt"

        if has_random_forest:
            st.markdown("### Random Forest Parameters")
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                rf_n_estimators = st.number_input("Number of estimators", min_value=10, value=200, step=10, help="Number of trees in the forest", key="rf_n_est")
                rf_max_depth = st.number_input("Max depth", min_value=1, value=10, step=1, help="Maximum depth of trees", key="rf_max_d")
                rf_min_samples_split = st.number_input("Min samples split", min_value=2, value=5, step=1, help="Minimum samples to split a node", key="rf_min_split")
            with col_rf2:
                rf_min_samples_leaf = st.number_input("Min samples leaf", min_value=1, value=2, step=1, help="Minimum samples in a leaf", key="rf_min_leaf")
                rf_max_features = st.selectbox("Max features", ["sqrt", "log2", None], index=0, help="Number of features to consider", key="rf_max_feat")

        tfidf_max_features = 20000
        lr_C = 1.0

        if has_tfidf:
            st.markdown("### TF-IDF Parameters")
            col_tfidf1, col_tfidf2 = st.columns(2)
            with col_tfidf1:
                tfidf_max_features = st.number_input("TF-IDF max features", min_value=1000, value=20000, step=1000, help="Maximum number of TF-IDF features", key="tfidf_features")
            with col_tfidf2:
                lr_C = st.number_input("Logistic Regression C", min_value=0.001, value=1.0, step=0.1, help="Inverse regularization strength", key="lr_C")

        rl_threshold = 0.85
        rl_n_estimators = 100

        if has_recordlinkage:
            st.markdown("### RecordLinkage Parameters")
            col_rl1, col_rl2 = st.columns(2)
            with col_rl1:
                rl_threshold = st.slider("Jaro-Winkler threshold", 0.0, 1.0, 0.85, 0.05, help="String similarity threshold", key="rl_thresh")
            with col_rl2:
                rl_n_estimators = st.number_input("Random Forest n_estimators", min_value=10, value=100, step=10, help="Number of trees", key="rl_n_est")

        splink_threshold = 0.5
        dedupe_sample_size = 15000
        dedupe_threshold = 0.5

        if has_probabilistic:
            st.markdown("### Probabilistic Models Parameters")
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                if "Splink" in selected_models:
                    splink_threshold = st.slider("Match probability threshold", 0.0, 1.0, 0.5, 0.05, help="Minimum probability for a match", key="splink_thresh")
                if "Dedupe" in selected_models:
                    dedupe_sample_size = st.number_input("Training sample size", min_value=1000, value=15000, step=1000, help="Number of samples for active learning", key="dedupe_sample")
            with col_prob2:
                if "Dedupe" in selected_models:
                    dedupe_threshold = st.slider("Match threshold", 0.0, 1.0, 0.5, 0.05, help="Minimum score for a match", key="dedupe_thresh")

        st.markdown("---")
        
        with st.form("external_train_form", clear_on_submit=False):
            run_external_btn = st.form_submit_button("🚀 Train External Models", help="Train all selected models with the specified parameters")

            if run_external_btn:
                if not selected_models:
                    st.error("⚠️ Please select at least one model to train!")
                elif not os.path.exists(people_csv):
                    st.error(f"⚠️ People CSV not found: {people_csv}")
                elif not os.path.exists(labels_csv):
                    st.error(f"⚠️ Labels CSV not found: {labels_csv}")
                elif sample_frac <= 0 or sample_frac > 1:
                    st.error("⚠️ Sample fraction must be between 0 and 1!")
                else:
                    try:
                        from dashboard.model_registry import TrainingContext, run_external_models

                        os.makedirs(out_dir, exist_ok=True)

                        ctx = TrainingContext(
                            people_csv=people_csv,
                            labels_csv=labels_csv,
                            run_shard_dir=out_dir,
                            sample_frac=float(sample_frac),
                            epochs=int(max(mlp_epochs if has_mlp else 0, ditto_epochs if has_ditto else 0, crossencoder_epochs if has_crossencoder else 0, 3))
                        )

                        if has_mlp:
                            ctx.mlp_epochs = mlp_epochs
                            ctx.mlp_learning_rate = mlp_learning_rate
                            ctx.mlp_batch_size = mlp_batch_size
                            ctx.mlp_hidden_dim = mlp_hidden_dim
                            ctx.mlp_num_layers = mlp_num_layers
                            ctx.mlp_dropout = mlp_dropout
                            ctx.mlp_weight_decay = mlp_weight_decay

                        if has_ditto:
                            ctx.ditto_epochs = ditto_epochs
                            ctx.ditto_learning_rate = ditto_learning_rate
                            ctx.ditto_batch_size = ditto_batch_size
                            ctx.ditto_max_length = ditto_max_length
                            ctx.ditto_dropout = ditto_dropout
                            ctx.ditto_weight_decay = ditto_weight_decay
                            ctx.ditto_optimizer = ditto_optimizer
                            ctx.ditto_scheduler = ditto_scheduler
                            ctx.ditto_early_stopping = ditto_early_stopping

                        if has_crossencoder:
                            ctx.crossencoder_epochs = crossencoder_epochs
                            ctx.crossencoder_learning_rate = crossencoder_learning_rate
                            ctx.crossencoder_batch_size = crossencoder_batch_size
                            ctx.crossencoder_max_length = crossencoder_max_length
                            ctx.crossencoder_dropout = crossencoder_dropout
                            ctx.crossencoder_weight_decay = crossencoder_weight_decay
                            ctx.crossencoder_optimizer = crossencoder_optimizer
                            ctx.crossencoder_scheduler = crossencoder_scheduler
                            ctx.crossencoder_early_stopping = crossencoder_early_stopping

                        if has_xgboost:
                            ctx.xgb_n_estimators = xgb_n_estimators
                            ctx.xgb_max_depth = xgb_max_depth
                            ctx.xgb_learning_rate = xgb_learning_rate
                            ctx.xgb_min_samples_split = xgb_min_samples_split
                            ctx.xgb_subsample = xgb_subsample

                        if has_lightgbm:
                            ctx.lgbm_n_estimators = lgbm_n_estimators
                            ctx.lgbm_max_depth = lgbm_max_depth
                            ctx.lgbm_learning_rate = lgbm_learning_rate
                            ctx.lgbm_min_samples_split = lgbm_min_samples_split
                            ctx.lgbm_subsample = lgbm_subsample

                        if has_gradient_boosting:
                            ctx.gb_n_estimators = gb_n_estimators
                            ctx.gb_max_depth = gb_max_depth
                            ctx.gb_learning_rate = gb_learning_rate
                            ctx.gb_min_samples_split = gb_min_samples_split
                            ctx.gb_min_samples_leaf = gb_min_samples_leaf
                            ctx.gb_subsample = gb_subsample

                        if has_random_forest:
                            ctx.rf_n_estimators = rf_n_estimators
                            ctx.rf_max_depth = rf_max_depth
                            ctx.rf_min_samples_split = rf_min_samples_split
                            ctx.rf_min_samples_leaf = rf_min_samples_leaf
                            ctx.rf_max_features = rf_max_features

                        if has_tfidf:
                            ctx.tfidf_max_features = tfidf_max_features
                            ctx.lr_C = lr_C

                        if has_recordlinkage:
                            ctx.rl_threshold = rl_threshold
                            ctx.rl_n_estimators = rl_n_estimators

                        if has_probabilistic:
                            if "Splink" in selected_models:
                                ctx.splink_threshold = splink_threshold
                            if "Dedupe" in selected_models:
                                ctx.dedupe_sample_size = dedupe_sample_size
                                ctx.dedupe_threshold = dedupe_threshold

                        ctx.random_state = random_state

                        with st.status(f"Training {len(selected_models)} model(s)...", expanded=True) as status:
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            
                            status_text.write(f"📊 Models to train: {', '.join(selected_models)}")
                            status_text.write(f"📉 Sample size: {sample_frac*100:.2f}% of data")
                            progress_bar.progress(0.05)
                            
                            results = {}
                            for idx, model in enumerate(selected_models):
                                progress = idx / len(selected_models)
                                status_text.write(f"🚀 Training **{model}** ({idx + 1}/{len(selected_models)})...")
                                progress_bar.progress(min(0.05 + (progress * 0.95), 0.95))
                                
                                model_results = run_external_models([model], ctx)
                                results.update(model_results)
                                
                                if model_results and model.lower() in model_results:
                                    status_text.write(f"✅ **{model}** training complete")
                            
                            progress_bar.progress(1.0)
                            status_text.write(f"✅ All {len(selected_models)} model(s) trained successfully!")
                            status.update(label="Training complete!", state="complete", expanded=False)

                        st.success(f"✅ Trained {len(selected_models)} model(s) successfully!")
                        
                        if results:
                            st.markdown("### 📊 Automatic Evaluation Results")
                            
                            with st.spinner("Evaluating trained models..."):
                                try:
                                    model_dirs = []
                                    for model_name, outputs in results.items():
                                        if isinstance(outputs, dict) and "edges" in outputs:
                                            edges_path = outputs["edges"]
                                            model_dir = os.path.dirname(edges_path)
                                            model_dirs.append(os.path.basename(model_dir))
                                    
                                    if model_dirs and os.path.exists(people_csv_path) and os.path.exists(labels_csv_path):
                                        eval_results = analyze_backends(
                                            run_dir=external_output_dir,
                                            backends=model_dirs,
                                            people_csv=people_csv_path,
                                            labels_csv=labels_csv_path
                                        )
                                        
                                        summary_data = []
                                        metric_keys = ["prec_labeled", "rec_labeled", "f1_labeled", "edges_total"]
                                        
                                        for backend in model_dirs:
                                            row = {"Model": backend}
                                            backend_metrics = eval_results.get(backend, {})
                                            for key in metric_keys:
                                                value = backend_metrics.get(key)
                                                if key in ["prec_labeled", "rec_labeled", "f1_labeled"] and value is not None:
                                                    row[key] = value
                                                elif isinstance(value, (int, float)):
                                                    row[key] = value
                                                else:
                                                    row[key] = None
                                            summary_data.append(row)
                                        
                                        if summary_data:
                                            summary_df = pd.DataFrame(summary_data)
                                            
                                            def highlight_best(s):
                                                if s.name in ["prec_labeled", "rec_labeled", "f1_labeled"]:
                                                    numeric_vals = pd.to_numeric(s, errors='coerce')
                                                    if numeric_vals.notna().any():
                                                        is_max = numeric_vals == numeric_vals.max()
                                                        return ['background-color: #90EE90' if v else '' for v in is_max]
                                                return ['' for _ in s]
                                            
                                            col_rename = {
                                                "prec_labeled": "Precision", 
                                                "rec_labeled": "Recall", 
                                                "f1_labeled": "F1 Score", 
                                                "edges_total": "Total Pairs"
                                            }
                                            summary_df = summary_df.rename(columns=col_rename)
                                            
                                            styled_df = summary_df.style.apply(highlight_best, subset=['Precision', 'Recall', 'F1 Score'])
                                            
                                            if 'F1 Score' in summary_df.columns:
                                                f1_vals = pd.to_numeric(summary_df['F1 Score'], errors='coerce')
                                                if f1_vals.notna().any():
                                                    best_idx = f1_vals.idxmax()
                                                    best_model = summary_df.iloc[best_idx]['Model']
                                                    st.info(f"🏆 **Best Overall Model (by F1):** {best_model} - F1: {f1_vals.iloc[best_idx]:.4f}")
                                            
                                            st.dataframe(styled_df, hide_index=True)
                                        else:
                                            st.warning("No evaluation metrics generated")
                                    else:
                                        st.info("💡 Evaluation skipped - missing people.csv or labels.csv")
                                        
                                except Exception as eval_error:
                                    st.warning(f"Automatic evaluation failed: {eval_error}")
                            
                            with st.expander("📁 Show Training Outputs", expanded=False):
                                for model_name, outputs in results.items():
                                    st.markdown(f"**{model_name}**")
                                    if isinstance(outputs, dict):
                                        for output_type, path in outputs.items():
                                            if isinstance(path, str):
                                                exists = "✓" if os.path.exists(path) else "✗"
                                                st.write(f"{exists} {output_type}: `{path}`")
                                            else:
                                                st.write(f"  - {output_type}: {path}")
                                    else:
                                        st.write(outputs)
                        else:
                            st.warning("⚠️ No outputs generated. Check logs for errors.")

                    except ImportError as e:
                        st.error(f"❌ Missing dependency: {e}")
                        st.info("Install missing packages: `pip install <package-name>`")
                    except Exception as e:
                        st.error(f"❌ External model training error: {e}")
                        if st.checkbox("Show detailed error", key="show_error_detail"):
                            import traceback
                            st.code(traceback.format_exc())

# --------------------------------------------------------------------------------------
# EVALUATION TAB
# --------------------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluate Backends")

    # Get available run directories
    available_runs = _get_available_run_directories()

    # Default paths
    default_people = os.path.join(ICEID_ROOT, "raw_data", "people.csv")
    default_labels = os.path.join(ICEID_ROOT, "raw_data", "manntol_einstaklingar_new.csv")

    # Create display names for run directories (show relative path and modification time)
    run_display_names = []
    run_paths = []
    for run_path in available_runs:
        rel_path = os.path.relpath(run_path, ICEID_ROOT)
        mod_time = os.path.getmtime(run_path)
        import datetime
        mod_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        display_name = f"{rel_path} (modified: {mod_str})"
        run_display_names.append(display_name)
        run_paths.append(run_path)

    # Add option for custom path
    run_display_names.append("Custom path...")
    run_paths.append("")

    col1, col2 = st.columns(2)

    with col1:
        people_csv_eval = st.text_input("people.csv path", value=default_people, key="people_eval")
        labels_csv_eval = st.text_input("labels.csv path (bi_einstaklingur)", value=default_labels, key="labels_eval")

    with col2:
        # Run directory selection
        run_selection = st.selectbox(
            "Select run directory",
            options=run_display_names,
            index=0 if run_display_names else len(run_display_names) - 1,
            key="run_selection_eval",
            help="Select from available run directories or choose custom path"
        )

        # Get the selected run directory
        if run_selection == "Custom path...":
            run_dir = st.text_input(
                "Custom run directory path",
                value="",
                key="custom_run_dir_eval",
                help="Enter the full path to a run directory containing backend subdirectories"
            )
        else:
            run_dir = run_paths[run_display_names.index(run_selection)]

        # Backend selection (only show if run_dir is valid)
        if run_dir and os.path.exists(run_dir):
            available_backends = _get_available_backends(run_dir)
            if available_backends:
                backends_eval = st.multiselect(
                    "Select backends to evaluate",
                    options=available_backends,
                    default=available_backends,
                    key="backends_eval",
                    help="Select the specific backend folders (e.g., 'gbdt', 'logreg') to evaluate."
                )
            else:
                st.warning(f"No valid backend directories found in: **{run_dir}**")
                backends_eval = []
        else:
            if run_dir:
                st.error(f"Selected run directory not found: **{run_dir}**")
            backends_eval = []

    st.markdown("---")
    
    # Evaluation Button
    eval_btn = st.button("Run Evaluation", disabled=not (run_dir and backends_eval), help="Evaluate the selected backends and calculate performance metrics")

    if eval_btn:
        try:
            with st.spinner(f"Evaluating {len(backends_eval)} backend(s)... This may take a while for large datasets."):
                st.info(f"📊 Analyzing: {', '.join(backends_eval)}")
                
                run_report = analyze_backends(
                    run_dir=run_dir,
                    backends=backends_eval,
                    people_csv=people_csv_eval,
                    labels_csv=labels_csv_eval
                )
            
            st.success(f"✅ Evaluated {len(backends_eval)} backend(s) successfully!")

            # ----------------------------------------------------------------------------------
            # Summary Table
            # ----------------------------------------------------------------------------------
            st.markdown("### Summary Metrics")
            
            # Prepare data for the summary table
            summary_data = []
            metric_keys = ["prec_labeled", "rec_labeled", "f1_labeled", "edges_total", "nodes_with_any_match"]
            
            for backend in backends_eval:
                row = {"Backend": backend}
                backend_metrics = run_report.get(backend, {})
                for key in metric_keys:
                    value = backend_metrics.get(key)
                    if key in ["prec_labeled", "rec_labeled", "f1_labeled"]:
                        row[key] = _fmt_pct(value)
                    elif isinstance(value, (int, float)):
                        row[key] = f"{value:,}"
                    else:
                        row[key] = "—"
                summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                def highlight_best(s):
                    if s.name in ["prec_labeled", "rec_labeled", "f1_labeled"]:
                        raw_values = []
                        for val in s:
                            if isinstance(val, str):
                                try:
                                    cleaned = val.replace('%', '').replace(',', '').strip()
                                    raw_values.append(float(cleaned) if cleaned and cleaned != '—' else None)
                                except:
                                    raw_values.append(None)
                            else:
                                raw_values.append(None)
                        
                        if any(v is not None for v in raw_values):
                            max_val = max(v for v in raw_values if v is not None)
                            return ['background-color: #90EE90' if v == max_val else '' for v in raw_values]
                    return ['' for _ in s]
                
                col_rename = {
                    "prec_labeled": "Precision (Labeled)", 
                    "rec_labeled": "Recall (Labeled)", 
                    "f1_labeled": "F1 Score (Labeled)", 
                    "edges_total": "Total Pairs",
                    "nodes_with_any_match": "Nodes w/ Match"
                }
                summary_df = summary_df.rename(columns=col_rename)
                
                styled_df = summary_df.style.apply(highlight_best, subset=['Precision (Labeled)', 'Recall (Labeled)', 'F1 Score (Labeled)'])
                
                if 'F1 Score (Labeled)' in summary_df.columns:
                    f1_values = []
                    for val in summary_df['F1 Score (Labeled)']:
                        if isinstance(val, str):
                            try:
                                cleaned = val.replace('%', '').replace(',', '').strip()
                                f1_values.append(float(cleaned) if cleaned and cleaned != '—' else None)
                            except:
                                f1_values.append(None)
                        else:
                            f1_values.append(None)
                    
                    if any(v is not None for v in f1_values):
                        best_idx = max(range(len(f1_values)), key=lambda i: f1_values[i] if f1_values[i] is not None else -1)
                        best_backend = summary_df.iloc[best_idx]['Backend']
                        best_f1 = f1_values[best_idx]
                        st.info(f"🏆 **Best Overall Backend (by F1):** {best_backend} - F1: {best_f1:.2f}%")
                
                st.dataframe(styled_df, hide_index=True)

            # ----------------------------------------------------------------------------------
            # Detailed Metrics & Conclusions
            # ----------------------------------------------------------------------------------
            st.markdown("### Detailed Analysis and Conclusions")
            metric_explanations = _metric_explanations()
            
            for backend in backends_eval:
                st.markdown(f"#### 🔎 {backend.capitalize()} Backend")
                backend_report = run_report.get(backend, {})
                
                if not backend_report:
                    st.warning("No report generated for this backend. Check logs for errors.")
                    continue
                
                # Automated Conclusions
                conclusions = _auto_conclusions(backend_report)
                
                # Display key conclusions
                st.markdown("**Key Findings:**")
                for key, msg in conclusions.items():
                    st.write(f"- **{metric_explanations.get(key, key.replace('_', ' ').title())}**: {msg}")
                
                # Display full metrics
                with st.expander("Show All Raw Metrics", expanded=False):
                    metrics_to_show = {}
                    for k, v in backend_report.items():
                        if isinstance(v, (float, int)):
                            metrics_to_show[k] = f"{v:.4f}" if isinstance(v, float) else f"{v:,}"
                        elif isinstance(v, tuple) and len(v) == 2: # CI
                            metrics_to_show[k] = f"({_fmt_pct(v[0])}, {_fmt_pct(v[1])})"
                        elif isinstance(v, dict): # Cluster stats
                            metrics_to_show[k] = {vk: (f"{vv:.4f}" if isinstance(vv, float) else f"{vv:,}") for vk, vv in v.items()}
                        else:
                            metrics_to_show[k] = str(v)
                            
                    st.json(metrics_to_show)
                
                st.markdown("---")

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.exception(e)

# --------------------------------------------------------------------------------------
# INSPECT & EDIT TAB
# --------------------------------------------------------------------------------------
with tab_inspect:
    st.subheader("Inspect Data, Pairs, and Clusters")
    
    # Use the local function to display the tab content
    show_inspector_tab(ICEID_ROOT)


# --------------------------------------------------------------------------------------
# DATA EDITOR TAB (Placeholder/Example)
# --------------------------------------------------------------------------------------
with tab_data_editor:
    st.subheader("Data Management and Settings Editor")
    
    # Data Overview Section
    st.markdown("### 📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("People Records", "Loading...", help="Total number of people in the dataset")
    with col2:
        st.metric("Available Backends", "Loading...", help="Number of model backends available")
    with col3:
        st.metric("Saved Versions", "Loading...", help="Number of manually edited versions")
    
    # Try to load actual data for metrics
    try:
        if "people_count" not in st.session_state:
            from dashboard.ds_io import load_people
            people_df = load_people(os.path.join(ICEID_ROOT, "raw_data", "people.csv"), "dummy_sig")
            st.session_state["people_count"] = len(people_df)
        
        if "backend_count" not in st.session_state:
            from dashboard.ds_io import detect_backends
            runs_dir = os.path.join(ICEID_ROOT, "runs")
            backend_count = 0
            if os.path.exists(runs_dir):
                for run_dir in os.listdir(runs_dir):
                    run_path = os.path.join(runs_dir, run_dir)
                    if os.path.isdir(run_path):
                        backend_count += len(detect_backends(run_path))
            st.session_state["backend_count"] = backend_count
        
        if "version_count" not in st.session_state:
            versions_dir = os.path.join(ICEID_ROOT, "versions")
            version_count = 0
            if os.path.exists(versions_dir):
                version_count = len([d for d in os.listdir(versions_dir) 
                                   if os.path.isdir(os.path.join(versions_dir, d))])
            st.session_state["version_count"] = version_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("People Records", f"{st.session_state.get('people_count', 0):,}")
        with col2:
            st.metric("Available Backends", st.session_state.get("backend_count", 0))
        with col3:
            st.metric("Saved Versions", st.session_state.get("version_count", 0))
            
    except Exception as e:
        st.warning(f"Could not load data metrics: {e}")
    
    st.markdown("---")
    
    # CSV Data Editor Section
    st.markdown("### 📝 CSV Data Editor")
    
    csv_files = [
        ("people.csv", os.path.join(ICEID_ROOT, "raw_data", "people.csv")),
        ("labels.csv", os.path.join(ICEID_ROOT, "raw_data", "manntol_einstaklingar_new.csv")),
        ("parishes.csv", os.path.join(ICEID_ROOT, "raw_data", "parishes.csv")),
        ("counties.csv", os.path.join(ICEID_ROOT, "raw_data", "counties.csv")),
        ("districts.csv", os.path.join(ICEID_ROOT, "raw_data", "districts.csv")),
    ]
    
    available_csvs = [(name, path) for name, path in csv_files if os.path.exists(path)]
    
    if available_csvs:
        selected_index = st.selectbox(
            "Select CSV file to view/edit",
            options=range(len(available_csvs)),
            format_func=lambda x: available_csvs[x][0],
            help="Choose a CSV file to inspect and potentially edit"
        )
        
        selected_csv_name, selected_csv_path = available_csvs[selected_index]
        
        st.markdown(f"#### Viewing: {selected_csv_name}")
        
        try:
            @st.cache_data(ttl=300)
            def load_csv_data(path: str):
                return pd.read_csv(path)
            
            df = load_csv_data(selected_csv_path)
            
            # Initialize working copy in session state if not exists
            working_copy_key = f"working_df_{selected_csv_name}"
            if working_copy_key not in st.session_state:
                st.session_state[working_copy_key] = df.copy()
            
            working_df = st.session_state[working_copy_key]
            
            st.markdown(f"**Shape:** {working_df.shape[0]:,} rows × {working_df.shape[1]} columns")
            
            # Column editing controls
            st.markdown("### 🔧 Column Operations")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Rename Column**")
                old_col = st.selectbox("Select column to rename", working_df.columns, key=f"rename_col_{selected_csv_name}", help="Choose the column you want to rename")
                new_col = st.text_input("New column name", value=old_col, key=f"new_name_{selected_csv_name}", help="Enter the new name for the column")
                if st.button("Rename", key=f"rename_btn_{selected_csv_name}", help="Rename the selected column") and new_col != old_col:
                    if new_col not in working_df.columns:
                        working_df = working_df.rename(columns={old_col: new_col})
                        st.session_state[working_copy_key] = working_df
                        st.success(f"Renamed '{old_col}' to '{new_col}'")
                        st.rerun()
                    else:
                        st.error(f"Column '{new_col}' already exists!")
            
            with col2:
                st.markdown("**Drop Column**")
                cols_to_drop = st.multiselect("Select columns to drop", working_df.columns, key=f"drop_cols_{selected_csv_name}", help="Select one or more columns to remove from the dataset")
                if st.button("Drop Selected", key=f"drop_btn_{selected_csv_name}", help="Remove the selected columns from the DataFrame") and cols_to_drop:
                    working_df = working_df.drop(columns=cols_to_drop)
                    st.session_state[working_copy_key] = working_df
                    st.success(f"Dropped columns: {', '.join(cols_to_drop)}")
                    st.rerun()
            
            with col3:
                st.markdown("**Reset Changes**")
                if st.button("Reset to Original", key=f"reset_btn_{selected_csv_name}", help="Discard all changes and reload the original CSV data"):
                    st.session_state[working_copy_key] = df.copy()
                    st.success("Reset to original data")
                    st.rerun()
            
            # Show basic info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current Columns:**")
                st.write(list(working_df.columns))
            with col2:
                st.markdown("**Data Types:**")
                st.write(working_df.dtypes.to_dict())
            
            # Show sample data
            st.markdown("**Sample Data (first 100 rows):**")
            st.dataframe(working_df.head(100), height=400)
            
            # Show statistics for numeric columns
            numeric_cols = working_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Column Statistics:**")
                st.dataframe(working_df[numeric_cols].describe())
            
            # Missing data visualization
            if working_df.isnull().any().any():
                st.markdown("### 📊 Missing Data Analysis")
                
                # Missing data summary
                missing_summary = working_df.isnull().sum()
                missing_percent = (missing_summary / len(working_df)) * 100
                missing_df = pd.DataFrame({
                    'Column': missing_summary.index,
                    'Missing Count': missing_summary.values,
                    'Missing %': missing_percent.values.round(2)
                }).sort_values('Missing Count', ascending=False)
                
                st.markdown("**Missing Data Summary:**")
                st.dataframe(missing_df[missing_df['Missing Count'] > 0])
                
                # Try to show missingno plots
                try:
                    import missingno as msno
                    import matplotlib.pyplot as plt
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Missing Data Matrix:**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        msno.matrix(working_df, ax=ax)
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**Missing Data Bar Chart:**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        msno.bar(working_df, ax=ax)
                        st.pyplot(fig)
                        plt.close()
                    
                    # Heatmap if not too many columns
                    if working_df.shape[1] <= 20:
                        st.markdown("**Missing Data Correlation Heatmap:**")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        msno.heatmap(working_df, ax=ax)
                        st.pyplot(fig)
                        plt.close()
                        
                except ImportError:
                    st.warning("Install 'missingno' to see missing data visualizations: `pip install missingno`")
                except Exception as e:
                    st.warning(f"Could not generate missing data plots: {e}")
            else:
                st.success("✅ No missing data found in the dataset!")
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Download working copy
                working_csv_data = working_df.to_csv(index=False)
                st.download_button(
                    f"📥 Download Working Copy",
                    data=working_csv_data,
                    file_name=f"working_{selected_csv_name}",
                    mime="text/csv"
                )
            with col2:
                # Download original
                original_csv_data = df.to_csv(index=False)
                st.download_button(
                    f"📥 Download Original",
                    data=original_csv_data,
                    file_name=selected_csv_name,
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error loading {selected_csv_name}: {e}")
    else:
        st.warning("No CSV files found in the raw_data directory")
    
    st.markdown("---")
    
    # Expected Data Format Documentation
    st.markdown("### 📋 Expected Data Format for Models")
    
    with st.expander("View Model Data Requirements", expanded=False):
        st.markdown("""
        **Core Required Columns for ICE-ID Pipeline:**
        
        **people.csv:**
        - `id` (int64): Unique record identifier
        - `first_name` (str): Person's first name (normalized to lowercase)
        - `patronym` (str): Patronymic name (normalized to lowercase) 
        - `surname` (str): Family surname (normalized to lowercase)
        - `birthyear` (int64): Year of birth
        - `sex` (str): Gender (normalized to lowercase: 'karl', 'kona')
        - `person` (float64): Hand-labeled person ID for ground truth
        - `father` (float64): Father's record ID
        - `mother` (float64): Mother's record ID
        - `partner` (float64): Partner's record ID
        - `parish` (int64): Parish identifier
        - `district` (int64): District identifier
        - `farm` (int64): Farm identifier
        - `heimild` (int64): Source/census identifier
        
        **labels.csv (manntol_einstaklingar_new.csv):**
        - `bi_einstaklingur` (int64): Ground truth person ID
        - Other columns used for evaluation
        
        **Model-Specific Requirements:**
        
        **PyTorch Models (Ditto, ZeroER):**
        - Text fields: `first_name`, `patronym`, `surname` (concatenated)
        - Numeric features: `birthyear`, `sex` (encoded), `parish`, `district`
        - Target: Binary classification (match/no-match)
        
        **Scikit-learn Models (RF, GB, LR):**
        - All numeric columns from people.csv
        - Text features: TF-IDF vectors from name fields
        - Categorical encoding for string fields
        
        **ICE-ID Pipeline:**
        - Uses all core columns for blocking and feature engineering
        - Generates candidate pairs, then applies ML models
        - Outputs: `edges.csv` (id1, id2) and `clusters.csv` (cluster_id, size, members)
        """)
    
    st.markdown("---")
    
    # Settings Management Section
    try:
        from dashboard.settings_manager import settings_manager, ModelConfig, PyTorchModelSettings, SklearnModelSettings, DataProcessingSettings
        
        st.markdown("### 🛠️ Model Settings Manager")
        
        available_settings_list = settings_manager.list_configs()
        
        if available_settings_list:
            st.info(f"Found {len(available_settings_list)} saved configurations")
        else:
            st.info("No saved configurations found. Create your first one below!")
        
        setting_action = st.radio(
            "Action",
            ["View/Edit Existing", "Create New", "Delete Configuration"],
            key="setting_action",
            help="Choose what you want to do with model configurations"
        )
        
        if setting_action == "View/Edit Existing" and available_settings_list:
            selected_config_name = st.selectbox(
                "Select Configuration to Edit",
                available_settings_list,
                key="edit_config_name",
                help="Choose a saved configuration to view or modify"
            )
            if selected_config_name:
                config_to_edit = settings_manager.load_config(selected_config_name)
                
                st.markdown(f"#### Editing: {selected_config_name}")
                
                # Show current settings summary
                summary = settings_manager.get_config_summary(config_to_edit)
                st.json(summary)
                
                st.markdown("#### Modify Settings")
                
                # Basic settings
                config_to_edit.model_name = st.text_input("Model Name", value=config_to_edit.model_name)
                config_to_edit.model_type = st.selectbox("Model Type", ["pytorch", "sklearn", "iceid"], 
                                                       index=["pytorch", "sklearn", "iceid"].index(config_to_edit.model_type))
                
                # Model-specific settings
                if config_to_edit.model_type == "pytorch" and config_to_edit.pytorch_settings:
                    st.markdown("**PyTorch Settings**")
                    col1, col2 = st.columns(2)
                    with col1:
                        config_to_edit.pytorch_settings.hidden_dim = st.number_input("Hidden Dimension", 
                                                                                    value=config_to_edit.pytorch_settings.hidden_dim, 
                                                                                    min_value=32, step=32)
                        config_to_edit.pytorch_settings.learning_rate = st.number_input("Learning Rate", 
                                                                                       value=config_to_edit.pytorch_settings.learning_rate, 
                                                                                       min_value=1e-6, format="%.2e")
                    with col2:
                        config_to_edit.pytorch_settings.epochs = st.number_input("Epochs", 
                                                                               value=config_to_edit.pytorch_settings.epochs, 
                                                                               min_value=1)
                        config_to_edit.pytorch_settings.batch_size = st.number_input("Batch Size", 
                                                                                    value=config_to_edit.pytorch_settings.batch_size, 
                                                                                    min_value=1)
                
                elif config_to_edit.model_type == "sklearn" and config_to_edit.sklearn_settings:
                    st.markdown("**Scikit-learn Settings**")
                    col1, col2 = st.columns(2)
                    with col1:
                        config_to_edit.sklearn_settings.rf_n_estimators = st.number_input("Random Forest Estimators", 
                                                                                         value=config_to_edit.sklearn_settings.rf_n_estimators, 
                                                                                         min_value=10)
                        config_to_edit.sklearn_settings.lr_C = st.number_input("Logistic Regression C", 
                                                                              value=config_to_edit.sklearn_settings.lr_C, 
                                                                              min_value=0.01)
                    with col2:
                        config_to_edit.sklearn_settings.gb_learning_rate = st.number_input("Gradient Boosting Learning Rate", 
                                                                                          value=config_to_edit.sklearn_settings.gb_learning_rate, 
                                                                                          min_value=0.01)
                        config_to_edit.sklearn_settings.cv_folds = st.number_input("Cross-validation Folds", 
                                                                                  value=config_to_edit.sklearn_settings.cv_folds, 
                                                                                  min_value=2)
                
                # Data processing settings
                if config_to_edit.data_settings:
                    st.markdown("**Data Processing Settings**")
                    col1, col2 = st.columns(2)
                    with col1:
                        config_to_edit.data_settings.text_max_length = st.number_input("Text Max Length", 
                                                                                      value=config_to_edit.data_settings.text_max_length, 
                                                                                      min_value=64, step=64)
                        config_to_edit.data_settings.use_tfidf = st.checkbox("Use TF-IDF", 
                                                                            value=config_to_edit.data_settings.use_tfidf)
                    with col2:
                        config_to_edit.data_settings.normalize_numerical = st.checkbox("Normalize Numerical", 
                                                                                      value=config_to_edit.data_settings.normalize_numerical)
                        config_to_edit.data_settings.categorical_encoding = st.selectbox("Categorical Encoding", 
                                                                                        ["onehot", "label", "target"], 
                                                                                        index=["onehot", "label", "target"].index(config_to_edit.data_settings.categorical_encoding))
                
                if st.button(f"💾 Save Changes to {selected_config_name}", help="Update and save the modified configuration"):
                    settings_manager.save_config(config_to_edit, selected_config_name)
                    st.success(f"✅ Settings **{selected_config_name}** updated successfully!")
                    st.rerun()
                
        elif setting_action == "Create New":
            st.markdown("#### Create New Configuration")
            
            new_name = st.text_input("Configuration Name", value="my_config", help="Choose a descriptive name for your configuration")
            model_type = st.selectbox("Model Type", ["pytorch", "sklearn", "iceid"], help="Select the type of model this configuration is for")
            
            if model_type == "pytorch":
                st.markdown("**PyTorch Settings**")
                col1, col2 = st.columns(2)
                with col1:
                    hidden_dim = st.number_input("Hidden Dimension", value=128, min_value=32, step=32)
                    learning_rate = st.number_input("Learning Rate", value=1e-3, min_value=1e-6, format="%.2e")
                with col2:
                    epochs = st.number_input("Epochs", value=5, min_value=1)
                    batch_size = st.number_input("Batch Size", value=32, min_value=1)
                
                new_config = ModelConfig(
                    model_name=new_name,
                    model_type="pytorch",
                    pytorch_settings=PyTorchModelSettings(
                        hidden_dim=hidden_dim,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        batch_size=batch_size
                    ),
                    data_settings=DataProcessingSettings()
                )
                
            elif model_type == "sklearn":
                st.markdown("**Scikit-learn Settings**")
                col1, col2 = st.columns(2)
                with col1:
                    rf_estimators = st.number_input("Random Forest Estimators", value=100, min_value=10)
                    lr_c = st.number_input("Logistic Regression C", value=1.0, min_value=0.01)
                with col2:
                    gb_learning_rate = st.number_input("Gradient Boosting Learning Rate", value=0.1, min_value=0.01)
                    cv_folds = st.number_input("Cross-validation Folds", value=5, min_value=2)
                
                new_config = ModelConfig(
                    model_name=new_name,
                    model_type="sklearn",
                    sklearn_settings=SklearnModelSettings(
                        rf_n_estimators=rf_estimators,
                        lr_C=lr_c,
                        gb_learning_rate=gb_learning_rate,
                        cv_folds=cv_folds
                    ),
                    data_settings=DataProcessingSettings()
                )
                
            else:  # iceid
                st.markdown("**ICE-ID Pipeline Settings**")
                col1, col2 = st.columns(2)
                with col1:
                    backends = st.multiselect("Backends", ["gbdt", "logreg", "rf", "svm"], default=["gbdt", "logreg"])
                    neg_ratio = st.number_input("Negative Ratio", value=2.0, min_value=0.1)
                with col2:
                    thresh_grid = st.number_input("Threshold Grid Size", value=101, min_value=11, step=2)
                    dual_blocking = st.checkbox("Dual Blocking", value=False)
                
                new_config = ModelConfig(
                    model_name=new_name,
                    model_type="iceid",
                    iceid_backends=backends,
                    iceid_neg_ratio=neg_ratio,
                    iceid_thresh_grid=thresh_grid,
                    iceid_dual_blocking=dual_blocking,
                    data_settings=DataProcessingSettings()
                )
            
            if st.button("🚀 Create Configuration", help="Save this new model configuration with the specified name"):
                settings_manager.save_config(new_config, new_name)
                st.success(f"✅ Configuration **{new_name}** created successfully!")
                st.rerun()
                
        elif setting_action == "Delete Configuration" and available_settings_list:
            st.markdown("#### Delete Configuration")
            config_to_delete = st.selectbox("Select Configuration to Delete", available_settings_list, key="delete_config_name", help="Choose a configuration to permanently remove")
            
            if config_to_delete:
                # Show what will be deleted
                try:
                    config_preview = settings_manager.load_config(config_to_delete)
                    summary = settings_manager.get_config_summary(config_preview)
                    st.json(summary)
                except Exception as e:
                    st.warning(f"Could not load config preview: {e}")
                
                if st.button("🗑️ Delete Configuration", type="secondary", help="Permanently delete the selected configuration"):
                    if settings_manager.delete_config(config_to_delete):
                        st.success(f"✅ Configuration **{config_to_delete}** deleted successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete configuration")

    except ImportError as e:
        st.error(f"Settings manager could not be imported: {e}")
        st.warning("External model settings management is disabled.")