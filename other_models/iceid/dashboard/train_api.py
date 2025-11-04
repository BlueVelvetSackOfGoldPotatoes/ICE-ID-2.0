from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator, Any, Tuple
try:
    from .model_registry import TrainingContext, run_external_models  # type: ignore
except Exception:
    TrainingContext = None  # type: ignore
    run_external_models = None  # type: ignore


@dataclass
class RunResult:
    """A container for the final results of a pipeline execution.

    Attributes:
        ok (bool): True if the process exited with code 0, False otherwise.
        cmd (List[str]): The command that was executed.
        summary (dict | None): The final JSON summary parsed from stdout, if any.
        stdout (str): The entire captured standard output.
        stderr (str): The entire captured standard error.
    """
    ok: bool
    cmd: List[str]
    summary: dict | None
    stdout: str
    stderr: str


def build_command(
    people_csv: str,
    labels_csv: str,
    out_dir: str,
    num_shards: int = 1,
    shard_id: int = 0,
    n_workers: int = 4,
    max_block_size: int = 5000,
    max_pairs_per_block: int = 250000,
    backends: str = "gbdt,logreg",
    backends_list: Optional[List[str]] = None,
    gbdt_neg_ratio: float = 2.0,
    thresh_grid: int = 101,
    seed: int = 42,
    sample_frac: float = 1.0,
    fn_prefix: int = 2,
    pat_prefix: int = 3,
    year_bucket_width: int = 5,
    dual_blocking: bool = False,
    soft_filter_max_year_diff: int = 15,
    soft_filter_sex: bool = True,
    preview_window: float = 0.05,
    preview_limit: int = 500,
    device: Optional[str] = "auto",
    extra_args: Optional[List[str]] = None,
    external_models: Optional[List[str]] = None,
    external_epochs: int = 1,
) -> List[str]:
    """Construct the CLI
    
    Args:
        people_csv (str): Path to the main people data file.
        labels_csv (str): Path to the ground truth labels file.
        out_dir (str): The root output directory for the run.
        num_shards (int): Total number of shards to split the run into.
        shard_id (int): The 0-based index of the shard to run.
        n_workers (int): Number of parallel worker processes.
        max_block_size (int): Maximum records allowed in a single block.
        max_pairs_per_block (int): Maximum candidate pairs to generate per block.
        backends (str): Comma-separated list of matching backends to run.
        gbdt_neg_ratio (float): Negative-to-positive sample ratio for GBDT training.
        thresh_grid (int): Number of thresholds to evaluate for precision/recall.
        seed (int): Random seed for reproducibility.
        sample_frac (float): Fraction of records to sample for a quick run.
        fn_prefix (int): Minimum prefix length for first-name-like tokens in blocking.
        pat_prefix (int): Minimum prefix length for patronym/last-name-like tokens.
        year_bucket_width (int): Size of year buckets for blocking.
        dual_blocking (bool): Whether to use a secondary blocking strategy.
        soft_filter_max_year_diff (int): Maximum allowed birth year difference.
        soft_filter_sex (bool): Whether to enforce same-sex comparisons.
        preview_window (float): Fraction of edges near the decision boundary to preview.
        preview_limit (int): Maximum number of preview rows to generate.

    Returns:
        List[str]: A list of strings representing the command-line arguments.
    """
    use_backends = backends if backends else (
        ",".join(backends_list) if backends_list else "gbdt,logreg"
    )
    cmd = [
        sys.executable, "-u", "-m", "iceid.main",  # -u for unbuffered output
        "--people_csv", people_csv,
        "--labels_csv", labels_csv,
        "--out_dir", out_dir,
        "--num_shards", str(num_shards),
        "--shard_id", str(shard_id),
        "--n_workers", str(n_workers),
        "--max_block_size", str(max_block_size),
        "--max_pairs_per_block", str(max_pairs_per_block),
        "--backends", use_backends,
        "--gbdt_neg_ratio", str(gbdt_neg_ratio),
        "--thresh_grid", str(thresh_grid),
        "--seed", str(seed),
        "--sample_frac", str(sample_frac),
        "--fn_prefix", str(fn_prefix),
        "--pat_prefix", str(pat_prefix),
        "--year_bucket_width", str(year_bucket_width),
        "--preview_window", str(preview_window),
        "--preview_limit", str(preview_limit),
    ]
    if dual_blocking:
        cmd.append("--dual_blocking")
    if not soft_filter_sex:
        cmd.append("--no_soft_filter_sex")
    cmd += ["--soft_filter_max_year_diff", str(soft_filter_max_year_diff)]
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


# -----------------------
# Streaming (for live UI)
# -----------------------

_KV_RE = re.compile(r"([a-zA-Z0-9_]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(%)?")

def _parse_keyvals(line: str) -> Dict[str, float]:
    """Parses loose 'key=value' pairs from a log line into a dictionary of floats.

    It handles various formats like 'train_loss=0.432 val_loss=0.501 acc=88.2%'.

    Args:
        line (str): The log line to parse.

    Returns:
        Dict[str, float]: A dictionary of parsed metric keys and their float values.
    """
    out: Dict[str, float] = {}
    for m in _KV_RE.finditer(line):
        k = m.group(1).lower()
        v = float(m.group(2))
        if m.group(3) == "%":
            v = v / 100.0
        out[k] = v
    return out


def run_pipeline_streaming(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Executes a command and yields structured events as the process runs.

    This function streams the stdout of a subprocess, parsing it in real-time to
    generate events for live UI updates. It captures raw log lines, progress
    percentages, epoch metrics, and a final summary JSON.

    Args:
        cmd (List[str]): The command to execute as a list of strings.
        cwd (Optional[str]): The working directory for the command. Defaults to None.
        env (Optional[Dict[str, str]]): Environment variables for the subprocess. Defaults to None.

    Yields:
        Iterator[Dict[str, Any]]: An iterator of event dictionaries, each with a 'type'
            key (e.g., 'line', 'progress', 'epoch', 'summary', 'done').
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge for simpler streaming
        text=True,
        bufsize=1,
    )

    epoch_re = re.compile(r"epoch\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    one_epoch_re = re.compile(r"(?:^|\s)epoch\s*[:#-]?\s*(\d+)(?!/)", re.IGNORECASE)
    tqdm_pct_re = re.compile(r"(\d{1,3})%")  # catches '95%' anywhere in the line

    collected_out: List[str] = []
    last_json_candidate: List[str] = []

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        collected_out.append(line)
        last_json_candidate.append(line)
        if len(last_json_candidate) > 4000:
            last_json_candidate = last_json_candidate[-4000:]

        # Always yield the raw line
        yield {"type": "line", "text": line}

        # Parse progress percentage (for tqdm-style logs)
        m_pct = tqdm_pct_re.search(line)
        if m_pct:
            try:
                pct = int(m_pct.group(1))
                if 0 <= pct <= 100:
                    yield {"type": "progress", "pct": pct}
            except Exception:
                pass

        # Parse epoch and metrics (if present)
        epoch = None
        max_epoch = None
        m = epoch_re.search(line)
        if m:
            epoch = int(m.group(1))
            max_epoch = int(m.group(2))
        else:
            m2 = one_epoch_re.search(line)
            if m2:
                epoch = int(m2.group(1))

        metrics = _parse_keyvals(line)
        aliases = {
            "trainacc": "train_acc",
            "valacc": "val_acc",
            "accuracy": "acc",
            "trainaccuracy": "train_acc",
            "valaccuracy": "val_acc",
            "precision": "prec",
            "precision_labeled": "prec_labeled",
        }
        for k, v in list(metrics.items()):
            nk = aliases.get(k, k)
            if nk != k:
                metrics[nk] = v

        if epoch is not None or any(k in metrics for k in (
            "train_loss", "val_loss", "train_acc", "val_acc",
            "prec_labeled", "prec", "weighted_purity_on_labeled"
        )):
            yield {"type": "epoch", "epoch": epoch, "max_epoch": max_epoch, "metrics": metrics}

    code = proc.wait()
    out = "\n".join(collected_out)
    err = ""  # merged

    # Try to parse trailing JSON summary from the tail
    summary = None
    joined_tail = "\n".join(last_json_candidate)
    try:
        last = joined_tail.rfind("{")
        if last >= 0:
            summary = json.loads(joined_tail[last:])
            yield {"type": "summary", "summary": summary}
    except Exception:
        summary = None

    result = RunResult(ok=(code == 0), cmd=cmd, summary=summary, stdout=out, stderr=err)
    yield {"type": "done", "result": result}


def run_external(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Runs external models via the registry and returns their outputs.

    Args:
        ctx (Dict[str, Any]): Context with keys: people_csv, labels_csv, run_shard_dir, sample_frac, epochs, models.

    Returns:
        Dict[str, Any]: Mapping of model short-name to artifacts.
    """
    if run_external_models is None or TrainingContext is None:
        return {"error": "external models registry unavailable"}
    tc = TrainingContext(
        people_csv=ctx["people_csv"],
        labels_csv=ctx["labels_csv"],
        run_shard_dir=ctx["run_shard_dir"],
        sample_frac=float(ctx.get("sample_frac", 1.0)),
        epochs=int(ctx.get("epochs", 1)),
    )
    return run_external_models(list(ctx.get("models", [])), tc)


# -----------------------
# Artifact scanning/priming
# -----------------------

# Conservative allow/exclude lists so we don't overwrite model results.
_ALLOWED_DIRS = {
    "blocks", "blocking", "candidates", "pairs", "features", "features_cache",
    "cache", "previews", "tmp"
}
_EXCLUDE_DIRS = {
    "gbdt", "logreg", "splink", "dedupe"
}
# File name patterns (lowercased) that are *outputs* we do NOT copy over.
_EXCLUDE_FILE_PATTERNS = (
    "matches_",           # matches_*.csv
    "compare_backends.json",
    "per_row_matches.csv",
    "clusters.csv",
    "edges.csv",
)

def _bytes_to_human(n: int) -> str:
    """Converts a byte count to a human-readable string (e.g., KB, MB, GB).

    Args:
        n (int): The number of bytes.

    Returns:
        str: A human-readable string representing the size.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def scan_artifacts(shard_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scans a shard directory for reusable artifacts from a previous run.

    This identifies heavy, pre-computed files and directories (like blocks and features)
    that can be copied to a new run directory to save time, while ignoring model
    outputs and results.

    Args:
        shard_dir (str): The path to the shard directory to scan.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing two lists:
            - A list of reusable directories.
            - A list of reusable files.
            Each item is a dictionary with 'path' and 'size' keys.
    """
    dir_items: List[Dict[str, Any]] = []
    file_items: List[Dict[str, Any]] = []

    if not os.path.isdir(shard_dir):
        return dir_items, file_items

    # Directories deemed reusable
    for name in sorted(os.listdir(shard_dir)):
        p = os.path.join(shard_dir, name)
        if os.path.isdir(p):
            low = name.lower()
            if low in _ALLOWED_DIRS and low not in _EXCLUDE_DIRS:
                # compute size (best-effort)
                total = 0
                for root, _, files in os.walk(p):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        try:
                            total += os.path.getsize(fp)
                        except Exception:
                            pass
                dir_items.append({"path": p, "size": total})

    # Top-level reusable files (skip big model outputs)
    for name in sorted(os.listdir(shard_dir)):
        p = os.path.join(shard_dir, name)
        if os.path.isfile(p):
            low = name.lower()
            if any(low.startswith(bad) or low == bad for bad in _EXCLUDE_FILE_PATTERNS):
                continue
            try:
                sz = os.path.getsize(p)
            except Exception:
                sz = 0
            file_items.append({"path": p, "size": sz})

    return dir_items, file_items


def prime_out_dir(src_shard_dir: str, dst_shard_dir: str) -> Dict[str, Any]:
    """Copies reusable artifacts from a source directory to a destination directory.

    This "primes" a new output directory with pre-computed data from a previous
    run, accelerating the pipeline by skipping initial heavy computation steps.

    Args:
        src_shard_dir (str): The source directory containing the reusable artifacts.
        dst_shard_dir (str): The destination directory for the new run.

    Returns:
        Dict[str, Any]: A report dictionary with lists of 'copied' and 'skipped' items.
    """
    os.makedirs(dst_shard_dir, exist_ok=True)

    copied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Directories
    for name in sorted(os.listdir(src_shard_dir)):
        sp = os.path.join(src_shard_dir, name)
        if not os.path.isdir(sp):
            continue
        low = name.lower()
        if low in _ALLOWED_DIRS and low not in _EXCLUDE_DIRS:
            dp = os.path.join(dst_shard_dir, name)
            try:
                # Merge copy (py3.8+: dirs_exist_ok)
                shutil.copytree(sp, dp, dirs_exist_ok=True)
                copied.append({"path": dp, "kind": "dir"})
            except Exception as e:
                skipped.append({"path": sp, "kind": "dir", "reason": str(e)})

    # Files
    for name in sorted(os.listdir(src_shard_dir)):
        sp = os.path.join(src_shard_dir, name)
        if not os.path.isfile(sp):
            continue
        low = name.lower()
        if any(low.startswith(bad) or low == bad for bad in _EXCLUDE_FILE_PATTERNS):
            continue
        dp = os.path.join(dst_shard_dir, name)
        try:
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            shutil.copy2(sp, dp)
            copied.append({"path": dp, "kind": "file"})
        except Exception as e:
            skipped.append({"path": sp, "kind": "file", "reason": str(e)})

    return {"copied": copied, "skipped": skipped}
