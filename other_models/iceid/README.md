# ICE-ID 2.0: Entity Resolution for Icelandic Historical Census Data

**ICE-ID** (Icelandic Census Entity IDentification) is a comprehensive entity resolution system designed to link historical census records and identify unique individuals across multiple data sources. The system combines modern machine learning techniques with domain-specific knowledge to handle the unique challenges of Icelandic historical data.

## What is ICE-ID?

ICE-ID solves the challenging problem of **entity resolution** (also known as record linkage or deduplication) for Icelandic historical census data. Given thousands of census records with:
- Name variations and spelling inconsistencies
- Missing or incomplete information
- Family relationships (parents, partners, children)
- Geographic and temporal information

ICE-ID predicts which records refer to the same individual, forming clusters of matched identities. This is critical for:
- 📊 **Historical demography research** - Understanding population dynamics over time
- 👨‍👩‍👧‍👦 **Genealogical studies** - Reconstructing family trees and lineages
- 🔬 **Social science research** - Analyzing historical social structures and mobility

## Key Features

This repository provides a **complete end-to-end system** with:

- 🎯 **Multiple ML Models** - Train and compare 12+ entity resolution approaches (GBDT, Neural Networks, TF-IDF, Ditto, ZeroER, Random Forest, and more)
- 📊 **Interactive Dashboard** - Streamlit-based UI for training, evaluation, and manual inspection
- ✏️ **Manual Editing Tools** - Inspect and correct predictions with visual graph networks
- 📈 **Comprehensive Evaluation** - Precision metrics, confidence intervals, cluster purity analysis
- 🔄 **Version Control** - Save and track manual edits to predictions
- 🌳 **Genealogical Visualization** - Explore family relationships in interactive tree graphs

## Who Should Use ICE-ID?

- **Researchers** working with historical census data or entity resolution problems
- **Data scientists** developing record linkage systems
- **Genealogists** analyzing Icelandic family histories
- **Historians** studying population dynamics and social structures

---

## Folder structure

```
.
├── dashboard
│   ├── app.py                 # Streamlit app with 3 tabs: Training, Evaluation, Inspect & Edit
│   ├── ds_io.py               # Data I/O helpers (people/edges/clusters), caching, canonicalization
│   ├── edits.py               # Edge add/remove + reclustering (Union-Find)
│   ├── eval_api.py            # Programmatic evaluation of backends (precision, CI, cluster purity, etc.)
│   ├── graphing.py            # PyVis graph builders: match network & family tree
│   ├── __init__.py
│   ├── inspector_tab.py       # Full Inspect/Edit UI, factored out of app.py
│   └──  train_api.py           # CLI builder + streaming process runner; artifact scanning/priming
├── raw_data
│   ├── counties.csv
│   ├── districts.csv
│   ├── manntol_einstaklingar_new.csv
│   ├── parishes.csv
│   └── people.csv
└── runs
    └── hundred_loose_dual
        └── shard_s0_of_1
            ├── artifacts.json
            ├── compare_backends.json
            ├── gbdt/
            │   ├── cluster_eval_val_details.csv
            │   ├── cluster_eval_val.json
            │   ├── clusters.csv
            │   ├── edges.csv
            │   └── matches_gbdt.csv
            ├── gbdt_metrics.json
            ├── logreg/
            │   ├── cluster_eval_val_details.csv
            │   ├── cluster_eval_val.json
            │   ├── clusters.csv
            │   ├── edges.csv
            │   └── matches_logreg.csv
            ├── logreg_metrics.json
            ├── per_row_matches.csv
            └── threshold_window_samples.csv
```

### High-level flow

- **Training tab (`app.py` ⇄ `train_api.py`)**
  - Builds and runs the full clustering pipeline (`python -m other_models.main …`, wired via `train_api.build_command`).
  - Live logs, progress, and metric curves are parsed and displayed as training proceeds.
  - **Priming** can re-use heavy precomputed artifacts from another run (blocks/candidates/features).
  - **Device toggle** lets you choose **auto / cpu / cuda**; `train_api` passes `--device` to the backend and enforces CPU via `CUDA_VISIBLE_DEVICES=""` when chosen.
  - **Real-time progress feedback** with progress bar and status updates showing which model is currently training.

- **Evaluation tab (`app.py` ⇄ `eval_api.py`)**
  - Loads `people.csv` + labels (`manntol_einstaklingar_new.csv`) to construct ground truth.
  - For each backend subfolder (e.g., `gbdt/`, `logreg/`), reads `edges.csv` (+ `clusters.csv` if present) and computes:
    - Precision on labeled pairs with **Wilson 95% CI**,
    - counts of novel / semi-novel edges,
    - coverage (nodes touched),
    - cluster purity / size stats if `clusters.csv` exists.
  - Auto-interpretation explains metrics in plain language.
  - **Real-time progress feedback** with expandable status container showing which backend is currently being analyzed.

- **Inspect & Edit tab (`inspector_tab.py` + `ds_io.py` + `graphing.py` + `edits.py`)**
  - Loads a backend’s `edges.csv` + `clusters.csv` and `raw_data/people.csv`.
  - Visualizes:
    - **Match Network** between predicted identities (PyVis).
    - **Genealogical Tree** derived from `father / mother / partner` (PyVis).
  - Edit tools:
    - **Add link** (merge identities) or **Remove link** (unmerge).
    - **Rebuild clusters** from edited edges (Union-Find).
    - **Save snapshot** into `versions/` as a new version (with `meta.json`, `edges.csv`, `clusters.csv`).

---

## Quick start

### 1) Environment

#### With Pipenv (recommended, since you already used it)

```bash
# from repo root
pipenv install --python 3.10
pipenv shell
pip install -U pip
# install extras you need (streamlit, pandas, pyvis, plotly, etc.)
pip install streamlit pandas numpy pyvis plotly scikit-learn
```

#### With `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt   # if provided
# or install packages manually if you're developing
```

### 2) Launch the dashboard

From the repo root:

```bash
streamlit run dashboard/app.py
```
or with logs
```bash
python -m streamlit run dashboard/app.py --logger.level=debug
```

The app expects the default data under `raw_data/` and a sample run under `runs/hundred_loose_dual/shard_s0_of_1/`. You can override paths in the UI.

---

## Using the dashboard

### Training tab

- **Data paths**
  - `people.csv`: main table (id, names, birthyear, sex, parents/partner, etc.).
  - `labels.csv`: `manntol_einstaklingar_new.csv` (or any file with `id` → `bi_einstaklingur` mapping). If your `people.csv` includes a `person` column, that takes precedence for ground truth in evaluation.
- **Core params**
  - `backends`: which matchers to run (e.g., `gbdt,logreg`).
  - `num_shards`, `shard_id`: split and run subsets.
  - Blocking/candidate params: `max_block_size`, `max_pairs_per_block`, name prefix lengths, year bucket width, filters, etc.
- **Device selection**
  - `auto`: don’t force a device; backend decides.
  - `cpu`: passes `--device cpu` and sets `CUDA_VISIBLE_DEVICES=""` to avoid CUDA.
  - `cuda`: passes `--device cuda` (backend should error if no GPU).
- **Priming**
  - Point **Source shard dir** to an existing run’s shard (e.g., a finished `shard_s0_of_1`) and **Prime** your **Target shard dir** to copy heavy upstream artifacts (blocks, candidates, features) to speed up re-runs. Model outputs (`edges.csv`, `clusters.csv`, etc.) are *not* copied.
- **Live view**
  - Logs stream as the pipeline runs; the app parses “`key=value`” pairs for loss/accuracy/precision curves and a final JSON summary if the backend prints one.

### Evaluation tab

- Choose:
  - `people.csv` path,
  - `labels.csv` path,
  - `Shard directory` (the folder that contains your backend subfolders or its own `edges.csv`/`clusters.csv`).
- Optional: list backends manually (`gbdt,logreg`), or leave blank to auto-detect.
- Click **Analyze**. You’ll see:
  - A compact per-backend table,
  - Metric explanations,
  - Auto-interpretation (precision grade, CI tightness, discovery potential, coverage, cluster purity).

### Inspect & Edit tab

- **Load sources**
  - Set `people.csv`, `Run shard directory`, and a `Versions root` for saving snapshots.
  - Optionally load a **saved version** as baseline.
- **Select a backend**
  - Dropdown from auto-detected backends under the shard directory.
- **Cluster view**
  - Summary comparing predicted cluster to hand labels (`person`), highlighting:
    - Anchor (majority GT - ground truth - group),
    - Discrepant (other GT groups),
    - New (no GT).
- **Visualizations**
  - **Match Network** (within cluster):
    - Group nodes by `Label Status`, `Hand-Label (person)`, `Parish`, or `Census (heimild)`.
    - Control node limits.
  - **Genealogical Tree**:
    - Build from `father/mother/partner` relationships around a root or all members.
    - Toggle physics/hierarchical layout; control depth and node limits.
- **Editing**
  - **Add link** between two record IDs (within or outside the cluster).
  - **Remove link** by selecting from existing in-cluster edges or by specifying any pair (IDs).
  - Each edit **rebuilds clusters** via Union-Find and jumps to the cluster containing your focus record.
- **Save snapshot**
  - Writes a new `versions/<timestamped_or_named>/` with:
    - `meta.json` (provenance),
    - `edges.csv` (canonicalized),
    - `clusters.csv` (`members` saved as `id1;id2;…`).

---

## File-by-file detail

### `dashboard/app.py`
- **The Streamlit app**. Defines three tabs (“Training”, “Evaluation”, “Inspect & Edit”).
- Handles UI widgets, validation, and live run rendering.
- Wires to:
  - `train_api` for building/running the training pipeline,
  - `eval_api.analyze_backends` for evaluation,
  - `inspector_tab.show_inspector_tab` for the editor.
- **Device control**: exposes a `Compute device` selectbox and plumbs it to the backend.

### `dashboard/train_api.py`
- **build_command(…)**:
  - Constructs the `python -u -m other_models.main …` CLI with all parameters from the UI.
  - Passes `--device` when set, and respects `soft_filter_*`, blocking parameters, etc.
- **run_pipeline_streaming(…)**:
  - Spawns the subprocess and yields structured events:
    - `line` (raw log line),
    - `progress` (percent from tqdm-like logs),
    - `epoch` (parsed `key=value` metrics),
    - `summary` (final trailing JSON),
    - `done` (exit code + buffers).
- **scan_artifacts / prime_out_dir**:
  - Identify and copy reusable upstream artifacts (blocks, candidates, features) to speed up re-runs.

### `dashboard/eval_api.py`
- Loads ground truth by preferring `people.person` if present, otherwise merging `labels_csv`’s `bi_einstaklingur`.
- For each backend:
  - Reads `edges.csv` (+ `clusters.csv` if present),
  - Computes labeled precision with **Wilson 95% CI**,
  - Counts semi-novel & novel edges, estimated true discoveries,
  - Coverage (nodes with any match),
  - Cluster metrics (singleton rate, size stats, labeled purity).
- Returns a JSON-like dict friendly to the UI.

### `dashboard/inspector_tab.py`
- Implements the **full Inspect/Edit UI** as a callable tab (`show_inspector_tab`).
- Manages session state:
  - Baseline backend,
  - Working copy (editable) edges & clusters,
  - People/Parish maps.
- Calls into:
  - `ds_io` for loading, caching, and canonicalization,
  - `graphing` for PyVis network and family tree,
  - `edits` for link add/remove and reclustering,
  - `save_version` to persist snapshots.

### `dashboard/ds_io.py`
- **load_people(…)**:
  - Reads `people.csv`, selects/coerces columns, normalizes strings to lowercase, makes IDs numeric (`Int64`→`int64`).
  - Raises a helpful error if `people.csv` is missing.
- **load_parish_map(…)**:
  - Maps parish `id → name` (used for legend labeling).
- **detect_backends(…)**:
  - Finds subdirectories with `clusters.csv`.
- **_read_*_cached** + `st.cache_*`:
  - Efficient caching of CSV reading/parsing in Streamlit runs.
- **canonicalize_edges(…)**:
  - Ensure undirected `(id1,id2)` with `id1 < id2` and deduplicate.
- **save_version / load_saved_version**:
  - Version snapshots for edited results.
- **build_children_index(…)**:
  - Precomputed parent/partner indices for fast queries.

### `dashboard/edits.py`
- Minimal **Union-Find (DSU)** + utilities:
  - `add_edge`, `remove_edge`, `canonicalize_edges`,
  - `rebuild_clusters_from_edges` to recompute connected components and produce `clusters_df`.

### `dashboard/graphing.py`
- **build_cluster_network_edges**:
  - Filters to member subgraph; degree-prunes if node limit exceeded.
- **build_family_tree_edges**:
  - BFS over `father/mother/partner` relationships with depth & node limits.
- **build_pyvis_network_graph / build_pyvis_family_tree**:
  - Create styled PyVis graphs (fixed column layout for networks; hierarchical or physics for family trees).
  - Legend color mapping and groupings (Label Status, Parish, etc.).
- **to_graphviz_digraph**:
  - DOT export (kept for reference/back-compat).
---

## Data expectations

- **`raw_data/people.csv`** should contain (or the code will create defaults for missing columns):
  - `id`, `heimild`, `nafn_norm`, `first_name`, `middle_name`, `patronym`, `surname`,
  - `birthyear`, `sex`, `status`, `marriagestatus`, `person`, `partner`, `father`, `mother`,
  - `farm`, `county`, `parish`, `district`.
- **`raw_data/manntol_einstaklingar_new.csv`** should map `id → bi_einstaklingur` (if `people.person` absent).
- The sample **`runs/hundred_loose_dual/shard_s0_of_1/`** contains two backends (`gbdt/`, `logreg/`) with `edges.csv` and `clusters.csv`.

---

## GPU vs CPU

- In the **Training** tab, pick **Compute device**:
  - **auto**: leave decisions to the backend (default).
  - **cpu**: sets `--device cpu` and `CUDA_VISIBLE_DEVICES=""` so common DL libraries avoid CUDA.
  - **cuda**: sets `--device cuda`; backend should error if no GPU is available.
- This is wired through `train_api.build_command(…, device=…)` and enforced in `run_pipeline_streaming(…, env=…)`.

---

## Advanced: flexible schemas and backend hooks

### Flexible CSV schemas (alternate headers)

Use lightweight schema objects to adapt non-standard column names without changing core code.

```python
from dashboard.schemas import PeopleSchema, LabelsSchema, EdgeSchema, ClusterSchema, schema_signature
from dashboard import ds_io, eval_api

# People CSV with different headers
people_schema = PeopleSchema(rename_map={
    "ID": "id",
    "FirstName": "first_name",
    "BirthYear": "birthyear",
})

people_csv = "/path/to/people_alt.csv"
people = ds_io.load_people(
    people_csv,
    ds_io._file_sig(people_csv),
    schema=people_schema,
    schema_sig=schema_signature(people_schema),
)

# Labels CSV with different id/cluster column names
labels_schema = LabelsSchema(id_col="rec_id", cluster_col="entity_id")
results = eval_api.analyze_backends(
    run_dir="/path/to/run",
    people_csv=people_csv,
    labels_csv="/path/to/labels_alt.csv",
    labels_schema=labels_schema,
)

# Backend outputs with alternate edge/cluster headers
edges, clusters = ds_io.load_edges_clusters_with_schema(
    backend_dir="/path/to/run/gbdt",
    edge_schema=EdgeSchema(id1_col="left", id2_col="right"),
    cluster_schema=ClusterSchema(
        cluster_id_col="cid",
        size_col="n",
        members_col="members",
        members_sep=";",
    ),
)
```

All schema parameters are optional; default behavior remains unchanged if you omit them.

### Backend hooks (additional models and flags)

Register backends and compose extra CLI flags without changing existing flows.

```python
from dashboard.backends import BackendSpec, register_backend, build_extra_args
from dashboard.train_api import build_command

# Register a custom backend with default flags
register_backend(BackendSpec(name="xgb", cli_args={"n_estimators": "200"}))

# Build extra args and include the custom backend
extra = build_extra_args({"xgb": {"max_depth": "6"}})
cmd = build_command(
    people_csv="/data/people.csv",
    labels_csv="/data/labels.csv",
    out_dir="/runs/exp1",
    backends_list=["gbdt", "xgb"],
    extra_args=extra,
)
```

`backends_list` and `extra_args` are optional and preserve prior defaults if not used.

### Settings Management

Save and load model configurations with PyTorch and scikit-learn compatibility:

```python
from dashboard.settings_manager import settings_manager, ModelConfig, PyTorchModelSettings

# Create a custom PyTorch configuration
config = ModelConfig(
    model_name="My Neural Network",
    model_type="pytorch",
    pytorch_settings=PyTorchModelSettings(
        hidden_dim=512,
        learning_rate=1e-4,
        batch_size=64,
        epochs=20,
        optimizer="adamw"
    ),
    sample_frac=0.2
)

# Save configuration
settings_manager.save_config(config, "my_neural_net")

# Load configuration
loaded_config = settings_manager.load_config("my_neural_net")

# Get configuration summary
summary = settings_manager.get_config_summary(loaded_config)
```

#### Available Model Types

- **ICE-ID Pipeline**: Traditional entity resolution pipeline
- **Ditto (HF)**: HuggingFace DistilBERT-based classifier
- **ZeroER (SBERT)**: Sentence-BERT zero-shot approach
- **TF-IDF + Logistic Regression**: Traditional ML baseline
- **Random Forest**: Ensemble tree-based model
- **Gradient Boosting**: Gradient boosting classifier

#### PyTorch Settings

- Architecture: `hidden_dim`, `num_layers`, `dropout`, `activation`
- Training: `learning_rate`, `batch_size`, `epochs`, `weight_decay`
- Optimizer: `optimizer`, `scheduler`, `warmup_steps`
- Data: `max_length`, `padding`, `truncation`
- Performance: `device`, `mixed_precision`, `num_workers`

#### Scikit-learn Settings

- Random Forest: `rf_n_estimators`, `rf_max_depth`, `rf_min_samples_split`
- Gradient Boosting: `gb_n_estimators`, `gb_learning_rate`, `gb_max_depth`
- Logistic Regression: `lr_penalty`, `lr_C`, `lr_solver`
- SVM: `svm_C`, `svm_kernel`, `svm_gamma`
- Cross-validation: `cv_folds`, `cv_scoring`

#### Data Processing Settings

- Text: `text_max_length`, `text_lowercase`, `text_stemming`
- Features: `use_tfidf`, `tfidf_max_features`, `tfidf_ngram_range`
- Numerical: `normalize_numerical`, `numerical_scaler`
- Categorical: `encode_categorical`, `categorical_encoding`
- Splitting: `train_split`, `val_split`, `test_split`

## Tab-by-tab quick guide to new features

### Inspect & Edit
- Start editing via the button in `Current Dataset Pointers`. Edits apply to a working copy only.
- Edge Navigation shows labels (Anchor Group, Discrepant Member, New Prediction) and row colors.
- Add/Remove edges updates clusters immediately. Save a snapshot under `versions/`.
- Quick Dataset Swap appears next to Edge Navigation and in the sidebar to switch between backends and saved versions.

### Evaluation
- Select a run directory; backends auto-populate. Metrics table plus detailed conclusions are shown.

### Training
- Priming copies reusable artifacts from a source shard to your target to accelerate runs.
- Device selection supports Auto/CPU/GPU with environment wiring.

### Data Editor
- Pick a CSV from `raw_data/`.
- Column Operations:
  - Rename a column: select and provide a new name. Prevents collisions.
  - Drop columns: multi-select and apply. Changes affect the in-memory working copy only.
  - Reset to original restores from disk.
- Downloads:
  - Working Copy (edited), Original (from disk).
- Missing Data Analysis:
  - Summary table of counts and percentages.
  - Visuals using `missingno` (matrix, bar, and correlation heatmap for ≤20 columns). Install with `pip install missingno`.
- Data Expectations expander documents required columns per model type.

## Folder and file reference

### Top-level
- `dashboard/`: Streamlit app and supporting modules.
- `raw_data/`: Input CSVs used by the dashboard and pipeline.
- `runs/`: Outputs from training/evaluation runs (organized by experiment and shard).
- `versions/` (created by you from the Inspector tab): Saved snapshots of edited results.
- `other_models/` (if present at repo root): Additional training/eval code not specific to the dashboard.

### raw_data/
- `people.csv`
  - One row per record (census entry). Expected columns (minimum useful set):
    - `id` (int): unique record id
    - `first_name`, `middle_name`, `patronym`, `surname` (str): normalized lowercase name fields
    - `birthyear` (int), `sex` (str: e.g., `karl`, `kona`)
    - `person` (int or empty): ground-truth person id (cluster label) if available
    - `father`, `mother`, `partner` (int or empty): relationship links to other `id`s
    - `farm`, `county`, `parish`, `district`, `heimild` (ints): locality/source ids
  - The loader (`ds_io.load_people`) coerces types and lowercases strings; missing columns are added as empty if absent.
- `manntol_einstaklingar_new.csv`
  - Ground-truth labels table used during evaluation when `people.person` is missing.
  - Must contain a mapping from a record id to a cluster/person id (e.g., `id` → `bi_einstaklingur`).
- `parishes.csv`, `districts.csv`, `counties.csv`
  - Lookup tables. Expected columns include `id` and `name` (or similarly named). Used to map ids → human-readable names (e.g., legend labels in graphs).

### runs/
- Structure is typically: `runs/<experiment_name>/shard_s<i>_of_<n>/`.
- Within each shard directory you may have one or more backend folders (e.g., `gbdt/`, `logreg/`), each containing:
  - `edges.csv`: predicted undirected links, columns `id1`, `id2` (canonicalized with `id1 < id2`).
  - `clusters.csv`: connected components from `edges.csv`, columns:
    - `cluster_id` (int)
    - `size` (int)
    - `members` (str): semicolon-separated ids, e.g., `1;2;3`.
  - Optional diagnostics (e.g., `matches_*.csv`, `cluster_eval_val.json`, etc.).
- Shard folders may also contain heavy precomputed artifacts (e.g., `blocks/`, `candidates/`, `features/`) that can be reused via the Training tab’s “Prime output shard” feature.

### versions/
- Created by the Inspector tab when you click “Save snapshot”. A version directory contains:
  - `meta.json`: provenance (source backend dir, people.csv path, timestamp, comment).
  - `edges.csv`, `clusters.csv`: your edited working copy at save time.
- You can load any version as a baseline from the Inspector tab.

### Evaluation outputs
- Evaluation tab summarizes per-backend metrics, using `people.csv` and either `people.person` or `manntol_einstaklingar_new.csv`.
- Key computed metrics include precision (with Wilson 95% CI), coverage, and cluster purity; details are rendered in the UI, not stored back to disk by default.

### Data editing (Data Editor tab)
- Operates on an in-memory working copy for the selected CSV.
- Supported operations: rename columns, drop columns, reset to original.
- "Download Working Copy" exports the edited table; the original file on disk is not modified.
- Missing data analysis: summary table plus optional visuals (requires `missingno`).
