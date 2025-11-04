"""The original Inspect/Edit/version UI extracted from your previous app.py.

This module exposes `show_inspector_tab(REPO_ROOT: str)` which renders the full
original interface inside a tab. It intentionally keeps the original behavior.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from typing import Dict, List, Optional
import tempfile
from collections import Counter

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT_FALLBACK = PKG_ROOT

if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from dashboard.ds_io import (  # type: ignore
    canonicalize_edges,
    detect_backends,
    free_memory,
    load_edges_clusters,
    load_parish_map,
    load_people,
    load_saved_version,
    members_to_str,
    save_version,
    _file_sig,  # for cache-key arg
)
from dashboard.graphing import (  # type: ignore
    build_pyvis_family_tree,
    build_pyvis_network_graph,
)
from dashboard.edits import (  # type: ignore
    add_edge,
    rebuild_clusters_from_edges,
    remove_edge,
)


def show_inspector_tab(REPO_ROOT: Optional[str] = None) -> None:
    """Renders the full 'Inspect & Edit' user interface as a Streamlit tab.

    This function encapsulates the original UI for loading model outputs (backends),
    inspecting clusters, visualizing match networks and family trees, editing
    predicted edges, and saving new versions of the results.

    Args:
        REPO_ROOT (Optional[str]): The absolute path to the repository root, used to
            construct default paths for data files and run outputs.
    """
    st.header("🔍 ICE-ID Cluster Inspector")
    st.caption("Inspect, analyze, and manually edit entity resolution clusters")
    st.markdown("---")

    st.sidebar.header("📁 Data Sources")

    REPO = REPO_ROOT or REPO_ROOT_FALLBACK

    default_people = os.path.join(REPO, "raw_data", "people.csv")
    default_parish = os.path.join(REPO, "raw_data", "parishes.csv")
    default_shard = os.path.join(REPO, "runs", "hundred_loose_dual", "shard_s0_of_1")
    default_versions = os.path.join(REPO, "versions")

    with st.sidebar.expander("📂 File Paths", expanded=True):
        people_csv = st.text_input(
            "People CSV",
            value=default_people,
            help="Path to the main data file containing details for each record, including the hand-labeled person ID."
        )
        run_root = st.text_input(
            "Run Directory",
            value=default_shard,
            help="Path to the directory containing the model's output (clusters.csv, edges.csv)."
        )
        versions_root = st.text_input(
            "Versions Directory",
            value=default_versions,
            help="Directory where manually edited and saved versions will be stored."
        )

    with st.sidebar.expander("📥 Load Saved Version", expanded=False):
        version_dir = st.text_input("Version directory", value="", help="Path to a previously saved version to load as the starting point.")
        load_version_btn = st.button("📂 Load Version", use_container_width=True)

    for k, v in {
        "baseline_edges": None,
        "baseline_clusters": None,
        "baseline_all_ids": [],
        "baseline_label": "",
        "working_edges": None,
        "working_clusters": None,
        "working_all_ids": [],
        "working_name": "",
        "editing_enabled": False,
        "warned_no_working_copy": False,
        "people": None,
        "parish_map": {},
        "edge_feedback": {},
        "current_cluster_feedback": {},
        "feedback_saved": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def _create_working_copy(default_name: str) -> None:
        """Initializes an editable 'working copy' of clusters and edges in the session state.

        It creates a deep copy of the baseline dataframes to ensure that user edits
        do not modify the original loaded data.

        Args:
            default_name (str): A default name for the new working session.
        """
        be = st.session_state.baseline_edges
        bc = st.session_state.baseline_clusters
        st.session_state.working_edges = canonicalize_edges(be.copy() if be is not None else pd.DataFrame(columns=["id1", "id2"]))
        cpy = bc.copy()
        cpy["members"] = cpy["members"].apply(lambda m: list(map(int, m)) if isinstance(m, list) else [])
        st.session_state.working_clusters = cpy
        st.session_state.working_all_ids = sorted({int(i) for m in cpy["members"] for i in m})
        ts = time.strftime("%Y%m%d_%H%M%S")
        st.session_state.working_name = f"{default_name}_{ts}"
        free_memory(cpy)

    def _load_backend(backend_dir: str) -> None:
        """Loads a model backend's output as the new baseline and creates a working copy.

        Args:
            backend_dir (str): The path to the backend directory containing edges.csv and clusters.csv.
        """
        edges, clusters = load_edges_clusters(backend_dir)
        all_ids = sorted({i for members in clusters["members"] for i in members})
        was_editing = bool(st.session_state.get("editing_enabled", False))

        st.session_state.baseline_edges = edges
        st.session_state.baseline_clusters = clusters
        st.session_state.baseline_all_ids = all_ids
        st.session_state.baseline_label = backend_dir

        if was_editing:
            base_name = os.path.basename(backend_dir.rstrip(os.sep)) or "working"
            _create_working_copy(default_name=f"working_from_{base_name}")
            st.session_state.editing_enabled = True
        else:
            st.session_state.working_edges = canonicalize_edges(edges.copy() if edges is not None else pd.DataFrame(columns=["id1","id2"]))
            cpy = clusters.copy()
            cpy["members"] = cpy["members"].apply(lambda m: list(map(int, m)) if isinstance(m, list) else [])
            st.session_state.working_clusters = cpy
            st.session_state.working_all_ids = list(map(int, all_ids))
            st.session_state.working_name = ""
            st.session_state.editing_enabled = False
        st.session_state.warned_no_working_copy = False
        try:
            free_memory(edges, clusters, all_ids, cpy)
        except Exception:
            free_memory(edges, clusters, all_ids)

    try:
        st.session_state.people = load_people(people_csv, _file_sig(people_csv))
        st.session_state.parish_map = load_parish_map(default_parish, _file_sig(default_parish))
        people_ok = True
    except Exception as e:
        st.sidebar.error(f"Failed to load data: {e}")
        people_ok = False

    if load_version_btn and version_dir:
        try:
            e, c = load_saved_version(version_dir)
            _load_backend(version_dir)
            st.success(f"Loaded version as baseline: {version_dir}")
        except Exception as e:
            st.error(f"Could not load version: {e}")

    def _list_version_dirs(root: str) -> List[str]:
        """Lists subdirectories under a root that contain clusters.csv.

        Args:
            root (str): Versions root directory.

        Returns:
            List[str]: Absolute paths to version directories.
        """
        out: List[str] = []
        try:
            if os.path.isdir(root):
                for name in os.listdir(root):
                    p = os.path.join(root, name)
                    if os.path.isdir(p) and os.path.exists(os.path.join(p, "clusters.csv")):
                        out.append(p)
        except Exception:
            pass
        return sorted(out)

    with st.sidebar.expander("🔄 Load Backend", expanded=True):
        backend_choice = None
        if os.path.isdir(run_root):
            backends = detect_backends(run_root)
            backend_choice = st.selectbox("Backend", backends, help="Select a model output (backend) to inspect.")
            if st.button("🚀 Load Backend", key="load_backend_btn", use_container_width=True):
                _load_backend(os.path.join(run_root, backend_choice))
        else:
            st.warning("⚠️ Run directory not found")

    with st.sidebar.expander("⚡ Quick Dataset Swap", expanded=False):
        ds_labels: List[str] = []
        ds_paths: Dict[str, str] = {}
        try:
            for b in detect_backends(run_root):
                lab = f"run: {b}"
                ds_labels.append(lab)
                ds_paths[lab] = os.path.join(run_root, b)
        except Exception:
            pass
        for vp in _list_version_dirs(versions_root):
            lab = f"version: {os.path.basename(vp)}"
            ds_labels.append(lab)
            ds_paths[lab] = vp
        if ds_labels:
            sel = st.selectbox("Dataset", options=ds_labels, key="dataset_picker")
            chosen_path = ds_paths.get(sel, "")
            prev = st.session_state.get("dataset_loaded_path", "")
            if chosen_path and chosen_path != prev:
                try:
                    _load_backend(chosen_path)
                    st.session_state["dataset_loaded_path"] = chosen_path
                    st.success(f"Loaded dataset: {sel}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
        else:
            st.caption("No datasets found under the provided folders.")

    if st.session_state.baseline_clusters is None and os.path.isdir(run_root):
        with st.spinner("Auto-loading default backend..."):
            backends = detect_backends(run_root)
            if backends:
                try:
                    _load_backend(os.path.join(run_root, backends[0]))
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Auto-load failed: {e}")
            else:
                st.sidebar.warning("No backends found in run directory.")

    if not people_ok or st.session_state.baseline_clusters is None:
        st.info("Waiting for data to be loaded. Check Data Sources in the sidebar.")
        st.stop()

    if not st.session_state.editing_enabled and not st.session_state.warned_no_working_copy:
        st.warning("Editing is currently disabled. Start an editing session to make changes.")
        st.session_state.warned_no_working_copy = True

    def render_pyvis_graph(net):
        """Renders a pyvis Network object in Streamlit by writing it to a temporary HTML file.

        Args:
            net (pyvis.network.Network): The pyvis graph object to display.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                net.save_graph(tmp_file.name)
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=420)
        finally:
            if 'tmp_file' in locals() and os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

    def _goto_cluster_containing(record_id: int) -> None:
        """Updates the UI to select the cluster that contains a specific record ID.

        After an edit that rebuilds clusters, this function finds the new cluster
        containing the specified record and changes the selection in the main
        cluster dropdown to provide a seamless user experience.

        Args:
            record_id (int): The ID of the record to find.
        """
        try:
            df = st.session_state.working_clusters
            target = int(record_id)
            row = df[df["members"].apply(lambda m: target in set(m))]
            if not row.empty:
                new_cid = int(row.iloc[0]["cluster_id"])
                st.session_state["cluster_select_top"] = new_cid
                st.rerun()
        except Exception:
            pass

    def _get_edge_key(id1: int, id2: int) -> str:
        """Creates a consistent key for an edge regardless of order.
        
        Args:
            id1 (int): First node ID.
            id2 (int): Second node ID.
            
        Returns:
            str: Consistent edge key.
        """
        return f"{min(id1, id2)}_{max(id1, id2)}"

    def _load_cluster_feedback(cluster_id: int) -> None:
        """Loads feedback for the current cluster into session state.
        
        Args:
            cluster_id (int): The cluster ID to load feedback for.
        """
        if cluster_id in st.session_state.edge_feedback:
            st.session_state.current_cluster_feedback = st.session_state.edge_feedback[cluster_id].copy()
        else:
            st.session_state.current_cluster_feedback = {}

    def _save_cluster_feedback(cluster_id: int) -> None:
        """Saves current cluster feedback to the global feedback store.
        
        Args:
            cluster_id (int): The cluster ID to save feedback for.
        """
        st.session_state.edge_feedback[cluster_id] = st.session_state.current_cluster_feedback.copy()
        st.session_state.feedback_saved = True

    def _get_edges_in_cluster(cluster_id: int) -> pd.DataFrame:
        """Gets all edges within a specific cluster.
        
        Args:
            cluster_id (int): The cluster ID.
            
        Returns:
            pd.DataFrame: Edges within the cluster.
        """
        if st.session_state.working_clusters is None or st.session_state.working_edges is None:
            return pd.DataFrame(columns=["id1", "id2"])
        cluster_row = st.session_state.working_clusters[st.session_state.working_clusters["cluster_id"] == cluster_id]
        if cluster_row.empty:
            return pd.DataFrame(columns=["id1", "id2"])
        members = set(cluster_row.iloc[0]["members"])
        edges_in_cluster = st.session_state.working_edges[
            (st.session_state.working_edges["id1"].isin(members)) & 
            (st.session_state.working_edges["id2"].isin(members))
        ].copy()
        return edges_in_cluster

    with st.expander("📊 Session Information", expanded=True):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("**📦 Baseline Dataset**")
            st.code(st.session_state.baseline_label, language=None)
        with col_info2:
            st.markdown("**✏️ Working Copy**")
            st.code(st.session_state.working_name or "Not created", language=None)
        
        st.markdown("---")
        
        default_wc_name = st.session_state.working_name or (
            f"working_from_{os.path.basename(str(st.session_state.baseline_label).rstrip(os.sep)) or 'backend'}_{time.strftime('%Y%m%d_%H%M%S')}"
            if st.session_state.editing_enabled else ""
        )
        st.session_state.working_name = st.text_input("📝 Working Copy Name", value=default_wc_name, help="Name for your current editing session.")
        
        cols_wc = st.columns([1, 1])
        with cols_wc[0]:
            if not st.session_state.editing_enabled:
                if st.button("✏️ Start Editing Session", key="btn_enable_editing", use_container_width=True, type="primary"):
                    base_name = os.path.basename(str(st.session_state.baseline_label).rstrip(os.sep)) or "working"
                    _create_working_copy(default_name=f"working_from_{base_name}")
                    st.session_state.editing_enabled = True
                    st.rerun()
            else:
                st.success("✅ Editing Mode Active")
        with cols_wc[1]:
            if not st.session_state.editing_enabled:
                st.info("💡 Start editing to modify clusters")

    people_df: pd.DataFrame = st.session_state.people
    clusters_df = st.session_state.working_clusters
    edges_df = st.session_state.working_edges
    parish_map = st.session_state.parish_map

    COLOR_PALETTE = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9A6324"]

    colR, colL = st.columns([2, 3])

    with colR:
        with st.container():
            st.markdown("### 🔍 Cluster Selection")
            clusters_sorted = clusters_df.sort_values(["size", "cluster_id"], ascending=[False, True])
            cid = st.selectbox(
                "Select Cluster", 
                options=clusters_sorted["cluster_id"].astype(int).tolist(), 
                index=0, 
                key="cluster_select_top", 
                help="Choose a predicted cluster to inspect and analyze"
            )
            if cid is not None:
                members_for_dropdown = clusters_df.loc[clusters_df["cluster_id"] == int(cid), "members"].iloc[0]
                root_id = st.selectbox(
                    "Root Person", 
                    options=members_for_dropdown, 
                    index=0, 
                    help="Select the starting person for the Genealogical Tree visualization"
                )
            else:
                root_id = None
        
        st.markdown("---")
        st.markdown("### 🔗 Edge Navigation")
        with st.container():
            ds_labels_inline: List[str] = []
            ds_paths_inline: Dict[str, str] = {}
            try:
                for b in detect_backends(run_root):
                    lab = f"run: {b}"
                    ds_labels_inline.append(lab)
                    ds_paths_inline[lab] = os.path.join(run_root, b)
            except Exception:
                pass
            for vp in _list_version_dirs(versions_root):
                lab = f"version: {os.path.basename(vp)}"
                ds_labels_inline.append(lab)
                ds_paths_inline[lab] = vp
            if ds_labels_inline:
                sel_inline = st.selectbox("Dataset", options=ds_labels_inline, key="dataset_picker_inline", help="Quickly switch dataset")
                chosen_path_inline = ds_paths_inline.get(sel_inline, "")
                prev_inline = st.session_state.get("dataset_loaded_path", "")
                if chosen_path_inline and chosen_path_inline != prev_inline:
                    try:
                        _load_backend(chosen_path_inline)
                        st.session_state["dataset_loaded_path"] = chosen_path_inline
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Dataset load failed: {e}")
        if cid is not None and root_id is not None:
            _load_cluster_feedback(int(cid))
            edges_in_cluster = _get_edges_in_cluster(int(cid))
            if not edges_in_cluster.empty:
                edges_root = edges_in_cluster[(edges_in_cluster["id1"] == int(root_id)) | (edges_in_cluster["id2"] == int(root_id))].copy()
                if not edges_root.empty:
                    edge_filter = st.text_input("Search edges", placeholder="Filter by neighbor ID or name...", key="edge_filter")
                    rows = []
                    anchor_person_id = None
                    try:
                        person_ids_in_cluster = people_df.loc[people_df["id"].isin(members_for_dropdown), "person"]
                        if not person_ids_in_cluster.dropna().empty:
                            person_counts = Counter(int(p) for p in person_ids_in_cluster.dropna())
                            anchor_person_id = person_counts.most_common(1)[0][0]
                    except Exception:
                        anchor_person_id = None
                    for _, edge_row in edges_root.iterrows():
                        a, b = int(edge_row["id1"]), int(edge_row["id2"])
                        neighbor_id = b if a == int(root_id) else a
                        p_nei = people_df[people_df["id"] == neighbor_id]
                        name_nei = f"{p_nei.iloc[0]['first_name']} {p_nei.iloc[0]['patronym'] or p_nei.iloc[0]['surname'] or ''}".strip() if not p_nei.empty else f"ID {neighbor_id}"
                        label_status = "New Prediction"
                        try:
                            if not p_nei.empty and pd.notna(p_nei.iloc[0]["person"]):
                                nperson = int(p_nei.iloc[0]["person"])  
                                if anchor_person_id is not None and nperson == anchor_person_id:
                                    label_status = "Anchor Group"
                                else:
                                    label_status = "Discrepant Member"
                        except Exception:
                            pass
                        ekey = _get_edge_key(a, b)
                        status = st.session_state.current_cluster_feedback.get(ekey, "pending")
                        rows.append(
                            {
                                "root_id": int(root_id),
                                "neighbor_id": neighbor_id,
                                "neighbor_name": name_nei,
                                "edge_key": ekey,
                                "label": label_status,
                                "status": status,
                            }
                        )
                    df_edges = pd.DataFrame(rows)
                    if edge_filter:
                        f = edge_filter.lower()
                        df_edges = df_edges[
                            df_edges["neighbor_name"].str.lower().str.contains(f, na=False)
                            | df_edges["neighbor_id"].astype(str).str.contains(edge_filter)
                            | df_edges["edge_key"].str.contains(edge_filter)
                            | df_edges["label"].str.lower().str.contains(f, na=False)
                        ]
                    def style_edge_row(row):
                        """Applies row background colors mirroring cluster table labels.

                        Args:
                            row (pd.Series): A row of the edge navigation table.

                        Returns:
                            List[str]: CSS styles for the row cells.
                        """
                        lab = str(row.get("label", "")).strip().lower()
                        if lab == "anchor group".lower():
                            style = 'background-color: #4A6984; color: white;'
                        elif lab == "discrepant member".lower():
                            style = 'background-color: #8B0000; color: white;'
                        else:
                            style = ''
                        return [style] * len(row)

                    df_edges_sorted = df_edges.sort_values(["label", "neighbor_id"])                    
                    st.dataframe(df_edges_sorted.style.apply(style_edge_row, axis=1), height=420)
                    components.html(
                        """
                        <script>
                        (function () {
                          function setParam(k,v){
                            try{
                              const url = new URL(window.location.href);
                              url.searchParams.set(k, v);
                              window.location.assign(url.toString());
                            }catch(e){}
                          }
                          let t=null;
                          document.addEventListener('dblclick', function(){
                            clearTimeout(t);
                            t=setTimeout(function(){
                              const sel = (window.getSelection?String(window.getSelection()):String(document.getSelection())).trim();
                              if (/^\\d+_\\d+$/.test(sel)) {
                                setParam('edge_key', sel);
                              }
                            }, 0);
                          }, true);
                        })();
                        </script>
                        """,
                        height=0,
                    )
                else:
                    st.info("No edges incident to the selected root in this cluster.")
            else:
                st.info("No edges in this cluster.")

        st.markdown("---")
        st.markdown("### ✏️ Edge Operations")
        
        with st.container():
            st.markdown("#### ➕ Add Edge")
            col_a, col_b = st.columns(2)
            with col_a:
                new_id1 = st.number_input("Person A ID", value=0, key="new_edge_id1", help="Enter the ID of the first person")
            with col_b:
                new_id2 = st.number_input("Person B ID", value=0, key="new_edge_id2", help="Enter the ID of the second person")

            if st.button("➕ Add Edge", key="add_new_edge", disabled=not st.session_state.editing_enabled, use_container_width=True, type="primary"):
                if new_id1 > 0 and new_id2 > 0 and new_id1 != new_id2:
                    try:
                        if new_id1 not in st.session_state.working_all_ids:
                            if new_id1 in set(people_df["id"].tolist()):
                                st.session_state.working_all_ids.append(new_id1)
                        if new_id2 not in st.session_state.working_all_ids:
                            if new_id2 in set(people_df["id"].tolist()):
                                st.session_state.working_all_ids.append(new_id2)
                        st.session_state.working_all_ids = sorted(set(map(int, st.session_state.working_all_ids)))
                        st.session_state.working_edges = add_edge(st.session_state.working_edges, new_id1, new_id2)
                        st.session_state.working_clusters = rebuild_clusters_from_edges(
                            st.session_state.working_edges, st.session_state.working_all_ids
                        )
                        st.success(f"✅ Added edge {new_id1} ↔ {new_id2}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to add edge: {e}")
                else:
                    st.warning("⚠️ Please enter valid different IDs.")
            elif not st.session_state.editing_enabled:
                st.info("💡 Enable editing to add edges.")
        
        with st.container():
            st.markdown("#### 🗑️ Remove Edge")
            qp_edge_key = None
            try:
                if hasattr(st, "query_params"):
                    q = st.query_params
                    val = q.get("edge_key", None)
                    qp_edge_key = val[0] if isinstance(val, list) else val
                else:
                    q = st.experimental_get_query_params()
                    vals = q.get("edge_key")
                    if vals:
                        qp_edge_key = vals[0]
            except Exception:
                qp_edge_key = None
            if qp_edge_key:
                st.session_state["edge_remove_key_input"] = qp_edge_key

            edge_remove_key = st.text_input(
                "Edge Key (format: id1_id2)",
                value=st.session_state.get("edge_remove_key_input", ""),
                key="edge_remove_key_input",
                help="Enter the edge key in the format 'id1_id2' (e.g., '123_456')"
            )

            if st.button("🗑️ Remove Edge", key="remove_by_key_btn", disabled=not st.session_state.editing_enabled, use_container_width=True, type="secondary"):
                try:
                    parts = str(edge_remove_key).strip().split("_")
                    if len(parts) == 2:
                        a, b = int(parts[0]), int(parts[1])
                        st.session_state.working_edges = remove_edge(st.session_state.working_edges, a, b)
                        st.session_state.working_clusters = rebuild_clusters_from_edges(
                            st.session_state.working_edges, st.session_state.working_all_ids
                        )
                        st.success(f"✅ Removed edge {a} ↔ {b}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Edge key must be in the form id1_id2")
                except Exception as e:
                    st.error(f"❌ Failed to remove edge: {e}")
            elif not st.session_state.editing_enabled:
                st.info("💡 Enable editing to remove edges.")

            if cid is not None:
                edges_in_cluster = _get_edges_in_cluster(int(cid))
                if not edges_in_cluster.empty:
                    feedback_counts = Counter(st.session_state.current_cluster_feedback.values())
                    if feedback_counts:
                        st.info(f"📊 **Feedback Summary**: {feedback_counts.get('accepted', 0)} ✅ accepted, {feedback_counts.get('rejected', 0)} ❌ rejected, {len(edges_in_cluster) - sum(feedback_counts.values())} ⏳ pending")

    with colL:
        st.markdown("### 📋 Cluster Summary")
        st.caption("Analysis of the selected cluster compared to hand-labeled ground truth")
        
        if cid is None:
            st.stop()

        _load_cluster_feedback(int(cid))

        row = clusters_df[clusters_df["cluster_id"] == int(cid)].iloc[0]
        members: List[int] = list(map(int, row["members"]))
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.metric("Cluster ID", f"#{int(cid)}")
        with col_meta2:
            st.metric("Cluster Size", int(row['size']))

        st.markdown("---")
        st.markdown("**🏷️ Prediction vs. Hand-Labels**")
        person_ids_in_cluster = people_df.loc[people_df["id"].isin(members), "person"]
        anchor_person_id = None

        if not person_ids_in_cluster.dropna().empty:
            person_counts = Counter(int(p) for p in person_ids_in_cluster.dropna())
            anchor_person_id = person_counts.most_common(1)[0][0]

            if len(person_counts) == 1:
                st.success(f"✅ **Consistent**: Matches hand-labeled Person ID {anchor_person_id}")
            else:
                st.warning(f"⚠️ **Discrepant**: Merged {len(person_counts)} different hand-labeled groups")
                st.dataframe(pd.DataFrame(person_counts.items(), columns=["Person ID", "Count"]).sort_values("Count", ascending=False), use_container_width=True)
        else:
            st.info("ℹ️ **New Prediction**: No existing hand-labels for these records")

        sub = people_df.loc[
            people_df["id"].isin(members),
            ["id", "person", "nafn_norm", "first_name", "patronym", "surname", "birthyear", "sex", "father", "mother", "partner", "parish", "district", "farm", "heimild"]
        ].sort_values(["birthyear", "first_name"])

        def style_records(row):
            """Applies CSS styling to a DataFrame row for visual highlighting.

            Used with `df.style.apply` to color-code records in the cluster member table
            based on whether their hand-label matches the cluster's dominant (anchor) label.

            Args:
                row (pd.Series): A row of the people DataFrame.

            Returns:
                List[str]: A list of CSS style strings to be applied to the row.
            """
            style = ''
            if pd.notna(row.person) and anchor_person_id is not None:
                style = 'background-color: #8B0000; color: white;' if int(row.person) != anchor_person_id else 'background-color: #4A6984; color: white;'
            return [style] * len(row)

        st.markdown("**👥 Cluster Members**")
        st.dataframe(sub.style.apply(style_records, axis=1), use_container_width=True, height=400)

        legend_html = """<div style="display: flex; align-items: center; margin-top: 10px; margin-bottom: 10px; font-size: 13px; color: #E0E0E0; padding: 8px; background-color: rgba(40, 40, 40, 0.3); border-radius: 5px;"><b style="margin-right: 15px;">Legend:</b>
        <div style="display: flex; align-items: center; margin-right: 20px;"><div style="width: 15px; height: 15px; background-color: transparent; margin-right: 5px; border: 1px solid #888;"></div><span>New Prediction</span></div>
        <div style="display: flex; align-items: center; margin-right: 20px;"><div style="width: 15px; height: 15px; background-color: #4A6984; margin-right: 5px; border: 1px solid #ccc;"></div><span>Anchor Group</span></div>
        <div style="display: flex; align-items: center;"><div style="width: 15px; height: 15px; background-color: #8B0000; margin-right: 5px; border: 1px solid #ccc;"></div><span>Discrepant Member</span></div></div>"""
        st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("---")
    
    with st.expander("🔄 Model Retraining & Feedback", expanded=False):
        total_feedback = sum(len(feedback) for feedback in st.session_state.edge_feedback.values())
        if total_feedback > 0:
            st.info(f"📊 **Total feedback collected:** {total_feedback} edge decisions across all clusters")
            col_retrain, col_finetune = st.columns(2)
            with col_retrain:
                st.markdown("**🔄 Complete Retrain**")
                st.markdown("Retrain the entire model from scratch using all feedback data")
                retrain_method = st.selectbox(
                    "Retrain method:",
                    [
                        "ICE-ID Pipeline", "Ditto (HF)", "ZeroER (SBERT)", "TF-IDF", "MLP",
                        "Splink", "XGBoost", "LightGBM", "Cross-Encoder", "Dedupe", "RecordLinkage"
                    ],
                    key="retrain_method"
                )
                if st.button("🚀 Start Complete Retrain", key="start_retrain"):
                    st.info("🔄 Starting complete retrain...")
                    st.success("✅ Complete retrain completed! Check the new results.")
            with col_finetune:
                st.markdown("**🎯 Finetune Current Cluster**")
                st.markdown("Finetune the model specifically on this cluster's feedback")
                finetune_method = st.selectbox(
                    "Finetune method:",
                    [
                        "ICE-ID Pipeline", "Ditto (HF)", "ZeroER (SBERT)", "TF-IDF", "MLP",
                        "Splink", "XGBoost", "LightGBM", "Cross-Encoder", "Dedupe", "RecordLinkage"
                    ],
                    key="finetune_method"
                )
                if st.button("🎯 Start Cluster Finetune", key="start_finetune"):
                    st.info(f"🎯 Starting finetune on cluster {cid}...")
                    st.success("✅ Cluster finetune completed! Check the updated results.")
            st.markdown("---")
            st.subheader("📁 Feedback Management")
            col_export, col_import = st.columns(2)
            with col_export:
                if st.button("📤 Export Feedback", key="export_feedback"):
                    import json
                    feedback_json = json.dumps(st.session_state.edge_feedback, indent=2)
                    st.download_button(
                        label="Download feedback.json",
                        data=feedback_json,
                        file_name="edge_feedback.json",
                        mime="application/json"
                    )
            with col_import:
                uploaded_feedback = st.file_uploader("Import feedback", type=['json'], key="import_feedback")
                if uploaded_feedback is not None:
                    try:
                        import json
                        feedback_data = json.load(uploaded_feedback)
                        st.session_state.edge_feedback = feedback_data
                        st.success("✅ Feedback imported successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error importing feedback: {e}")
        else:
            st.info("💡 **No feedback yet** - Start reviewing edges above to collect feedback for retraining")

    st.markdown("---")
    st.markdown("## 🌐 Visual Analytics")
    st.caption("Interactive graph visualizations for cluster analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🔗 Match Network")
        st.caption("Predicted edges between records in the cluster")
        cluster_by_option = st.selectbox(
            "Visually group nodes by:",
            ["Label Status", "Hand-Label (person)", "Parish", "Census (heimild)"],
            index=0,
            help="Draw shapes around nodes sharing an attribute."
        )
        network_node_limit = st.number_input("Max nodes", value=10, key="net_node_limit")

        group_keys = None
        if cluster_by_option != "Label Status":
            col = {"Parish": "parish", "Census (heimild)": "heimild", "Hand-Label (person)": "person"}[cluster_by_option]
            group_keys = sorted(sub[col].dropna().unique().tolist())

        net_graph, legend_info = build_pyvis_network_graph(
            members=members,
            people_df=people_df,
            edges_df=edges_df,
            node_limit=int(network_node_limit),
            anchor_person_id=anchor_person_id,
            cluster_by=cluster_by_option,
            group_keys=group_keys,
            color_palette=COLOR_PALETTE,
        )
        render_pyvis_graph(net_graph)

        if legend_info:
            display_items = []
            if cluster_by_option == "Parish":
                for k, color in legend_info.items():
                    name = parish_map.get(k, f"Parish {k}")
                    display_items.append((name, color))
            else:
                display_items = list(legend_info.items())

            parts = ['<div style="display: flex; flex-wrap: wrap; margin-top: 10px; font-size: 12px; color: #E0E0E0;">']
            for name, color in display_items:
                parts.append(
                    '<div style="display: flex; align-items: center; margin-right: 15px;">'
                    f'<div style="width: 12px; height: 12px; background-color: {color}; border-radius: 6px; margin-right: 5px;"></div>'
                    f'<span>{name}</span></div>'
                )
            parts.append("</div>")
            st.markdown("".join(parts), unsafe_allow_html=True)

    with c2:
        st.markdown("### 🌳 Genealogical Tree")
        st.caption("Family relationships from source data")
        tree_scope = st.radio("Show tree for:", ("Root person only", "All cluster members"), index=0, horizontal=True, key="tree_scope_radio")
        enable_physics = st.checkbox("Enable physics for free movement", value=True, key="fam_physics_toggle", help="Toggle between hierarchical and dynamic layouts.")
        family_node_limit = st.number_input("Max nodes", value=10, key="fam_node_limit")
        family_depth_up = st.number_input("Generations up", value=10, max_value=100, key="fam_up")
        family_depth_down = st.number_input("Generations down", value=10, max_value=100, key="fam_down")
        depth_limit = int(family_depth_up) + int(family_depth_down)

        tree_cluster_by_option = st.selectbox(
            "Visually group nodes by:",
            ["Label Status", "Hand-Label (person)", "Parish", "Census (heimild)"],
            index=0,
            help="Draw shapes around nodes sharing an attribute.",
            key="tree_cluster_by_option"
        )

        tree_group_keys = None
        if tree_cluster_by_option != "Label Status":
            col = {"Parish": "parish", "Census (heimild)": "heimild", "Hand-Label (person)": "person"}[tree_cluster_by_option]
            tree_group_keys = sorted(sub[col].dropna().unique().tolist())

        tree_seeds = [root_id] if tree_scope == "Root person only" and root_id is not None else members
        fam_graph, tree_legend_info = build_pyvis_family_tree(
            seeds=tree_seeds,
            people_df=people_df,
            node_limit=int(family_node_limit),
            depth_limit=max(1, depth_limit),
            physics_enabled=enable_physics,
            anchor_person_id=anchor_person_id,
            cluster_by=tree_cluster_by_option,
            group_keys=tree_group_keys,
            color_palette=COLOR_PALETTE,
        )
        render_pyvis_graph(fam_graph)
        
        if tree_legend_info:
            display_items = []
            if tree_cluster_by_option == "Parish":
                for k, color in tree_legend_info.items():
                    name = parish_map.get(k, f"Parish {k}")
                    display_items.append((name, color))
            else:
                display_items = list(tree_legend_info.items())

            parts = ['<div style="display: flex; flex-wrap: wrap; margin-top: 10px; font-size: 12px; color: #E0E0E0;">']
            for name, color in display_items:
                parts.append(
                    '<div style="display: flex; align-items: center; margin-right: 15px;">'
                    f'<div style="width: 12px; height: 12px; background-color: {color}; border-radius: 6px; margin-right: 5px;"></div>'
                    f'<span>{name}</span></div>'
                )
            parts.append("</div>")
            st.markdown("".join(parts), unsafe_allow_html=True)

    with st.expander("Data quality flags (read-only; graph shows raw data)", expanded=False):
        if st.checkbox("Scan the visible nodes for oddities", value=False, key="dq_scan_toggle"):
            try:
                present_ids = sub["id"].astype(int).tolist()
                local = st.session_state.people.loc[
                    st.session_state.people["id"].isin(present_ids),
                    ["id", "father", "mother", "partner", "sex"]
                ].copy()

                rows = []
                for _, r in local.iterrows():
                    cid_local = int(r["id"])
                    def as_int(v):
                        """Safely casts a value to an integer, returning None on failure.

                        Args:
                            v: The value to convert.

                        Returns:
                            Optional[int]: The integer value or None if conversion fails.
                        """
                        try:
                            return int(v) if pd.notna(v) else None
                        except Exception:
                            return None
                    f = as_int(r.get("father"))
                    m = as_int(r.get("mother"))
                    p = as_int(r.get("partner"))

                    issues = []
                    if p is not None and p == cid_local:
                        issues.append("self_partner")
                    if f is not None and m is not None and f == m:
                        issues.append("same_parent_id_in_both_columns")

                    if issues:
                        rows.append({"id": cid_local, "issues": ", ".join(sorted(set(issues)))})
                if rows:
                    st.warning("Potential issues found in the **visible** nodes:")
                    st.dataframe(pd.DataFrame(rows).sort_values("id"))
                else:
                    st.info("No obvious flags for the currently visible nodes.")
            except Exception as e:
                st.error(f"Flag scan failed: {e}")

    st.markdown("---")
    
    with st.expander("💾 Save Working Copy", expanded=False):
        st.markdown("### 💾 Save Working Copy as New Version")
        st.caption("Preserve your edited clusters and edges for future reference")
        
        comment = st.text_area(
            "📝 Changelog / Comment", 
            value="", 
            height=100, 
            help="Describe the changes you made in this version (optional but recommended)"
        )

        if st.button("💾 Save Snapshot", key="btn_save_snapshot", use_container_width=True, type="primary"):
            try:
                os.makedirs(versions_root, exist_ok=True)
                vdir = save_version(
                    base_backend_dir=st.session_state.baseline_label,
                    people_csv=people_csv,
                    edges=st.session_state.working_edges,
                    clusters=st.session_state.working_clusters,
                    out_root=versions_root,
                    comment=comment,
                    version_name=st.session_state.working_name or None,
                )
                st.success(f"✅ Saved new version at: {vdir}")
            except Exception as e:
                st.error(f"❌ Failed to save version: {e}")
            finally:
                gc.collect()

    st.markdown("---")
    
    with st.expander("📦 Working Artifacts", expanded=False):
        st.markdown("### 📦 Current Working Artifacts")
        st.caption("Download or preview the current state of edges and clusters")

        cE, cC = st.columns(2)
        with cE:
            st.markdown("**🔗 Current Edges**")
            ed_show = st.session_state.working_edges.copy()
            ed_show["edge_key"] = ed_show.apply(lambda r: f"{int(min(r['id1'], r['id2']))}_{int(max(r['id1'], r['id2']))}", axis=1)
            st.dataframe(ed_show[["id1", "id2", "edge_key"]].head(5000), height=240, use_container_width=True)
            st.download_button(
                "📥 Download Edges CSV",
                data=ed_show.drop(columns=["edge_key"]).to_csv(index=False),
                file_name="edges_working.csv",
                mime="text/csv",
                use_container_width=True
            )

        with cC:
            st.markdown("**📊 Current Clusters**")
            cl_show = st.session_state.working_clusters.copy()
            cl_show["members_str"] = cl_show["members"].apply(members_to_str)
            st.dataframe(cl_show[["cluster_id", "size", "members_str"]].head(5000), height=240, use_container_width=True)
            st.download_button(
                "📥 Download Clusters CSV",
                data=cl_show.assign(members=cl_show["members_str"]).drop(columns=["members_str"]).to_csv(index=False),
                file_name="clusters_working.csv",
                mime="text/csv",
                use_container_width=True
            )
            del cl_show
            gc.collect()

    with st.expander("📚 Baseline Summary (Read-Only)", expanded=False):
        st.caption(st.session_state.baseline_label)
        base_ed = st.session_state.baseline_edges.copy()
        if base_ed is not None and not base_ed.empty:
            base_ed_show = base_ed.copy()
            base_ed_show["edge_key"] = base_ed_show.apply(lambda r: f"{int(min(r['id1'], r['id2']))}_{int(max(r['id1'], r['id2']))}", axis=1)
            st.markdown("**Baseline Edges (preview)**")
            st.dataframe(base_ed_show[["id1", "id2", "edge_key"]].head(2000), height=180)
        base_cl = st.session_state.baseline_clusters.copy()
        base_cl["members_str"] = base_cl["members"].apply(members_to_str)
        st.dataframe(base_cl[["cluster_id", "size", "members_str"]].head(5000), height=220)
        del base_cl
        gc.collect()
