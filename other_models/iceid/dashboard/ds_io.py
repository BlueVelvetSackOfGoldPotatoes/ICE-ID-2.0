# other_models/dashboard/ds_io.py
"""I/O utilities for the dashboard (load people, edges, clusters, versions)
plus cached relationship indexes for fast family lookups.
"""

from __future__ import annotations

import ast
import gc
import json
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from .schemas import PeopleSchema, EdgeSchema, ClusterSchema, schema_signature


# -----------------------------
# Helpers
# -----------------------------
def _file_sig(path: str) -> str:
    """Creates a cache key from a file's path, modification time, and size.

    Args:
        path (str): The path to the file.

    Returns:
        str: A unique signature string for caching, or a 'missing' key if the file doesn't exist.
    """
    try:
        st_ = os.stat(path)
        return f"{path}|{int(st_.st_mtime)}|{st_.st_size}"
    except FileNotFoundError:
        return f"{path}|missing"


def canonicalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures edges are undirected (id1 < id2) and removes duplicates.

    This standardizes edge representation, making lookups and joins consistent.

    Args:
        df (pd.DataFrame): A DataFrame with 'id1' and 'id2' columns.

    Returns:
        pd.DataFrame: A new DataFrame with canonicalized and unique edges.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["id1", "id2"])
    out = df[["id1", "id2"]].copy()
    a = out["id1"].astype("int64", copy=False)
    b = out["id2"].astype("int64", copy=False)
    id1 = np.minimum(a, b)
    id2 = np.maximum(a, b)
    out.loc[:, "id1"] = id1
    out.loc[:, "id2"] = id2
    out = out.drop_duplicates(ignore_index=True)
    return out


def _parse_members_column(col: pd.Series) -> pd.Series:
    """Parses a cluster members column into lists of integers.

    Handles string formats like '1;2;3' and list-like strings '[1, 2, 3]'.

    Args:
        col (pd.Series): The DataFrame column containing cluster members.

    Returns:
        pd.Series: A Series where each element is a list of integer member IDs.
    """
    def parse_one(x):
        if isinstance(x, list):
            return [int(i) for i in x]
        if isinstance(x, (np.ndarray, tuple)):
            return [int(i) for i in list(x)]
        if isinstance(x, str):
            s = x.strip()
            if ";" in s and "[" not in s:
                return [int(i) for i in s.split(";") if i]
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    return [int(i) for i in v]
            except Exception:
                pass
        return []
    return col.apply(parse_one)


def members_to_str(members: List[int]) -> str:
    """Converts a list of member IDs into a semicolon-separated string.

    Args:
        members (List[int]): A list of integer IDs.

    Returns:
        str: A string like "id1;id2;id3".
    """
    return ";".join(str(i) for i in members)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_parish_map(parish_csv: str, _sig: str) -> Dict[int, str]:
    """Loads and caches a mapping from parish ID to parish name.

    Args:
        parish_csv (str): The path to the parish CSV file.
        _sig (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        Dict[int, str]: A dictionary mapping integer parish IDs to their string names.
    """
    if "missing" in _sig:
        return {}
    df = pd.read_csv(parish_csv, usecols=["id", "name"], dtype={"id": "Int64", "name": str})
    return pd.Series(df.name.values, index=df.id).to_dict()

@st.cache_data(show_spinner=False)
def load_people(people_csv: str, _sig: str, schema: Optional[PeopleSchema] = None, schema_sig: str = "none") -> pd.DataFrame:
    """Loads and caches the main people.csv file with necessary type coercion.

    This function selects a subset of columns, converts IDs and years to nullable integers,
    and normalizes key string fields to lowercase.

    Args:
        people_csv (str): Path to the people.csv file.
        _sig (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        pd.DataFrame: The loaded and preprocessed people data.
    """
    if "missing" in _sig:
        raise FileNotFoundError(f"people.csv not found: {people_csv}")

    usecols = [
        "id", "heimild", "nafn_norm", "first_name", "middle_name", "patronym", "surname",
        "birthyear", "sex", "status", "marriagestatus", "person", "partner", "father", "mother",
        "farm", "county", "parish", "district"
    ]
    df = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if schema is not None and schema.rename_map:
        df = df.rename(columns=schema.rename_map)
    for c in usecols:
        if c not in df.columns:
            df[c] = ""
    df = df[usecols].copy()

    # numerics to Int64
    for numc in ["id", "heimild", "birthyear", "farm", "county", "parish", "district", "partner", "father", "mother", "person"]:
        df[numc] = pd.to_numeric(df[numc], errors="coerce").astype("Int64")

    # normalize strings
    for sc in ["nafn_norm", "first_name", "middle_name", "patronym", "surname", "sex", "status", "marriagestatus"]:
        df[sc] = df[sc].astype(str).str.lower()

    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int64", copy=False)
    return df


@st.cache_data(show_spinner=False)
def detect_backends(run_root: str) -> list[str]:
    """Scans a directory to find valid backend subdirectories.

    A valid backend is a subdirectory that contains a 'clusters.csv' file.

    Args:
        run_root (str): The path to the parent directory (e.g., a shard directory).

    Returns:
        list[str]: A sorted list of detected backend names.
    """
    if not os.path.isdir(run_root):
        return []
    subs = []
    for name in os.listdir(run_root):
        p = os.path.join(run_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "clusters.csv")):
            subs.append(name)
    subs.sort()
    return subs


@st.cache_data(show_spinner=False)
def _read_clusters_cached(cl_path: str, _sig: str) -> pd.DataFrame:
    """Reads and caches a clusters.csv file, parsing the 'members' column.

    Args:
        cl_path (str): The path to the clusters.csv file.
        _sig (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        pd.DataFrame: The loaded cluster data with a 'members' column of type list[int].
    """
    clusters = pd.read_csv(cl_path, dtype={"cluster_id": "Int64", "size": "Int64"}, low_memory=False)
    clusters["members"] = _parse_members_column(clusters.get("members", pd.Series([], dtype=str)))
    clusters["members"] = clusters["members"].apply(lambda x: list(map(int, x)))
    return clusters


@st.cache_data(show_spinner=False)
def _read_edges_cached(ed_path: str, _sig: str) -> pd.DataFrame:
    """Reads and caches an edges.csv file, ensuring edges are canonicalized.

    Args:
        ed_path (str): The path to the edges.csv file.
        _sig (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        pd.DataFrame: The loaded and canonicalized edge data.
    """
    if not os.path.exists(ed_path):
        return pd.DataFrame(columns=["id1", "id2"])
    edges = pd.read_csv(ed_path, dtype={"id1": "Int64", "id2": "Int64"}, low_memory=False)
    edges = edges.dropna(subset=["id1", "id2"]).copy()
    edges["id1"] = edges["id1"].astype("int64", copy=False)
    edges["id2"] = edges["id2"].astype("int64", copy=False)
    edges = canonicalize_edges(edges)
    return edges


def load_edges_clusters(backend_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads edges.csv and clusters.csv from a specific backend directory.

    Args:
        backend_dir (str): The path to the directory containing the files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the edges DataFrame and the clusters DataFrame.
    """
    ed_path = os.path.join(backend_dir, "edges.csv")
    cl_path = os.path.join(backend_dir, "clusters.csv")
    if not os.path.exists(cl_path):
        raise FileNotFoundError(f"clusters.csv not found in {backend_dir}")
    clusters = _read_clusters_cached(cl_path, _file_sig(cl_path))
    edges = _read_edges_cached(ed_path, _file_sig(ed_path))
    return edges, clusters


@st.cache_data(show_spinner=False)
def _read_edges_cached_with_schema(ed_path: str, _sig: str, schema: Optional[EdgeSchema] = None) -> pd.DataFrame:
    """Reads an edges file and applies an optional schema for column names.

    Args:
        ed_path (str): The path to the edges file.
        _sig (str): A file signature used for cache invalidation.
        schema (Optional[EdgeSchema]): Optional column mapping for id1/id2.

    Returns:
        pd.DataFrame: The loaded and canonicalized edge data.
    """
    if schema is None:
        return _read_edges_cached(ed_path, _sig)
    if not os.path.exists(ed_path):
        return pd.DataFrame(columns=["id1", "id2"])
    edges = pd.read_csv(ed_path, low_memory=False)
    c1 = schema.id1_col
    c2 = schema.id2_col
    if c1 not in edges.columns or c2 not in edges.columns:
        return pd.DataFrame(columns=["id1", "id2"])
    edges = edges.rename(columns={c1: "id1", c2: "id2"})
    edges = edges.dropna(subset=["id1", "id2"]).copy()
    edges["id1"] = pd.to_numeric(edges["id1"], errors="coerce").astype("Int64")
    edges["id2"] = pd.to_numeric(edges["id2"], errors="coerce").astype("Int64")
    edges = edges.dropna(subset=["id1", "id2"]).copy()
    edges["id1"] = edges["id1"].astype("int64", copy=False)
    edges["id2"] = edges["id2"].astype("int64", copy=False)
    return canonicalize_edges(edges)


@st.cache_data(show_spinner=False)
def _read_clusters_cached_with_schema(cl_path: str, _sig: str, schema: Optional[ClusterSchema] = None) -> pd.DataFrame:
    """Reads a clusters file and applies an optional schema for columns and members.

    Args:
        cl_path (str): The path to the clusters file.
        _sig (str): A file signature used for cache invalidation.
        schema (Optional[ClusterSchema]): Optional column mapping and members separator.

    Returns:
        pd.DataFrame: The loaded cluster data with canonical column names.
    """
    if schema is None:
        return _read_clusters_cached(cl_path, _sig)
    clusters = pd.read_csv(cl_path, low_memory=False)
    needed = [schema.cluster_id_col, schema.size_col, schema.members_col]
    for c in needed:
        if c not in clusters.columns:
            clusters[c] = [] if c == schema.members_col else pd.NA
    clusters = clusters.rename(columns={
        schema.cluster_id_col: "cluster_id",
        schema.size_col: "size",
        schema.members_col: "members",
    })
    if schema.members_sep:
        clusters["members"] = clusters["members"].astype(str).apply(lambda s: [int(x) for x in s.split(schema.members_sep) if x])
    else:
        clusters["members"] = _parse_members_column(clusters.get("members", pd.Series([], dtype=str)))
    clusters["members"] = clusters["members"].apply(lambda x: list(map(int, x)))
    clusters["cluster_id"] = pd.to_numeric(clusters["cluster_id"], errors="coerce").astype("Int64")
    clusters["size"] = pd.to_numeric(clusters["size"], errors="coerce").astype("Int64")
    return clusters


def load_edges_clusters_with_schema(backend_dir: str, edge_schema: Optional[EdgeSchema] = None, cluster_schema: Optional[ClusterSchema] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads edges and clusters with optional schemas for column adaptation.

    Args:
        backend_dir (str): The directory containing edges and clusters files.
        edge_schema (Optional[EdgeSchema]): Optional schema for edges columns.
        cluster_schema (Optional[ClusterSchema]): Optional schema for cluster columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A pair of dataframes for edges and clusters.
    """
    ed_path = os.path.join(backend_dir, "edges.csv")
    cl_path = os.path.join(backend_dir, "clusters.csv")
    if not os.path.exists(cl_path):
        raise FileNotFoundError(f"clusters.csv not found in {backend_dir}")
    edges = _read_edges_cached_with_schema(ed_path, _file_sig(ed_path), edge_schema)
    clusters = _read_clusters_cached_with_schema(cl_path, _file_sig(cl_path), cluster_schema)
    return edges, clusters


def save_version(
    base_backend_dir: str,
    people_csv: str,
    edges: pd.DataFrame,
    clusters: pd.DataFrame,
    out_root: str,
    comment: str = "",
    version_name: str | None = None,
) -> str:
    """Saves a snapshot of edges and clusters to a new versioned directory.

    The new directory will contain edges.csv, clusters.csv, and a meta.json file
    with information about the source data and creation time.

    Args:
        base_backend_dir (str): The original backend directory being versioned.
        people_csv (str): Path to the people.csv file used.
        edges (pd.DataFrame): The DataFrame of edges to save.
        clusters (pd.DataFrame): The DataFrame of clusters to save.
        out_root (str): The parent directory where the new version folder will be created.
        comment (str, optional): A user-provided comment to store in the metadata.
        version_name (str | None, optional): An optional name for the version folder. If None, a timestamp-based name is generated.

    Returns:
        str: The path to the newly created version directory.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = version_name or f"version_{ts}"
    out_dir = os.path.join(out_root, base_name)
    os.makedirs(out_dir, exist_ok=False)

    meta = {
        "created_at": ts,
        "source_backend_dir": os.path.abspath(base_backend_dir),
        "people_csv": os.path.abspath(people_csv),
        "comment": comment,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    e = canonicalize_edges(edges)
    e.to_csv(os.path.join(out_dir, "edges.csv"), index=False)

    c = clusters.copy()
    c = c[["cluster_id", "size", "members"]].copy()
    c["members"] = c["members"].apply(lambda m: ";".join(map(str, m)) if isinstance(m, list) else str(m))
    c.to_csv(os.path.join(out_dir, "clusters.csv"), index=False)

    return out_dir


def load_saved_version(version_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads edges and clusters from a previously saved version directory.

    Args:
        version_dir (str): The path to the version directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the loaded edges and clusters DataFrames.
    """
    ed_path = os.path.join(version_dir, "edges.csv")
    cl_path = os.path.join(version_dir, "clusters.csv")
    if not os.path.exists(cl_path):
        raise FileNotFoundError(f"clusters.csv missing in {version_dir}")
    clusters = _read_clusters_cached(cl_path, _file_sig(cl_path))
    edges = _read_edges_cached(ed_path, _file_sig(ed_path))
    return edges, clusters


def free_memory(*objs) -> None:
    """Explicitly deletes objects and triggers Python's garbage collector.

    This is a utility to help manage memory when dealing with large DataFrames.

    Args:
        *objs: A variable number of objects to delete.
    """
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


# -----------------------------
# Cached relationship indexes
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_children_index(people_df: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int]]:
    """Creates and caches indexes for fast family relationship lookups.

    This function builds maps for father-to-children, mother-to-children, and person-to-partner
    relationships from the main people DataFrame.

    Args:
        people_df (pd.DataFrame): The main people DataFrame.

    Returns:
        Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int]]: A tuple containing:
            - A dictionary mapping father IDs to a list of their children's IDs.
            - A dictionary mapping mother IDs to a list of their children's IDs.
            - A dictionary mapping a person's ID to their partner's ID.
    """
    # Work on copy with numeric types
    df = people_df[["id", "father", "mother", "partner"]].copy()
    for c in ("id", "father", "mother", "partner"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    father_index: Dict[int, List[int]] = {}
    mother_index: Dict[int, List[int]] = {}
    partner_map: Dict[int, int] = {}

    # Father map
    f_df = df.dropna(subset=["father"])[["father", "id"]]
    for f, kid in f_df.itertuples(index=False):
        father_index.setdefault(int(f), []).append(int(kid))

    # Mother map
    m_df = df.dropna(subset=["mother"])[["mother", "id"]]
    for m, kid in m_df.itertuples(index=False):
        mother_index.setdefault(int(m), []).append(int(kid))

    # Partner map (one-way; we’ll add both directions in graph code)
    p_df = df.dropna(subset=["partner"])[["id", "partner"]]
    for a, b in p_df.itertuples(index=False):
        partner_map[int(a)] = int(b)

    return father_index, mother_index, partner_map