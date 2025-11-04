# dashboard/edits.py
"""Edge editing and reclustering utilities.

Functions:
    canonicalize_edges(edges_df: pd.DataFrame) -> pd.DataFrame
    add_edge(edges_df: pd.DataFrame, a: int, b: int) -> pd.DataFrame
    remove_edge(edges_df: pd.DataFrame, a: int, b: int) -> pd.DataFrame
    rebuild_clusters_from_edges(edges_df: pd.DataFrame, all_ids: list[int]) -> pd.DataFrame
"""

from __future__ import annotations

from typing import List, Tuple
from collections import defaultdict

import pandas as pd

# -----------------------------
# Union-Find (Disjoint Set) for clustering
# -----------------------------
class DSU:
    """A Disjoint Set Union (DSU) or Union-Find data structure.
    
    Used to efficiently track connected components in a graph. Implements
    union by rank and path compression for optimal performance.
    """
    def __init__(self):
        """Initializes the DSU with empty parent and rank mappings."""

        self.parent = {}
        self.rank = {}

    def find(self, x):
        """Finds the representative (root) of the set containing element x.

        Args:
            x: The element to find.

        Returns:
            The representative of the set containing x.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        """Merges the sets containing elements a and b.

        Args:
            a: An element in the first set.
            b: An element in the second set.
        """
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def edges_to_clusters(all_ids: list[int], edges: list[tuple[int,int]]):
    """Computes clusters (connected components) from a list of edges.

    Args:
        all_ids (list[int]): The universe of all possible node IDs. Nodes not in
            an edge will form singleton clusters.
        edges (list[tuple[int,int]]): A list of tuples, where each tuple represents an edge.

    Returns:
        Tuple[dict, dict]: A tuple containing:
            - A dictionary mapping a stable cluster ID to a sorted list of its member IDs.
            - A dictionary mapping each member ID to its cluster ID.
    """
    dsu = DSU()
    for a, b in edges:
        dsu.union(int(a), int(b))
    for rid in all_ids:
        dsu.find(int(rid))
    comp_members = defaultdict(list)
    for rid in all_ids:
        comp_members[dsu.find(int(rid))].append(int(rid))
    # Stable cluster ids 0..K-1
    clusters = {}
    id_to_cluster = {}
    for i, (root, members) in enumerate(sorted(comp_members.items(), key=lambda kv: ( -len(kv[1]), min(kv[1]) ))):
        clusters[i] = sorted(members)
        for m in members:
            id_to_cluster[m] = i
    return clusters, id_to_cluster


def canonicalize_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated, canonicalized edges dataframe.

    Ensures id1 < id2 for each row and drops duplicate rows.

    Args:
        edges_df: DataFrame with columns id1, id2.

    Returns:
        New dataframe canonicalized and deduped.
    """
    if edges_df is None or edges_df.empty:
        return pd.DataFrame(columns=["id1", "id2"], dtype=int)
    df = edges_df.copy()
    df[["id1", "id2"]] = df[["id1", "id2"]].astype(int)
    # enforce ordering
    a = df["id1"].where(df["id1"] <= df["id2"], df["id2"])
    b = df["id2"].where(df["id1"] <= df["id2"], df["id1"])
    df["id1"], df["id2"] = a, b
    df = df.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)
    return df


def add_edge(edges_df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    """Add a single undirected edge (canonicalized) to the edges dataframe.

    Args:
        edges_df: Existing edges dataframe.
        a: Record id.
        b: Record id.

    Returns:
        Updated edges dataframe.
    """
    if a == b:
        return canonicalize_edges(edges_df)
    row = {"id1": int(min(a, b)), "id2": int(max(a, b))}
    if edges_df is None or edges_df.empty:
        return pd.DataFrame([row])
    out = pd.concat([edges_df, pd.DataFrame([row])], ignore_index=True)
    return canonicalize_edges(out)


def remove_edge(edges_df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    """Remove a single undirected edge if present.

    Args:
        edges_df: Existing edges dataframe.
        a: Record id.
        b: Record id.

    Returns:
        Updated edges dataframe.
    """
    if edges_df is None or edges_df.empty:
        return edges_df
    lo, hi = int(min(a, b)), int(max(a, b))
    df = canonicalize_edges(edges_df)
    mask = ~((df["id1"] == lo) & (df["id2"] == hi))
    return df[mask].reset_index(drop=True)


def rebuild_clusters_from_edges(edges_df: pd.DataFrame, all_ids: list[int]) -> pd.DataFrame:
    """Recompute cluster assignments from edges.

    Args:
        edges_df: Canonicalized edges dataframe.
        all_ids: Universe of ids to include (isolated nodes become singletons).

    Returns:
        clusters_df with columns: cluster_id, size, members (semicolon-delimited string).
    """
    edges_list = []
    if edges_df is not None and not edges_df.empty:
        edges_list = [(int(a), int(b)) for a, b in edges_df[["id1", "id2"]].itertuples(index=False)]
    clusters, _ = edges_to_clusters(list(map(int, all_ids)), edges_list)
    rows = []
    for cid, members in clusters.items():
        members_str = ";".join(str(m) for m in members)
        rows.append({"cluster_id": int(cid), "size": int(len(members)), "members": members_str})
    return pd.DataFrame(rows)
