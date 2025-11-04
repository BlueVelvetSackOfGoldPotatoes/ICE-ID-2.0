# other_models/dashboard/graphing.py
"""Graph construction utilities with size limits and light memory use.

Functions:
  - build_cluster_network_edges(members, edges_df, node_limit) -> list[tuple[int,int]]
  - build_family_tree_edges(members, people_df, node_limit, depth_limit) -> (edges, present_nodes)
  - build_pyvis_network_graph(...) -> pyvis.network.Network
  - build_pyvis_family_tree(...) -> pyvis.network.Network
  - to_graphviz_digraph(...) -> str   (kept for reference/back-compat)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pyvis.network import Network


def build_cluster_network_edges(
    members: Iterable[int],
    edges_df: pd.DataFrame,
    node_limit: int = 200,
) -> List[Tuple[int, int]]:
    """Filters edges to those between members, pruning to the most connected nodes if a limit is exceeded.

    If the number of unique members is over `node_limit`, it calculates the degree of each
    node within the subgraph and keeps only the edges between the `node_limit` nodes with
    the highest degree.

    Args:
        members (Iterable[int]): The IDs of the nodes in the cluster.
        edges_df (pd.DataFrame): A DataFrame of all predicted edges with 'id1' and 'id2' columns.
        node_limit (int): The maximum number of nodes to include in the graph.

    Returns:
        List[Tuple[int, int]]: A list of edge tuples (id1, id2) for the resulting graph.
    """
    mset = set(int(i) for i in members)
    if edges_df is None or edges_df.empty:
        return []

    sub = edges_df[
        edges_df["id1"].isin(mset) & edges_df["id2"].isin(mset)
    ][["id1", "id2"]]

    if len(mset) <= node_limit:
        return [(int(a), int(b)) for a, b in sub.itertuples(index=False)]

    deg = Counter()
    for a, b in sub.itertuples(index=False):
        deg[int(a)] += 1
        deg[int(b)] += 1

    if not deg:
        return []

    top_nodes = set([n for n, _ in deg.most_common(node_limit)])
    keep_edges = []
    for a, b in sub.itertuples(index=False):
        ia, ib = int(a), int(b)
        if ia in top_nodes and ib in top_nodes:
            keep_edges.append((ia, ib))
    return keep_edges


def build_family_tree_edges(
    members: Iterable[int],
    people_df: pd.DataFrame,
    node_limit: int = 150,
    depth_limit: int = 2,
) -> Tuple[List[Tuple[int, int, str]], List[int]]:
    """Builds a genealogical graph by traversing family links from a set of seed members.

    It performs a breadth-first search starting from `members` to find parents, children,
    and partners up to a specified `depth_limit` and `node_limit`.

    Args:
        members (Iterable[int]): The initial set of person IDs to start the traversal from.
        people_df (pd.DataFrame): The main people DataFrame containing relationship columns ('father', 'mother', 'partner').
        node_limit (int): The maximum number of people to include in the tree.
        depth_limit (int): The maximum number of relationship steps to traverse from the seeds.

    Returns:
        Tuple[List[Tuple[int, int, str]], List[int]]: A tuple containing:
            - A list of directed, labeled edges (source_id, target_id, relationship_type).
            - A list of all person IDs present in the final graph.
    """
    id_index = people_df.set_index("id", drop=False)

    seeds = [int(i) for i in members]
    from collections import deque
    q = deque((s, 0) for s in seeds)
    seen = set()
    present: List[int] = []
    edges: List[Tuple[int, int, str]] = []

    if not seeds:
        return edges, present

    while q and len(present) < node_limit:
        node, d = q.popleft()
        if node in seen:
            continue
        seen.add(node)
        present.append(node)

        if d >= depth_limit or node not in id_index.index:
            continue

        row = id_index.loc[node]

        # Parents -> child (keep labels literally as in the data)
        for rel in ("father", "mother"):
            try:
                pid = int(row[rel]) if pd.notna(row[rel]) else None
            except Exception:
                pid = None
            if pid and pid in id_index.index:
                edges.append((pid, node, rel))
                if pid not in seen and len(present) < node_limit:
                    q.append((pid, d + 1))

        # node -> children (same literal labels)
        for rel in ("father", "mother"):
            try:
                kids = people_df.loc[people_df[rel] == node, "id"].head(50).tolist()
            except Exception:
                kids = []
            for kid in kids:
                kid = int(kid)
                edges.append((node, kid, rel))
                if kid not in seen and len(present) < node_limit:
                    q.append((kid, d + 1))

        # Partner (bidirectional)
        try:
            partner = int(row["partner"]) if pd.notna(row["partner"]) else None
        except Exception:
            partner = None
        if partner and partner in id_index.index:
            edges.append((node, partner, "partner"))
            edges.append((partner, node, "partner"))
            if partner not in seen and len(present) < node_limit:
                q.append((partner, d + 1))

    # Dedup directed edges
    seen_e = set()
    dedup = []
    for a, b, lbl in edges:
        key = (int(a), int(b), str(lbl))
        if key not in seen_e:
            seen_e.add(key)
            dedup.append((int(a), int(b), str(lbl)))
    return dedup, present


def build_pyvis_network_graph(
    members: Iterable[int],
    people_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_limit: int,
    anchor_person_id: Optional[int] = None,
    cluster_by: Optional[str] = None,
    group_keys: Optional[List] = None,
    color_palette: Optional[List[str]] = None,
) -> Tuple[Network, Dict]:
    """Creates an interactive pyvis graph for a cluster, arranging nodes in columns by a chosen attribute.

    Nodes are laid out in vertical bands based on the `cluster_by` attribute (e.g., 'Parish',
    'Census', or 'Label Status'). This helps visualize the composition of a cluster.

    Args:
        members (Iterable[int]): The IDs of the nodes in the cluster.
        people_df (pd.DataFrame): DataFrame with person attributes for labeling and grouping.
        edges_df (pd.DataFrame): DataFrame of predicted edges to draw connections.
        node_limit (int): The maximum number of nodes to display.
        anchor_person_id (Optional[int]): The ground truth ID of the main person in the cluster, used for coloring.
        cluster_by (Optional[str]): The attribute used to group nodes into vertical columns.
        group_keys (Optional[List]): A predefined order for the groups to maintain a consistent layout.
        color_palette (Optional[List[str]]): A list of colors to use for different groups.

    Returns:
        Tuple[Network, Dict]: A tuple containing:
            - The configured pyvis Network object.
            - A dictionary mapping group names to colors for rendering a legend.
    """
    net = Network(height="440px", width="100%", notebook=True, directed=False, cdn_resources="in_line")
    net.set_options("""
    var options = {
      "layout": { "improvedLayout": true, "randomSeed": 42 },
      "physics": { "enabled": false },
      "interaction": { "hover": true, "dragNodes": false }
    }
    """)

    members = [int(i) for i in members]
    net_edges = build_cluster_network_edges(members, edges_df, node_limit=node_limit)

    nodes_in_edges = set()
    for a, b in net_edges:
        nodes_in_edges.add(int(a)); nodes_in_edges.add(int(b))
    if nodes_in_edges:
        nodes_to_render = sorted(nodes_in_edges)
    else:
        k = max(1, min(node_limit, len(members)))
        nodes_to_render = members[:k]

    sub_people = people_df.loc[people_df["id"].isin(nodes_to_render)].copy()
    labels: Dict[int, str] = {}
    for rid, fn, pat, sur, by in sub_people[["id", "first_name", "patronym", "surname", "birthyear"]].itertuples(index=False):
        by_s = str(int(by)) if pd.notna(by) else ""
        line2 = (pat or sur or "").strip()
        labels[int(rid)] = f"{fn} {line2}\n({by_s}) [#{int(rid)}]"

    # Maps
    person_id_map = sub_people.set_index("id")["person"].to_dict()
    parish_map = sub_people.set_index("id")["parish"].to_dict()
    heimild_map = sub_people.set_index("id")["heimild"].to_dict()

    # Colors
    STATUS_COLORS = {"Anchor": "#4A6984", "Discrepant": "#8B0000", "New": "#97c2fc"}
    attr_color_map = {key: color_palette[i % len(color_palette)] for i, key in enumerate(group_keys)} if group_keys and color_palette else {}

    # ----- Compute group value per node (also used to band layout) -----
    node_group_val: Dict[int, object] = {}
    used_groups, used_statuses = set(), set()

    def node_status(nid: int) -> str:
        raw = person_id_map.get(nid, None)
        has_label = (raw is not None) and pd.notna(raw)
        if has_label and (anchor_person_id is not None) and int(raw) == anchor_person_id:
            return "Anchor"
        elif has_label:
            return "Discrepant"
        else:
            return "New"

    for nid in nodes_to_render:
        if cluster_by and cluster_by != "Label Status":
            if cluster_by == "Hand-Label (person)":
                gv = person_id_map.get(nid)
            elif cluster_by == "Parish":
                gv = parish_map.get(nid)
            elif cluster_by == "Census (heimild)":
                gv = heimild_map.get(nid)
            else:
                gv = None
            node_group_val[nid] = gv
            if pd.notna(gv):
                used_groups.add(gv)
        else:
            s = node_status(nid)
            node_group_val[nid] = s
            used_statuses.add(s)

    # ----- Order groups for columns -----
    if cluster_by == "Label Status":
        groups_order = [g for g in ("Anchor", "Discrepant", "New") if g in used_statuses]
        group_to_color = {g: STATUS_COLORS[g] for g in groups_order}
    else:
        groups_order = []
        if used_groups:
            # keep stable order using group_keys if provided, else sort uniqs
            if group_keys:
                groups_order = [g for g in group_keys if g in used_groups]
            else:
                groups_order = sorted(list(used_groups), key=lambda x: (str(x)))
        group_to_color = {g: attr_color_map.get(g, "#97c2fc") for g in groups_order}

    # ----- Compute fixed positions: columns per group -----
    # layout params
    col_gap = 420
    row_gap = 80
    # bucket nodes by group value (None/NA go to "Ungrouped")
    grouped_nodes: Dict[object, List[int]] = {}
    for nid in nodes_to_render:
        gv = node_group_val.get(nid, None)
        if (gv is None) or (not pd.notna(gv)):
            gv = "__UNGROUPED__"
        grouped_nodes.setdefault(gv, []).append(nid)

    # ensure order includes ungrouped (at the end)
    if "__UNGROUPED__" in grouped_nodes and "__UNGROUPED__" not in groups_order:
        groups_order = list(groups_order) + ["__UNGROUPED__"]
        group_to_color["__UNGROUPED__"] = "#97c2fc"

    # map each group to an x column; y spreads within the column
    cols = {g: (i - (len(groups_order)-1)/2) * col_gap for i, g in enumerate(groups_order)}

    # ----- Add nodes with fixed positions and colors -----
    for g in groups_order:
        nodes = grouped_nodes.get(g, [])
        n = len(nodes)
        # center vertically around y=0
        y0 = -((n - 1) / 2.0) * row_gap
        for idx, nid in enumerate(nodes):
            # color:
            if cluster_by == "Label Status":
                color = STATUS_COLORS[node_group_val[nid]]
                font_color = "white" if node_group_val[nid] in ("Anchor", "Discrepant") else "black"
            else:
                color = group_to_color.get(g, "#97c2fc")
                font_color = "white"

            net.add_node(
                nid,
                label=labels.get(nid, f"#{nid}"),
                shape="box",
                color=color,
                font={"color": font_color},
                x=cols[g],
                y=y0 + idx * row_gap,
                physics=False,  # keep fixed in bands
                fixed={"x": True, "y": True},
            )

    # edges
    for a, b in net_edges:
        net.add_edge(a, b)

    # ----- Legend restricted to visible groups -----
    legend_info: Dict = {}
    if cluster_by == "Label Status":
        legend_info = {g: STATUS_COLORS[g] for g in groups_order if g in STATUS_COLORS}
    else:
        legend_info = {g: group_to_color[g] for g in groups_order if g != "__UNGROUPED__"}

    return net, legend_info


def build_pyvis_family_tree(
    seeds: Iterable[int],
    people_df: pd.DataFrame,
    node_limit: int,
    depth_limit: int,
    physics_enabled: bool = False,
    anchor_person_id: Optional[int] = None,
    cluster_by: Optional[str] = None,
    group_keys: Optional[List] = None,
    color_palette: Optional[List[str]] = None,
) -> Tuple[Network, Dict]:
    """Creates an interactive pyvis graph for a genealogical tree.

    The graph can be rendered with a hierarchical layout (default) or with physics enabled
    for a force-directed layout. Nodes can be colored and grouped by various attributes
    similar to the match network visualization.

    Args:
        seeds (Iterable[int]): The initial set of person IDs to build the tree around.
        people_df (pd.DataFrame): The main people DataFrame with family information.
        node_limit (int): The maximum number of nodes in the graph.
        depth_limit (int): The maximum traversal depth for finding relatives.
        physics_enabled (bool): If True, use a force-directed layout instead of hierarchical.
        anchor_person_id (Optional[int]): The ground truth ID of the anchor person for coloring nodes.
        cluster_by (Optional[str]): The attribute used to group nodes (e.g., 'Label Status', 'Parish', 'Census (heimild)').
        group_keys (Optional[List]): A predefined order for the groups to maintain a consistent layout.
        color_palette (Optional[List[str]]): A list of colors to use for different groups.

    Returns:
        Tuple[Network, Dict]: A tuple containing:
            - The configured pyvis Network object ready to be displayed.
            - A dictionary mapping group names to colors for rendering a legend.
    """
    net = Network(height="400px", width="100%", notebook=True, directed=True, cdn_resources="in_line")
    
    if physics_enabled:
        net.set_options("""
        var options = { "physics": { "enabled": true, "solver": "barnesHut" } }
        """)
    else:
        net.set_options("""
        var options = {
          "layout": { "hierarchical": { "enabled": true, "direction": "UD", "sortMethod": "directed" } },
          "physics": { "enabled": false }
        }
        """)
    
    seed_list = [int(s) for s in seeds]
    fam_edges_raw, present = build_family_tree_edges(
        members=seed_list, people_df=people_df, node_limit=node_limit, depth_limit=depth_limit
    )

    present_set = set(present)
    for seed in seed_list:
        if seed not in present_set:
            present_set.add(seed)

    sub_people = people_df.loc[people_df["id"].isin(present_set)]
    labels: Dict[int, str] = {}
    for rid, fn, pat, sur, by in sub_people[["id", "first_name", "patronym", "surname", "birthyear"]].itertuples(index=False):
        by_s = str(int(by)) if pd.notna(by) else ""
        line2 = (pat or sur or "").strip()
        labels[int(rid)] = f"{fn} {line2}\n({by_s}) [#{int(rid)}]"
    
    # Maps for grouping
    person_id_map = sub_people.set_index('id')['person'].to_dict()
    parish_map = sub_people.set_index('id')['parish'].to_dict()
    heimild_map = sub_people.set_index('id')['heimild'].to_dict()

    # Colors
    STATUS_COLORS = {"Anchor": "#4A6984", "Discrepant": "#8B0000", "New": "#97c2fc"}
    attr_color_map = {key: color_palette[i % len(color_palette)] for i, key in enumerate(group_keys)} if group_keys and color_palette else {}

    # Compute group value per node
    node_group_val: Dict[int, object] = {}
    used_groups, used_statuses = set(), set()

    def node_status(nid: int) -> str:
        raw = person_id_map.get(nid, None)
        has_label = (raw is not None) and pd.notna(raw)
        if has_label and (anchor_person_id is not None) and int(raw) == anchor_person_id:
            return "Anchor"
        elif has_label:
            return "Discrepant"
        else:
            return "New"

    for nid in present_set:
        if cluster_by and cluster_by != "Label Status":
            if cluster_by == "Hand-Label (person)":
                gv = person_id_map.get(nid)
            elif cluster_by == "Parish":
                gv = parish_map.get(nid)
            elif cluster_by == "Census (heimild)":
                gv = heimild_map.get(nid)
            else:
                gv = None
            node_group_val[nid] = gv
            if pd.notna(gv):
                used_groups.add(gv)
        else:
            s = node_status(nid)
            node_group_val[nid] = s
            used_statuses.add(s)

    # Order groups for consistent coloring
    if cluster_by == "Label Status":
        groups_order = [g for g in ("Anchor", "Discrepant", "New") if g in used_statuses]
        group_to_color = {g: STATUS_COLORS[g] for g in groups_order}
    else:
        groups_order = []
        if used_groups:
            if group_keys:
                groups_order = [g for g in group_keys if g in used_groups]
            else:
                groups_order = sorted(list(used_groups), key=lambda x: (str(x)))
        group_to_color = {g: attr_color_map.get(g, "#97c2fc") for g in groups_order}

    # Add nodes with appropriate colors
    for nid in sorted(present_set):
        if cluster_by == "Label Status":
            color = STATUS_COLORS[node_group_val[nid]]
            font_color = "white" if node_group_val[nid] in ("Anchor", "Discrepant") else "black"
        else:
            gv = node_group_val.get(nid, None)
            if (gv is None) or (not pd.notna(gv)):
                gv = "__UNGROUPED__"
            color = group_to_color.get(gv, "#97c2fc")
            font_color = "white"
        
        net.add_node(nid, label=labels.get(nid, f"#{nid}"), shape="box", color=color, font={"color": font_color})
    
    # Deduplicate ALL edges (directed): avoid double-thick lines
    edge_seen = set()  # (a,b,lbl)
    color_map = {"father": "#66c2a5", "mother": "#fc8d62", "partner": "#8da0cb"}
    for a, b, lbl in fam_edges_raw:
        if a not in present_set or b not in present_set:
            continue
        key = (int(a), int(b), str(lbl))
        if key in edge_seen:
            continue
        edge_seen.add(key)
        net.add_edge(
            a, b,
            label=lbl,
            color=color_map.get(lbl, "#cccccc"),
            dashes=(lbl == "partner"),
            arrows="to" if lbl in ("father", "mother") else ""
        )
    
    # Generate legend information
    legend_info: Dict = {}
    if cluster_by == "Label Status":
        legend_info = {g: STATUS_COLORS[g] for g in groups_order if g in STATUS_COLORS}
    else:
        legend_info = {g: group_to_color[g] for g in groups_order if g != "__UNGROUPED__"}
        
    return net, legend_info


def to_graphviz_digraph(
    edges: List[Tuple[int, int, str | None]],
    label_map: Dict[int, str] | None = None,
    directed: bool = False,
    rankdir: str = "LR",
) -> str:
    """Return a Graphviz DOT string (kept for back-compat)."""
    gtype = "digraph" if directed else "graph"
    conn = "->" if directed else "--"
    parts = [f'{gtype} G {{ rankdir="{rankdir}"; node [shape=box];']

    if label_map:
        for nid, lbl in label_map.items():
            safe = lbl.replace('"', r"\"")
            parts.append(f'  "{nid}" [label="{safe}"];')

    for e in edges:
        if len(e) == 3:
            a, b, lbl = e
        else:
            a, b = e
            lbl = None
        if lbl:
            parts.append(f'  "{a}" {conn} "{b}" [label="{lbl}"];')
        else:
            parts.append(f'  "{a}" {conn} "{b}";')

    parts.append("}")
    return "\n".join(parts)