from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PeopleSchema:
    """Schema for mapping arbitrary people CSV headers to canonical names.

    The mapping should rename input columns to the canonical names used by the
    application. Only the fields present in the mapping will be renamed; all
    other columns are left unchanged.

    Args:
        rename_map (Dict[str, str]): Mapping from input column name to canonical name.

    Returns:
        PeopleSchema: A frozen schema object used to guide column renaming.
    """
    rename_map: Dict[str, str]


@dataclass(frozen=True)
class LabelsSchema:
    """Schema describing the labels CSV column names.

    Args:
        id_col (str): The record identifier column name in the labels CSV.
        cluster_col (str): The column name holding the cluster or entity ID.

    Returns:
        LabelsSchema: A frozen schema object describing label columns.
    """
    id_col: str
    cluster_col: str


@dataclass(frozen=True)
class EdgeSchema:
    """Schema describing edge file column names.

    Args:
        id1_col (str): The first endpoint column name.
        id2_col (str): The second endpoint column name.

    Returns:
        EdgeSchema: A frozen schema object describing edge columns.
    """
    id1_col: str
    id2_col: str


@dataclass(frozen=True)
class ClusterSchema:
    """Schema describing cluster file column names and membership format.

    Args:
        cluster_id_col (str): The cluster identifier column name.
        size_col (str): The cluster size column name.
        members_col (str): The cluster members column name.
        members_sep (Optional[str]): Optional separator used to split members when
            stored as a flat string (for example ";").

    Returns:
        ClusterSchema: A frozen schema object describing cluster columns.
    """
    cluster_id_col: str
    size_col: str
    members_col: str
    members_sep: Optional[str] = None


def schema_signature(obj: object) -> str:
    """Builds a deterministic signature string for cache keys.

    Args:
        obj (object): A schema object or None.

    Returns:
        str: A stable signature string reflecting the schema configuration.
    """
    if obj is None:
        return "none"
    return repr(obj)


