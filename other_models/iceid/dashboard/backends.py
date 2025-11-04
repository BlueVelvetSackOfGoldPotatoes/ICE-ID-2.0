from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping


@dataclass(frozen=True)
class BackendSpec:
    """Specification for a train-time backend plugin.

    Args:
        name (str): Backend name to appear in --backends and output subfolder.
        cli_args (Mapping[str, str]): Key-value pairs converted to CLI flags.

    Returns:
        BackendSpec: A frozen plugin description.
    """
    name: str
    cli_args: Mapping[str, str]


_REGISTRY: Dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> None:
    """Registers a backend specification by name.

    Args:
        spec (BackendSpec): The specification to register.

    Returns:
        None: The registry is updated in place.
    """
    _REGISTRY[spec.name] = spec


def build_extra_args(selected: Mapping[str, Mapping[str, str]] | None) -> List[str]:
    """Builds additional CLI arguments for backends.

    Args:
        selected (Mapping[str, Mapping[str, str]] | None): Optional mapping of backend
            name to key-value overrides to merge with the registered defaults.

    Returns:
        List[str]: A flat list of CLI arguments suitable for extending build_command.
    """
    if not selected:
        return []
    args: List[str] = []
    for name, overrides in selected.items():
        base = dict(_REGISTRY.get(name, BackendSpec(name=name, cli_args={})).cli_args)
        base.update(overrides or {})
        for k, v in base.items():
            flag = f"--{name}_{k}"
            args.append(flag)
            args.append(str(v))
    return args


