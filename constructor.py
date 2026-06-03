"""
sft.constructor — Operator synthesis: task → genus → blueprint → OperatorFamily.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Full pipeline from natural-language task description to             ║
║  OperatorFamily.  Single-call: from_task("sort", data).             ║
║  Lower-level: plan_operator → blueprint, construct → OperatorFamily. ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── sft.tasks (classify_task, OperatorGenus)                      ║
║  ├── sft.core.OperatorFamily                                       ║
║  └── sft.families (graph_laplacian, toeplitz)                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
import warnings
from dataclasses import dataclass
from .core import OperatorFamily, coordinate_diagonal_basis
from .tasks import OperatorGenus, classify_task


@dataclass(frozen=True)
class OperatorBlueprint:
    genus: OperatorGenus
    N: int
    M: int
    basis_type: str
    description: str

    @classmethod
    def from_task(cls, task_name: str, data: np.ndarray) -> "OperatorBlueprint":
        return cls.from_dict(plan_operator(task_name, data))

    @classmethod
    def from_dict(cls, blueprint: dict) -> "OperatorBlueprint":
        return cls(
            genus=blueprint["genus"],
            N=int(blueprint["N"]),
            M=int(blueprint["M"]),
            basis_type=str(blueprint["basis_type"]),
            description=str(blueprint.get("description", "")),
        )

    def to_dict(self) -> dict:
        return {
            "genus": self.genus,
            "N": self.N,
            "M": self.M,
            "basis_type": self.basis_type,
            "description": self.description,
        }

    def build(self, data: np.ndarray) -> OperatorFamily:
        return construct(self, data)


def plan_operator(task_name: str, data: np.ndarray) -> dict:
    """Task → blueprint dict {genus, N, M, basis_type, description}."""
    genus = classify_task(task_name)
    N = data.shape[0] if data.ndim == 1 else max(data.shape[:2])
    bp = {"genus": genus, "N": N, "M": min(N, 32), "basis_type": "diagonal",
          "description": f"{genus.name} operator for: {task_name}"}
    if genus == OperatorGenus.GRAPH and data.ndim == 2 and data.shape[0] == data.shape[1]:
        bp["basis_type"] = "edge_laplacian"
        row, _ = np.triu(np.abs(data) > 1e-10, 1).nonzero()
        bp["M"] = min(len(row), 200)
    elif genus in (OperatorGenus.COMPRESS, OperatorGenus.QUAD):
        bp["basis_type"] = "toeplitz"; bp["M"] = min(N - 1, 16)
    return bp


def construct(blueprint: dict | OperatorBlueprint, data: np.ndarray) -> OperatorFamily:
    """Blueprint → OperatorFamily."""
    if isinstance(blueprint, OperatorBlueprint):
        blueprint = blueprint.to_dict()
    N, M, bt = blueprint["N"], blueprint["M"], blueprint["basis_type"]
    d = np.asarray(data, np.float64)
    if bt == "diagonal":
        return OperatorFamily(np.eye(N), coordinate_diagonal_basis(N, M))
    elif bt == "edge_laplacian":
        from .families import graph_laplacian; return graph_laplacian(np.abs(d) > 1e-10)
    elif bt == "toeplitz":
        from .families import toeplitz; return toeplitz(N, diagonals=M)
    warnings.warn(
        f"Unknown basis_type={bt!r}; falling back to deterministic random symmetric basis.",
        UserWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(0)
    basis = [(rng.standard_normal((N, N)) + rng.standard_normal((N, N)).T) / 2 for _ in range(M)]
    return OperatorFamily(np.eye(N), basis)


def synthesize(task_name: str, data: np.ndarray) -> OperatorFamily:
    """plan → construct pipeline."""
    return construct(plan_operator(task_name, data), data)


def from_task(task_name: str, data: np.ndarray) -> OperatorFamily:
    """Single call: NL problem → OperatorFamily.  Alias for synthesize."""
    return synthesize(task_name, data)
