"""Core operator-algebra API for SFT.

This module is deliberately about the mathematical core, not applied packages:
task intent -> invariant -> genus -> representation -> OperatorFamily -> laws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


CORE_BOUNDARIES = {
    "theory-core": (
        "Operator families, spectral derivatives, algebra laws, task invariants, "
        "cost models, and verification claims."
    ),
    "library-api": (
        "Stable, tested user-facing classes/functions built directly on theory-core."
    ),
    "showcase-demo": (
        "Reproducible demos that illustrate power without becoming core API."
    ),
    "research-lab": (
        "Promising ideas that need more evidence before promotion."
    ),
    "applied-package": (
        "Domain packages such as text graphs, spectral NLP, and LLM embeddings."
    ),
}


@dataclass(frozen=True)
class OperatorInvariant:
    """What the operator construction must preserve for the task to be solved."""

    name: str
    description: str
    evidence: str = "heuristic"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "evidence": self.evidence}


@dataclass(frozen=True)
class OperatorCost:
    """Performance and materialization estimate for an operator object."""

    N: int
    M: int
    basis_kind: str = "unknown"
    materialized_elements: int = 0
    memory_mb: float = 0.0
    eigensolve: str = "dense"
    materializes_basis: bool = False
    warnings: tuple[str, ...] = ()

    @classmethod
    def estimate(cls, obj: Any, *, operation: str | None = None) -> "OperatorCost":
        if hasattr(obj, "N") and hasattr(obj, "M"):
            N = int(obj.N)
            M = int(obj.M)
            basis_kind = str(getattr(obj, "basis_kind", "dense"))
            backend = getattr(obj, "_basis_backend", None)
            materialized = int(getattr(backend, "materialized_elements", M * N * N))
            eigensolve = "sparse" if basis_kind in {"sparse", "edge_laplacian"} or N >= 512 else "dense"
            memory_mb = materialized * 8.0 / (1024.0 ** 2)
            warnings = []
            materializes_basis = basis_kind != "dense" and operation in {"compose", "tensor"}
            if materializes_basis:
                warnings.append(f"{operation} may materialize structured basis")
            if operation == "tensor" and N > 200:
                warnings.append("tensor_sum can grow as N_a*N_b; prefer lazy/sparse product")
            return cls(
                N=N,
                M=M,
                basis_kind=basis_kind,
                materialized_elements=materialized,
                memory_mb=float(memory_mb),
                eigensolve=eigensolve,
                materializes_basis=materializes_basis,
                warnings=tuple(warnings),
            )
        arr = np.asarray(obj)
        N = int(arr.shape[0]) if arr.ndim else int(arr.size)
        return cls(N=N, M=0, materialized_elements=int(arr.size), memory_mb=arr.size * arr.itemsize / (1024.0 ** 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "N": self.N,
            "M": self.M,
            "basis_kind": self.basis_kind,
            "materialized_elements": self.materialized_elements,
            "memory_mb": self.memory_mb,
            "eigensolve": self.eigensolve,
            "materializes_basis": self.materializes_basis,
            "warnings": list(self.warnings),
        }


class CostModel:
    """Compatibility facade for estimating operator costs."""

    @staticmethod
    def estimate(obj: Any, *, operation: str | None = None) -> OperatorCost:
        return OperatorCost.estimate(obj, operation=operation)


@dataclass(frozen=True)
class OperatorLaw:
    """A named algebra law with a deterministic verifier."""

    claim_id: str
    name: str
    statement: str
    verifier: Callable[[], bool]
    negative_control: str = ""

    def verify(self) -> dict[str, Any]:
        ok = bool(self.verifier())
        return {
            "claim_id": self.claim_id,
            "name": self.name,
            "status": "PASS" if ok else "FAIL",
            "statement": self.statement,
            "negative_control": self.negative_control,
        }


@dataclass(frozen=True)
class LawReport:
    """Verification report for the laws attached to an operator/family."""

    results: tuple[dict[str, Any], ...]

    @property
    def status(self) -> str:
        return "PASS" if all(row["status"] == "PASS" for row in self.results) else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "results": list(self.results)}


@dataclass(frozen=True)
class LawSet:
    """Small fluent wrapper returned by ``family.laws()``."""

    family: Any

    def verify(self) -> LawReport:
        from . import verify

        core = verify.run_algebra_law_suite()
        return LawReport(tuple(core.values()))


@dataclass(frozen=True)
class OperatorSpec:
    """Task-level specification before constructing an OperatorFamily."""

    task_name: str
    data_shape: tuple[int, ...]
    genus: Any
    invariant: OperatorInvariant
    basis_type: str
    layer: str = "theory-core"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task_name: str, data: Any) -> "OperatorSpec":
        from .tasks import classify_task

        arr = np.asarray(data)
        genus = classify_task(task_name)
        invariant = infer_invariant(task_name, genus)
        basis_type = infer_basis_type(genus, arr)
        return cls(
            task_name=str(task_name),
            data_shape=tuple(arr.shape),
            genus=genus,
            invariant=invariant,
            basis_type=basis_type,
            metadata={"boundary": "core" if genus.name in {"PARAMETRIC", "CONTROL", "FLOW"} else "library-api"},
        )

    def on(self, data: Any) -> "OperatorSpec":
        return OperatorSpec.from_task(self.task_name, data)

    def plan(self):
        from .constructor import OperatorBlueprint

        dummy = np.zeros(self.data_shape or (1,))
        return OperatorBlueprint.from_task(self.task_name, dummy)

    def to_family(self, data: Any | None = None):
        from .constructor import construct

        if data is None:
            data = np.zeros(self.data_shape or (1,))
        return construct(self.plan(), np.asarray(data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "data_shape": self.data_shape,
            "genus": self.genus.name if hasattr(self.genus, "name") else str(self.genus),
            "invariant": self.invariant.to_dict(),
            "basis_type": self.basis_type,
            "layer": self.layer,
            "metadata": dict(self.metadata),
        }


class TaskBuilder:
    """Deferred task builder for ``sft.task('...').on(data)``."""

    def __init__(self, task_name: str):
        self.task_name = str(task_name)

    def on(self, data: Any) -> OperatorSpec:
        return OperatorSpec.from_task(self.task_name, data)


def task(task_name: str) -> TaskBuilder:
    return TaskBuilder(task_name)


def infer_invariant(task_name: str, genus: Any) -> OperatorInvariant:
    name = genus.name if hasattr(genus, "name") else str(genus)
    task_l = task_name.lower()
    if name == "MONO":
        return OperatorInvariant("order_cdf", "Preserve order, ranks, and empirical measure.", "verified")
    if name == "QUAD":
        return OperatorInvariant("frequency_band_energy", "Preserve energy in the selected basis band.", "verified")
    if name == "GRAPH":
        return OperatorInvariant("connectivity_laplacian", "Preserve graph connectivity/diffusion structure.", "verified")
    if name == "CONTROL":
        return OperatorInvariant("target_spectrum_residual", "Reduce residual to a target spectrum.", "verified")
    if name == "FLOW":
        return OperatorInvariant("spectral_monodromy", "Preserve spectral path/topology information.", "verified")
    if "text" in task_l or "semantic" in task_l:
        return OperatorInvariant("typed_relation_structure", "Applied semantic relation structure.", "needs-repro")
    return OperatorInvariant("spectral_response", "Preserve local spectral response under parameter changes.", "verified")


def infer_basis_type(genus: Any, data: np.ndarray) -> str:
    name = genus.name if hasattr(genus, "name") else str(genus)
    if name == "GRAPH" and data.ndim == 2 and data.shape[0] == data.shape[1]:
        return "edge_laplacian"
    if name in {"QUAD", "COMPRESS"}:
        return "toeplitz"
    return "diagonal"
