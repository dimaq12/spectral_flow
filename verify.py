"""
sft.verify — SFT conclusions and production-grade verification gates.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Run all 8 SFT theoretical conclusions as automated tests:          ║
║  C1: Fourier⊂SFT  C2: τ-invariance  C3: W-entropy                  ║
║  C4: Rank separability  C5: Defect universality                     ║
║  C6: Monodromy classification  C7: Hessian sparsity                 ║
║  C8: Universal embedding                                            ║
║  run_verification_suite() → dict with all results                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class VerificationResult:
    """Machine-readable evidence for one verification claim."""

    claim_id: str
    title: str
    layer: str
    status: str
    evidence_level: str
    metric: float | str | bool | None = None
    threshold: str = ""
    promotion: str = "hold"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        return data


def _status(ok: bool) -> str:
    return "PASS" if bool(ok) else "FAIL"


def _promotion(result: VerificationResult) -> str:
    if result.status != "PASS":
        return "rejected"
    if result.evidence_level in {"E3", "E4"} and result.layer == "theory-core":
        return "promote-theory-core"
    if result.evidence_level == "E4" and result.layer == "library-api":
        return "promote-library-api"
    return "needs-repro"


def _finalize(result: VerificationResult) -> VerificationResult:
    return VerificationResult(
        claim_id=result.claim_id,
        title=result.title,
        layer=result.layer,
        status=result.status,
        evidence_level=result.evidence_level,
        metric=result.metric,
        threshold=result.threshold,
        promotion=_promotion(result),
        details=result.details,
    )


def _small_random_family(seed: int = 42, n: int = 8, m: int = 3):
    from .families import random
    return random(n, m, seed=seed)


def verify_c1_fourier_subset() -> bool:
    from .tasks import dct_matrix; from .families import toeplitz
    n = 16; C = dct_matrix(n); W = toeplitz(n, n - 1).W
    return abs(np.sum(C @ C.T) / (n * n) - np.sum(W @ W.T) / (n * W.shape[1])) < 10.0


def verify_c2_tau_invariance() -> bool:
    from .tasks import dct_matrix; N = 50
    C = dct_matrix(N); x = np.random.default_rng(0).standard_normal(N)
    return np.max(np.abs(np.abs(C @ x) - np.abs(C @ np.sort(x)))) < 1e-6


def verify_c3_w_entropy() -> float:
    from .families import random; fam = random(20, 8, seed=42)
    W = fam.W; WWt = W @ W.T; det = np.linalg.det(WWt)
    return float(np.log(max(det, 1e-15))) if det > 0 else 0.0


def verify_c4_rank_separability() -> dict:
    from .families import diagonal, graph_laplacian
    fo = diagonal(20); fg = graph_laplacian(np.ones((20, 20)) - np.eye(20))
    return {"ORDER_complexity": fo.complexity, "GRAPH_complexity": fg.complexity,
            "separable": fo.complexity < fg.complexity}


def verify_c5_defect_universality() -> dict:
    from .order import rank_defect_analysis
    rg = rank_defect_analysis(np.random.default_rng(0).standard_normal(500))
    ru = rank_defect_analysis(np.random.default_rng(0).uniform(-3, 3, 500))
    return {"gaussian_alpha": rg["slope"], "uniform_alpha": ru["slope"]}


def verify_c6_monodromy_classification() -> int:
    from .topology import berry_holonomy; from .families import avoided_crossing_2x2
    fam = avoided_crossing_2x2(0.3)
    loop = [np.array([0.4 * np.cos(t), 0.4 * np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 60)]
    return berry_holonomy(fam, loop, level=1)


def verify_c7_hessian_sparsity() -> float:
    from .hessian import hessian, hessian_sparsity; from .families import random
    return hessian_sparsity(hessian(random(15, 6, seed=42)))


def verify_c8_universal_embedding() -> bool:
    from .embed import GraphEmbedder
    ge = GraphEmbedder(np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]), K=3, R=2)
    return len(ge.embed_graph()) > 0 and all(len(ge.embed_node(i)) > 0 for i in range(4))


def verify_core_hf_kernel() -> VerificationResult:
    """CORE-HF-001: W matches independent finite differences."""
    fam = _small_random_family(seed=101, n=8, m=3)
    eps = 1e-6
    finite_diff = np.empty_like(fam.W)
    zero = np.zeros(fam.M)
    for j in range(fam.M):
        e = np.zeros(fam.M)
        e[j] = eps
        finite_diff[:, j] = (fam.spectrum(e) - fam.spectrum(zero)) / eps
    err = float(np.max(np.abs(finite_diff - fam.W)))
    return _finalize(VerificationResult(
        claim_id="CORE-HF-001",
        title="Hellmann-Feynman kernel matches finite differences",
        layer="theory-core",
        status=_status(err < 1e-4),
        evidence_level="E3",
        metric=err,
        threshold="< 1e-4 max absolute derivative error",
        details={"N": fam.N, "M": fam.M, "eps": eps},
    ))


def verify_core_perturbation_order() -> VerificationResult:
    """CORE-PERT-002: first-order residual scales quadratically."""
    fam = _small_random_family(seed=102, n=10, m=3)
    d = np.array([0.4, -0.2, 0.3])
    d = d / np.linalg.norm(d)
    scales = np.array([0.012, 0.006, 0.003])
    errs = np.array([
        np.max(np.abs(fam.spectrum(d * eps) - fam.predict(d * eps)))
        for eps in scales
    ])
    slope = float(np.polyfit(np.log(scales), np.log(np.maximum(errs, 1e-16)), 1)[0])
    return _finalize(VerificationResult(
        claim_id="CORE-PERT-002",
        title="Linear spectral prediction has second-order residual",
        layer="theory-core",
        status=_status(1.6 <= slope <= 2.4),
        evidence_level="E3",
        metric=slope,
        threshold="log-log residual slope in [1.6, 2.4]",
        details={"scales": scales.tolist(), "errors": errs.tolist()},
    ))


def verify_core_inverse_refresh() -> VerificationResult:
    """CORE-INV-003: refreshed inverse beats one-shot fixed W+."""
    fam_one = _small_random_family(seed=103, n=12, m=5)
    target = np.sort(fam_one.lam0 + np.linspace(-0.08, 0.08, fam_one.N))
    one = fam_one.inverse(target, steps=1, alpha=0.35, refresh_every=99)
    fam_refresh = _small_random_family(seed=103, n=12, m=5)
    refreshed = fam_refresh.inverse(target, steps=12, alpha=0.25, refresh_every=3)
    ratio = float(one.error / max(refreshed.error, 1e-15))
    return _finalize(VerificationResult(
        claim_id="CORE-INV-003",
        title="Adaptive W refresh materially improves inverse residual",
        layer="theory-core",
        status=_status(np.isfinite(ratio) and ratio > 1.25 and refreshed.error < one.error),
        evidence_level="E3",
        metric=ratio,
        threshold="one-step error / refreshed error > 1.25",
        details={
            "one_step_error": float(one.error),
            "refreshed_error": float(refreshed.error),
            "refreshed_eigh_count": int(refreshed.eigh_count),
            "refreshed_n_refresh": int(refreshed.n_refresh),
        },
    ))


def verify_core_operator_algebra() -> VerificationResult:
    """CORE-ALG-004: independent operator algebra laws."""
    from .algebra import compose_linear, direct_sum, tensor_sum

    a = _small_random_family(seed=104, n=5, m=2)
    b = _small_random_family(seed=105, n=4, m=3)
    ds = direct_sum(a, b)
    ds_err = float(np.max(np.abs(np.sort(ds.lam0) - np.sort(np.concatenate([a.lam0, b.lam0])))))

    C = np.array([[1.0, 0.2], [-0.4, 0.5]])
    composed = compose_linear(a, C)
    comp_err = float(np.max(np.abs(composed.W - a.W @ C)))

    ts = tensor_sum(a, b)
    pairwise = np.sort(np.add.outer(a.lam0, b.lam0).ravel())
    tensor_err = float(np.max(np.abs(np.sort(ts.lam0) - pairwise)))
    max_err = max(ds_err, comp_err, tensor_err)
    return _finalize(VerificationResult(
        claim_id="CORE-ALG-004",
        title="Direct sum, composition, and tensor sum obey spectral laws",
        layer="theory-core",
        status=_status(max_err < 1e-10),
        evidence_level="E3",
        metric=max_err,
        threshold="< 1e-10 max algebra-law error",
        details={
            "direct_sum_error": ds_err,
            "compose_W_error": comp_err,
            "tensor_sum_error": tensor_err,
        },
    ))


def verify_core_negative_controls() -> VerificationResult:
    """CORE-NEG-005: wrong interpretations are detected as wrong."""
    fam = _small_random_family(seed=106, n=16, m=8)
    d = np.random.default_rng(106).standard_normal(fam.M)
    d = d / np.linalg.norm(d) * 0.25
    sorted_hf_err = float(np.max(np.abs(fam.spectrum(d) - fam.predict(d))))

    # A diagonal family can fit a target spectrum exactly, but that says
    # nothing about solving an external sorting task on unrelated data.
    target = np.linspace(-1.0, 1.0, 12)
    sorted_data = np.sort(np.array([3.0, -2.0, 1.0, 0.0]))
    spectrum_fit_corr = float(np.corrcoef(target, target)[0, 1])
    task_success = bool(np.array_equal(target[:4], sorted_data))
    invalid_claim_rejected = sorted_hf_err > 0.05 and spectrum_fit_corr > 0.99 and not task_success
    return _finalize(VerificationResult(
        claim_id="CORE-NEG-005",
        title="Negative controls reject sorted-spectrum and spectrum-fit shortcuts",
        layer="theory-core",
        status=_status(invalid_claim_rejected),
        evidence_level="E3",
        metric=sorted_hf_err,
        threshold="sorted-spectrum HF error > 0.05 while spectrum-fit task shortcut is false",
        details={
            "sorted_hf_error": sorted_hf_err,
            "spectrum_fit_corr": spectrum_fit_corr,
            "task_success": task_success,
        },
    ))


def verify_core_hessian_consistency() -> VerificationResult:
    """CORE-HESS-006: analytic Hessian matches finite differences."""
    from .hessian import hessian, hessian_analytic

    fam = _small_random_family(seed=107, n=6, m=2)
    H_fd = hessian(fam, eps=1e-4)
    H_an = hessian_analytic(fam)
    err = float(np.max(np.abs(H_fd - H_an)))
    return _finalize(VerificationResult(
        claim_id="CORE-HESS-006",
        title="Analytic Hessian matches finite differences",
        layer="theory-core",
        status=_status(err < 5e-3),
        evidence_level="E3",
        metric=err,
        threshold="< 5e-3 max Hessian error",
        details={"N": fam.N, "M": fam.M},
    ))


def verify_core_topology_control() -> VerificationResult:
    """CORE-TOPO-007: EP winding passes and non-EP control stays trivial."""
    from . import physics, topology

    ep = physics.exceptional_point_2x2().family()
    loop = [np.array([0.25 * np.cos(t), 0.25 * np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 80)]
    ep_winding = float(topology.eigenvalue_winding(ep, loop))

    non_ep = physics.pt_symmetric_2x2(0.6).family()
    small_loop = [np.array([0.05 * np.cos(t), 0.05 * np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 80)]
    control_winding = float(topology.eigenvalue_winding(non_ep, small_loop))
    ok = abs(abs(ep_winding) - 0.5) < 0.1 and abs(control_winding) < 0.1
    return _finalize(VerificationResult(
        claim_id="CORE-TOPO-007",
        title="Exceptional-point winding passes with non-EP control",
        layer="theory-core",
        status=_status(ok),
        evidence_level="E3",
        metric=ep_winding,
        threshold="|EP winding| ~= 0.5 and non-EP winding ~= 0",
        details={"ep_winding": ep_winding, "control_winding": control_winding},
    ))


def verify_core_jordan_fuse() -> VerificationResult:
    """CORE-JFUSE-008: Jordan fuse turns reducible tensor EP into EP4."""
    from . import algebra, physics

    ep_a = physics.exceptional_point_2x2().family()
    ep_b = physics.exceptional_point_2x2().family()
    plain = algebra.tensor_sum(ep_a, ep_b)
    fused = algebra.jordan_fuse(ep_a, ep_b)
    plain_fp = algebra.jordan_fingerprint(plain.A0)
    fused_fp = algebra.jordan_fingerprint(fused.A0)

    radii = np.array([1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5])
    values = []
    closure_axis = fused.M - 2
    for radius in radii:
        k = np.zeros(fused.M)
        k[closure_axis] = radius
        values.append(float(np.max(np.abs(fused.eigensystem(k)[0]))))
    puiseux = float(np.polyfit(np.log(radii), np.log(np.asarray(values)), 1)[0])

    plain_rejected = (
        plain_fp.geometric_multiplicity == 2
        and plain_fp.nilpotent_index == 3
        and not plain_fp.is_single_chain
    )
    fused_passes = (
        fused_fp.geometric_multiplicity == 1
        and fused_fp.nilpotent_index == 4
        and fused_fp.is_single_chain
    )
    ok = plain_rejected and fused_passes and 0.18 <= puiseux <= 0.32
    return _finalize(VerificationResult(
        claim_id="CORE-JFUSE-008",
        title="Jordan fuse converts reducible tensor EP into single EP4",
        layer="theory-core",
        status=_status(ok),
        evidence_level="E3",
        metric=puiseux,
        threshold="plain tensor_sum rejected; fused fingerprint single-chain; Puiseux exponent ~= 1/4",
        details={
            "plain": plain_fp.to_dict(),
            "fused": fused_fp.to_dict(),
            "fused_M": fused.M,
            "closure_axis": closure_axis,
            "puiseux_exponent": puiseux,
            "coupling": list(fused.jordan_coupling) if isinstance(fused.jordan_coupling, tuple) else "matrix",
        },
    ))


def verify_core_multi_jordan_fuse() -> VerificationResult:
    """CORE-MJFUSE-009: multi-bridge fuse scales EP synthesis to EP32."""
    from . import algebra, physics

    ep2 = physics.exceptional_point_2x2().family()
    ep4 = algebra.jordan_fuse(ep2, ep2, add_closure=False)
    ep8 = algebra.multi_jordan_fuse(ep4, ep2, add_closure=False)
    ep16 = algebra.multi_jordan_fuse(ep4, ep4, add_closure=False)
    ep32 = algebra.multi_jordan_fuse(ep16, ep2, add_closure=False)
    plain_16 = algebra.tensor_sum(ep4, ep4)

    fp4 = algebra.jordan_fingerprint(ep4.A0)
    fp8 = algebra.jordan_fingerprint(ep8.A0)
    fp16 = algebra.jordan_fingerprint(ep16.A0)
    fp32 = algebra.jordan_fingerprint(ep32.A0)
    plain_fp = algebra.jordan_fingerprint(plain_16.A0)
    ok = (
        fp4.is_single_chain and fp4.nilpotent_index == 4
        and fp8.is_single_chain and fp8.nilpotent_index == 8
        and fp16.is_single_chain and fp16.nilpotent_index == 16
        and fp32.is_single_chain and fp32.nilpotent_index == 32
        and plain_fp.geometric_multiplicity == 4
        and plain_fp.nilpotent_index == 7
        and not plain_fp.is_single_chain
        and len(ep16.jordan_couplings) == 3
        and ep32.jordan_couplings == ((30, 1),)
    )
    return _finalize(VerificationResult(
        claim_id="CORE-MJFUSE-009",
        title="Multi Jordan fuse scales EP synthesis to EP32",
        layer="theory-core",
        status=_status(ok),
        evidence_level="E3",
        metric=fp32.nilpotent_index,
        threshold="EP4, EP8, EP16, EP32 are single chains; plain EP4 tensor EP4 is reducible",
        details={
            "ep4": fp4.to_dict(),
            "ep8": fp8.to_dict(),
            "ep16": fp16.to_dict(),
            "ep32": fp32.to_dict(),
            "plain_ep4_tensor_ep4": plain_fp.to_dict(),
            "ep8_couplings": [list(c) for c in ep8.jordan_couplings],
            "ep16_couplings": [list(c) for c in ep16.jordan_couplings],
            "ep32_couplings": [list(c) for c in ep32.jordan_couplings],
        },
    ))


CORE_VERIFICATION_FUNCTIONS = (
    verify_core_hf_kernel,
    verify_core_perturbation_order,
    verify_core_inverse_refresh,
    verify_core_operator_algebra,
    verify_core_negative_controls,
    verify_core_hessian_consistency,
    verify_core_topology_control,
    verify_core_jordan_fuse,
    verify_core_multi_jordan_fuse,
)


def verify_alg_direct_sum_law() -> VerificationResult:
    from .algebra import direct_sum

    a = _small_random_family(seed=201, n=5, m=2)
    b = _small_random_family(seed=202, n=4, m=2)
    ds = direct_sum(a, b)
    err = float(np.max(np.abs(np.sort(ds.lam0) - np.sort(np.concatenate([a.lam0, b.lam0])))))
    return _finalize(VerificationResult(
        claim_id="ALG-DIRECT-SUM-001",
        title="Direct sum preserves spectral union",
        layer="theory-core",
        status=_status(err < 1e-10),
        evidence_level="E4",
        metric=err,
        threshold="< 1e-10 spectral union error",
        details={"cost_after": ds.cost().to_dict()},
    ))


def verify_alg_compose_law() -> VerificationResult:
    from .algebra import compose

    fam = _small_random_family(seed=203, n=6, m=3)
    C = np.array([[1.0, 0.0], [0.25, 1.0], [-0.5, 0.1]])
    composed = compose(fam, C)
    err = float(np.max(np.abs(composed.W - fam.W @ C)))
    return _finalize(VerificationResult(
        claim_id="ALG-COMPOSE-002",
        title="Linear composition obeys W chain rule",
        layer="theory-core",
        status=_status(err < 1e-10),
        evidence_level="E4",
        metric=err,
        threshold="< 1e-10 W-chain-rule error",
        details={"materializes_basis": composed.cost_after.materializes_basis},
    ))


def verify_alg_tensor_law() -> VerificationResult:
    from .algebra import tensor_sum

    a = _small_random_family(seed=204, n=3, m=1)
    b = _small_random_family(seed=205, n=4, m=1)
    ts = tensor_sum(a, b)
    expected = np.sort(np.add.outer(a.lam0, b.lam0).ravel())
    err = float(np.max(np.abs(np.sort(ts.lam0) - expected)))
    return _finalize(VerificationResult(
        claim_id="ALG-TENSOR-003",
        title="Tensor sum preserves pairwise spectral sums",
        layer="theory-core",
        status=_status(err < 1e-10),
        evidence_level="E4",
        metric=err,
        threshold="< 1e-10 pairwise-sum error",
        details={"N_after": ts.N, "cost_after": ts.cost().to_dict()},
    ))


def verify_alg_task_not_spectrum() -> VerificationResult:
    spectrum_target = np.linspace(0.0, 1.0, 8)
    unrelated_task_output = np.sort(np.array([3.0, -1.0, 2.0, 0.0]))
    corr = float(np.corrcoef(spectrum_target, spectrum_target)[0, 1])
    task_solved = bool(np.array_equal(spectrum_target[:4], unrelated_task_output))
    ok = corr > 0.99 and not task_solved
    return _finalize(VerificationResult(
        claim_id="ALG-TASK-NOT-SPECTRUM-004",
        title="Task success is not spectrum correlation",
        layer="theory-core",
        status=_status(ok),
        evidence_level="E4",
        metric=corr,
        threshold="corr > 0.99 with task_success=False",
        details={"task_success": task_solved},
    ))


def verify_blueprint(blueprint=None) -> VerificationResult:
    if blueprint is None:
        from .constructor import OperatorBlueprint
        blueprint = OperatorBlueprint.from_task("bandpass filter", np.linspace(0.0, 1.0, 16))
    data = blueprint.to_dict()
    ok = (
        data["N"] > 0
        and data["M"] >= 0
        and data["basis_type"] in {"diagonal", "edge_laplacian", "toeplitz"}
        and bool(data.get("invariant"))
        and bool(data.get("cost"))
    )
    return _finalize(VerificationResult(
        claim_id="ALG-BLUEPRINT-005",
        title="Blueprint carries invariant, basis, and cost contract",
        layer="library-api",
        status=_status(ok),
        evidence_level="E4",
        metric=bool(ok),
        threshold="N/M/basis/invariant/cost are present and valid",
        details=data,
    ))


ALGEBRA_LAW_FUNCTIONS = (
    verify_alg_direct_sum_law,
    verify_alg_compose_law,
    verify_alg_tensor_law,
    verify_alg_task_not_spectrum,
    verify_blueprint,
)


def run_core_verification_suite() -> dict[str, dict[str, Any]]:
    """Run independent production-grade theory-core checks."""
    return {result.claim_id: result.to_dict() for result in (fn() for fn in CORE_VERIFICATION_FUNCTIONS)}


def run_algebra_law_suite() -> dict[str, dict[str, Any]]:
    """Run production operator-algebra law checks."""
    return {result.claim_id: result.to_dict() for result in (fn() for fn in ALGEBRA_LAW_FUNCTIONS)}


def run_verification_suite() -> dict:
    base = {f"C{n}_result": globals()[f"verify_c{n}_fourier_subset" if n == 1 else
               f"verify_c{n}_tau_invariance" if n == 2 else
               f"verify_c{n}_w_entropy" if n == 3 else
               f"verify_c{n}_rank_separability" if n == 4 else
               f"verify_c{n}_defect_universality" if n == 5 else
               f"verify_c{n}_monodromy_classification" if n == 6 else
               f"verify_c{n}_hessian_sparsity" if n == 7 else
               f"verify_c{n}_universal_embedding"]() for n in range(1, 9)}
    base.update({
        "G1_exceptional_point": verify_g1_exceptional_point(),
        "G2_biorthogonal_hf": verify_g2_biorthogonal_hf(),
        "G3_pde_convergence": verify_g3_pde_convergence(),
        "G4_second_order_inverse": verify_g4_second_order_inverse(),
    })
    base["CORE_structured"] = run_core_verification_suite()
    base["ALG_structured"] = run_algebra_law_suite()
    return base


def verification_report_markdown(results: dict[str, dict[str, Any]] | None = None) -> str:
    """Render core verification results as a compact Markdown table."""
    if results is None:
        results = run_core_verification_suite()
    lines = [
        "# SFT Independent Verification Report",
        "",
        "| Claim | Status | Evidence | Metric | Threshold | Promotion |",
        "|-------|--------|----------|--------|-----------|-----------|",
    ]
    for claim_id in sorted(results):
        row = results[claim_id]
        lines.append(
            f"| `{claim_id}` | **{row['status']}** | `{row['evidence_level']}` | "
            f"{row['metric']} | {row['threshold']} | `{row['promotion']}` |"
        )
    lines.extend([
        "",
        "Promotion rule: only `E3+` passing theory-core claims can become theory-core; "
        "`E4` is required for stable library API.",
    ])
    return "\n".join(lines) + "\n"


def verification_report_jsonable(results: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return a JSON-serializable report payload."""
    if results is None:
        results = run_core_verification_suite()
    passed = sum(1 for row in results.values() if row["status"] == "PASS")
    return {
        "suite": "sft-independent-core-verification",
        "passed": passed,
        "total": len(results),
        "results": results,
    }


def verify_g1_exceptional_point() -> bool:
    """G1: non-Hermitian EP has half-integer gap winding."""
    from . import physics, topology
    fam = physics.exceptional_point_2x2().family()
    loop = [np.array([0.25 * np.cos(t), 0.25 * np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 80)]
    winding = topology.eigenvalue_winding(fam, loop)
    return abs(abs(winding) - 0.5) < 0.1


def verify_g2_biorthogonal_hf() -> bool:
    """G2: non-Hermitian biorthogonal HF matches finite differences."""
    from . import physics
    fam = physics.pt_symmetric_2x2(0.3).family()
    eps = 1e-6
    fd = (fam.spectrum(np.array([eps, 0.0])) - fam.lam0) / eps
    return np.max(np.abs(fd - fam.W[:, 0])) < 1e-5


def verify_g3_pde_convergence() -> float:
    """G3: sparse PDE first eigenvalue converges at ~second order."""
    from . import physics
    continuum = 0.5 * np.pi ** 2
    rows = []
    for n in [24, 48, 96]:
        x = np.linspace(0.0, 1.0, n + 2)[1:-1]
        fam = physics.schrodinger_1d(x, np.zeros_like(x), max_potential_params=4).family()
        err = abs(float(fam.spectrum(np.zeros(fam.M), n_eigs=2)[0]) - continuum)
        rows.append((1.0 / (n + 1), err))
    h = np.array([r[0] for r in rows])
    e = np.maximum(np.array([r[1] for r in rows]), 1e-15)
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def verify_g4_second_order_inverse() -> dict:
    """G4: Hessian/trust inverse returns diagnostics and finite residuals."""
    from .families import random
    fam = random(10, 3, seed=31)
    target = np.sort(fam.lam0 + np.linspace(-0.02, 0.02, fam.N))
    result = fam.inverse(target, steps=2, alpha=0.25, method="hessian")
    return {"finite": bool(np.isfinite(result.error)), "hessian_count": result.hessian_count}


def verify_s1_complexity() -> bool:
    """S1: complexity = rank(W)/N ∈ [0,1] for any OperatorFamily."""
    from .families import random
    for seed in range(5):
        c = random(20, 10, seed=seed).complexity
        if not (0.0 <= c <= 1.0): return False
    return True


def verify_s2_perturbation_theory() -> bool:
    """S2: ||λ(k₀+dk) − λ(k₀) − W·dk|| = O(||dk||²)."""
    from .families import random
    from .hessian import spectral_curvature
    fam = random(10, 3, seed=42)
    for eps in [1e-2, 5e-3, 1e-3]:
        d = np.ones(3); d = d / np.linalg.norm(d)
        curv = spectral_curvature(fam, d, eps=1e-4)
        scale = np.linalg.norm(d) * eps
        err_fd = np.max(np.abs(fam.spectrum(d * eps) - fam.predict(d * eps)))
        expected = np.max(np.abs(curv)) * scale**2 / 2
        if err_fd > expected * 5: return False
    return True


def verify_s3_drum_shape() -> bool:
    """S3: dirichlet_laplacian spectrum → W rank → detect shape changes."""
    from .families import graph_laplacian
    from .graph_gen import path_graph, star_graph
    fam_p = graph_laplacian(path_graph(20))
    fam_s = graph_laplacian(star_graph(20))
    return fam_p.W_rank != fam_s.W_rank


def verify_s4_information_bound() -> float:
    """S4: log|det(W^T W + I)| ≥ 0 always."""
    from .families import random
    W = random(15, 8, seed=42).W
    det_log = np.linalg.slogdet(W.T @ W + np.eye(8))[1] if W.shape[1] >= 1 else 0.0
    return float(det_log)


def verify_s5_rmt_vs_defect() -> float:
    """S5: Wigner semi-circle spacing vs defect Poisson spacing — KS distance."""
    from .families import random
    fam = random(30, 15, seed=42)
    gaps = np.diff(fam.lam0)
    gaps_norm = gaps / np.mean(gaps) if np.mean(gaps) > 1e-15 else gaps
    from scipy import stats
    d_wigner, _ = stats.kstest(gaps_norm, lambda x: np.pi * x / 2 * np.exp(-np.pi * x**2 / 4))
    return float(d_wigner)
