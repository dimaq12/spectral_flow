# Unreleased: Computational Breakthrough Benchmarks

This work does not publish a new version by itself.  It adds reproducible
benchmark evidence for the existing SFT 0.3 computational model.

## Benchmark Methodology

- Added `sft.benchmarks` with reusable timing, memory, eigensolve-count, error,
  speedup, and machine-metadata helpers.
- Added sparse PDE query benchmarks against `eigsh` and warm-start `lobpcg`.
- Added inverse graph design benchmark against `scipy.optimize.least_squares`
  with sparse eigensolver residuals.
- Added 2D PDE query/control benchmark against repeated sparse eigensolves.
- Added generated reports and a combined computational breakthrough summary.
- Added smoke tests for benchmark harness, industrial baselines, and report
  generation.

---

# SFT 0.3 Notes

SFT 0.3 adds the Spectral Geometry Lab: non-Hermitian spectral geometry,
exceptional-point topology, physics/PDE operator factories, second-order
spectral control, isospectral flows, and reproducible scientific demos.

## Spectral Geometry

- Added `OperatorFamily(..., hermitian=False)` for complex non-Hermitian
  families.
- Added `BiorthogonalState` with normalized left/right eigenvectors.
- Added complex Hellmann-Feynman kernels `W[i,j] = <L_i|B_j|R_i>`.
- Added complex monodromy, eigenvalue gap winding, and
  `exceptional_point_atlas(...)`.

## Spectral Control

- Added inverse method selector: `method="linear"|"hessian"|"trust"|"homotopy"`.
- Added Hessian/trust diagnostics: `trajectory`, `residual_history`,
  `hessian_count`, and `method` on `InverseResult`.
- Added fluent builder: `family.flow(target).via("linear").second_order().solve()`.

## Physics Lab

- Added `sft.physics` factories for PT-symmetric Hamiltonians,
  square-root exceptional points, 1D Schrödinger operators, and grid
  Laplacians.
- Added isospectral flows via `family.isospectral_flow(...)`.
- Added spectral transport paths in `sft.transport`.
- Added report demos for quantum exceptional points, PDE multigrid convergence,
  and inverse graph design.

## Verification

- Extended `sft.verify.run_verification_suite()` with geometry gates:
  exceptional-point winding, biorthogonal HF finite differences, PDE
  convergence, and second-order inverse diagnostics.

---

# SFT 0.2 Notes

SFT 0.2 moves the library toward a fast spectral operator system while keeping the 0.1 public API intact.

## Speed

- Added sparse-aware `OperatorFamily` construction for sparse `A0` and sparse graph adjacency inputs.
- Added partial spectrum support via `family.spectrum(k, n_eigs=K, which="SM")` and `family.eigensystem(...)`.
- Added partial inverse support via `family.inverse(target, n_eigs=K)` for selected spectral modes.
- Added `predict_many(DK)` for batched first-order spectral prediction.
- Routed topology operations through the core eigensolver layer.
- Extended benchmark reporting with `predict_many`, sparse partial spectrum, backend kind, dense memory estimate, `eigh_count`, and refresh count.

## Operator API

- Added fluent verbs: `refresh(k)`, `at(k)`, `shift(dk)`, `toward(target)`, and `solve(target)`.
- Added adapter parity for the same fluent verbs where safe.
- Added operator dunders: `A + B` for direct sum, `A @ C` for linear composition, and `A @ dk` for prediction.
- Added `OperatorBlueprint.from_task(...).build(data)` alongside the existing dict blueprint API.
- Added task aliases `sft.task(...)` and `sft.pipe(...)`.

## Stability

- Added validation for NaN/Inf in operator matrices, parameters, and targets.
- Added edge-case tests for empty bases, sparse graphs, adapter inputs, inverse helpers, and README-style snippets.
- Avoided runtime warnings for all-NaN tabular columns and single-point point clouds.
- Added structured-aware Hessian tensor paths for edge/diagonal backends.

## Compatibility

- Existing `OperatorFamily`, `fam.predict`, `fam.inverse`, `sft.solve`, adapter factories, and tuple unpacking of inverse results remain supported.
- `InverseResult` remains tuple-compatible and now reports `eigh_count` and `time_ms`.
- Licensing changed to the SFT Permissive Attribution License: broad use is allowed, with required attribution to Dmitry Sierikov and the project link.
