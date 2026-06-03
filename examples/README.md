# SFT Examples

These demos are executable smoke tests for the library's main promises:
fast spectral prediction, one-call adapters, topology, codecs, graph tools,
and operator-flavored domain experiments.

Run all report-generating demos from the repository root:

```bash
for f in sft/examples/demo_*.py; do python3 "$f"; done
```

Each script writes a markdown report into `sft/examples/reports/`.

## Best First Runs

| Demo | Why run it |
|------|------------|
| `demo_benchmark.py` | Core speed: build, `predict_many`, exact/partial eigensolve, inverse diagnostics, topology |
| `demo_breakthrough_pde.py` | Sparse PDE query batch vs `eigsh` and warm-start `lobpcg` |
| `demo_breakthrough_graph_inverse.py` | Inverse graph design vs `least_squares` + sparse `eigsh` |
| `demo_breakthrough_pde2d_control.py` | 2D PDE query/control workload vs sparse eigensolver baselines |
| `demo_adapters_load.py` | All 12 adapters from raw data to `OperatorFamily` |
| `demo_topology.py` | Berry holonomy and eigenvector sign flip |
| `demo_quantum_ep.py` | Non-Hermitian exceptional point and half-winding |
| `demo_pde_multigrid.py` | Real sparse PDE operator convergence across grids |
| `demo_inverse_graph_design.py` | Target spectrum -> recovered graph edge weights |
| `demo_codec.py` | `W @ dk` encode and `W_pinv @ y` decode |
| `demo_graphop.py` | O(1) bridge and articulation queries after precompute |

## Interactive Showcases

`demo.py` and `universal_demo.py` print walkthroughs instead of writing reports.
They are good for a quick terminal tour:

```bash
python3 sft/examples/demo.py
python3 sft/examples/universal_demo.py
```
