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
| `demo_benchmark.py` | Core speed: build, predict, exact eigensolve, inverse, topology |
| `demo_adapters_load.py` | All 12 adapters from raw data to `OperatorFamily` |
| `demo_topology.py` | Berry holonomy and eigenvector sign flip |
| `demo_codec.py` | `W @ dk` encode and `W_pinv @ y` decode |
| `demo_graphop.py` | O(1) bridge and articulation queries after precompute |

## Interactive Showcases

`demo.py` and `universal_demo.py` print walkthroughs instead of writing reports.
They are good for a quick terminal tour:

```bash
python3 sft/examples/demo.py
python3 sft/examples/universal_demo.py
```
