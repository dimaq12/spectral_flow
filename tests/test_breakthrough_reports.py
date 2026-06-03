"""Smoke-run tiny breakthrough reports."""
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_demo(name: str, env: dict[str, str]):
    full_env = os.environ.copy()
    full_env.update(env)
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=str(ROOT.parent),
        env=full_env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )


def test_breakthrough_pde_report_smoke(tmp_path):
    _run_demo("demo_breakthrough_pde.py", {
        "SFT_BREAKTHROUGH_REPORT_DIR": str(tmp_path),
        "SFT_BREAKTHROUGH_PDE_N": "80",
        "SFT_BREAKTHROUGH_PDE_Q": "8",
        "SFT_BREAKTHROUGH_PDE_K": "3",
        "SFT_BREAKTHROUGH_BASELINE_QUERIES": "3",
    })
    text = (tmp_path / "breakthrough_pde_queries.md").read_text()
    assert "industrial_eigsh_loop" in text
    assert "sft_predict_many" in text
    assert "Speedup" in text or "speedup" in text


def test_breakthrough_graph_report_smoke(tmp_path):
    _run_demo("demo_breakthrough_graph_inverse.py", {
        "SFT_BREAKTHROUGH_REPORT_DIR": str(tmp_path),
        "SFT_BREAKTHROUGH_GRAPH_N": "30",
        "SFT_BREAKTHROUGH_GRAPH_K": "4",
        "SFT_BREAKTHROUGH_GRAPH_REDUCED_DIM": "4",
        "SFT_BREAKTHROUGH_GRAPH_NFEV": "4",
    })
    text = (tmp_path / "breakthrough_graph_inverse.md").read_text()
    assert "least_squares + eigsh" in text
    assert "SFT trust inverse" in text


def test_breakthrough_pde2d_report_smoke(tmp_path):
    _run_demo("demo_breakthrough_pde2d_control.py", {
        "SFT_BREAKTHROUGH_REPORT_DIR": str(tmp_path),
        "SFT_BREAKTHROUGH_PDE2D_NX": "8",
        "SFT_BREAKTHROUGH_PDE2D_NY": "8",
        "SFT_BREAKTHROUGH_PDE2D_Q": "8",
        "SFT_BREAKTHROUGH_PDE2D_K": "3",
        "SFT_BREAKTHROUGH_PDE2D_M": "4",
        "SFT_BREAKTHROUGH_BASELINE_QUERIES": "3",
    })
    text = (tmp_path / "breakthrough_pde2d_control.md").read_text()
    assert "industrial_eigsh_loop" in text
    assert "SFT trust inverse" in text
