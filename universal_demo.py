#!/usr/bin/env python3
"""
sft/universal_demo.py — SFT applied to EVERYTHING.

  Audio, Image, Graph, Text, Timeseries — one kernel, one formula.

  import sft

  sft.audio(signal).kernel       → W for EQ
  sft.image(pixels).kernel       → W for region changes
  sft.graph(adjacency).kernel    → W for edges (GraphSFT)
  sft.text(corpus).kernel        → W for word associations
  sft.timeseries(data).kernel    → W for lag weights
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import sft


def demo_audio():
    print("═══ AUDIO ═══\n")
    rng = np.random.default_rng(0)
    sr = 44100
    t = np.arange(sr) / sr
    signal = (np.sin(2 * np.pi * 440 * t) +
              0.5 * np.sin(2 * np.pi * 880 * t) +
              0.3 * np.sin(2 * np.pi * 1320 * t) +
              0.1 * rng.standard_normal(sr))

    sound = sft.audio(signal, sample_rate=sr, frame_size=2048, n_bands=8)
    print(f"   Bands: {sound.n_bands}  ({[f'{f:.0f}Hz' for f in sound.band_freqs]})")
    print(f"   Kernel shape: {sound.kernel.shape}")
    print(f"   Complexity: rank/n = {sound.complexity:.3f}")
    print(f"   Spectrum head: {[f'{x:.3f}' for x in sound.spectrum[:8]]}")

    # Predict: boost band 0 by +0.1
    dk = np.zeros(sound.n_bands)
    dk[0] = 0.1
    lam_pred = sound.predict(dk)
    print(f"   Predicted λ change (band 0 +0.1): {[f'{lam_pred[i]-sound.spectrum[i]:.3e}' for i in range(4)]}")
    print()


def demo_image():
    print("═══ IMAGE ═══\n")
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    img = np.sin(x[:, None] * 6) * np.cos(y[None, :] * 4) + 0.05 * rng.standard_normal((64, 64))

    pic = sft.image(img, patch_size=8, n_regions=9)
    print(f"   Image: {img.shape}, patches: {pic.n_patches}")
    print(f"   Kernel shape: {pic.kernel.shape}")
    print(f"   Complexity: rank/n = {pic.complexity:.3f}")
    print(f"   Spectrum head: {[f'{x:.3f}' for x in pic.spectrum[:6]]}")
    print()


def demo_graph():
    print("═══ GRAPH ═══\n")
    # Path graph → GraphSFT
    N = 30
    adj = np.zeros((N, N))
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0

    net = sft.graph(adj)
    print(f"   Graph: {N} nodes, {net.n_edges} edges")
    print(f"   Kernel shape: {net.kernel.shape}  ({N} λ × {net.n_edges} edges)")
    print(f"   Complexity: rank/n = {net.complexity:.3f}")
    print(f"   GraphSFT: W(i,e) = (v_i(u)−v_i(v))²")
    print(f"   Isospectral dim: {net.isospectral_dim} (of {net.n_edges}) — ghost edges")
    print()


def demo_text():
    print("═══ TEXT ═══\n")
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
        "machine learning is fun and powerful",
        "deep learning transforms everything",
    ]

    doc = sft.text(corpus, window=3)
    print(f"   Vocabulary: {doc.n_words} words  {doc.vocab[:8]}...")
    print(f"   Kernel shape: {doc.kernel.shape}")
    print(f"   Complexity: rank/n = {doc.complexity:.3f}")
    print(f"   Co-occurrence matrix density: {(doc.cooc > 0).sum()/doc.cooc.size:.3f}")
    print()


def demo_timeseries():
    print("═══ TIMESERIES ═══\n")
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4 * np.pi, 500)
    ts = (np.sin(t * 3) * np.exp(-t / 10) +
           0.3 * np.cos(t * 1.7) +
           0.02 * t +
           0.05 * rng.standard_normal(500))

    ts_adapter = sft.timeseries(ts, window_len=40)
    print(f"   Timeseries: {len(ts)} points, window={ts_adapter.L}")
    print(f"   Kernel shape: {ts_adapter.kernel.shape}")
    print(f"   Complexity: rank/n = {ts_adapter.complexity:.3f}")
    print(f"   Singular spectrum (top 6): {[f'{x:.3f}' for x in ts_adapter.spectrum[:6]]}")
    print()


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SFT UNIVERSAL — one kernel, all domains                 ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    demo_audio()
    demo_image()
    demo_graph()
    demo_text()
    demo_timeseries()

    print("═══ ONE FORMULA ═══\n")
    print("   W(i,j) = v_i^T · B_j · v_i  =  ∂λ_i/∂k_j")
    print()
    print("   import sft")
    print("   sft.audio(signal).predict(eq_changes)")
    print("   sft.image(pixels).predict(brightness)")
    print("   sft.graph(adjacency).predict(edge_weights)")
    print("   sft.text(corpus).predict(word_associations)")
    print("   sft.timeseries(series).predict(lag_weights)")
    print()
    print("   Every domain. Every data type. One kernel.")


if __name__ == "__main__":
    main()
