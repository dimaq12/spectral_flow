# spectral_flow

**One formula. One kernel. All operators.**

```
W(i,j) = v_iᵀ · B_j · v_i  =  ∂λ_i/∂k_j
```

A(k) = A₀ + Σ kⱼ·Bⱼ — линейное семейство операторов.  
W — матрица спектрального потока: одна диагонализация → все производные собственных чисел по всем параметрам.

---

## Зачем

| Раньше | Сейчас |
|--------|--------|
| Для каждого возмущения k — новый `eigh` O(N³) | Один `eigh` при сборке, потом `W·dk` за O(N·M) |
| Spectral flow — абстрактная теория | W как Jacobian оператора — предсказание, inverse design, топология |
| Каждый домен — свой код | 12 адаптеров: аудио, изображение, граф, текст, видео, воксель, point cloud, молекула, финансы, таблица, меш |

---

## 15 секунд до результата

```python
import sft

# Ядро
fam = sft.families.random(N=100, M=30)
fam.W                    # (100,30) — ∂λ/∂k
fam.predict(dk)          # λ(k₀+dk) за O(N·M)
fam.inverse(target)      # найти k под целевой спектр
fam.complexity           # rank(W)/N — структурная сложность

# Адаптеры — killer feature
pic   = sft.image(pixels, patch_size=8)
sound = sft.audio(signal, sr=44100, n_bands=16)
net   = sft.graph(adjacency)

# Задача на естественном языке → оператор
fam = sft.from_task("sort these numbers", data)

# Графы — структурный анализ O(1) после precompute
gop = sft.graphop.GraphOperator(edges)
gop.is_bridge(0, 1)       # O(1)
gop.is_articulation(5)    # O(1)

# Топология
tracked, swaps = sft.topology.monodromy(fam, loop)
holonomy = sft.topology.berry_holonomy(fam, loop)

# Алгебра операторов
fam_sum = sft.algebra.direct_sum(a, b)   # ⊕
fam_ten = sft.algebra.tensor_sum(a, b)   # ⊗

# Инварианты — 5 ключей за один вызов
fp = sft.invariants.all_invariants(fam)

# Streaming CDF
stream = sft.streaming.StreamingCDF(capacity=1000)
for x in data_stream: stream.add(x)
stream.cdf(threshold)
```

---

## 22 модуля

| Модуль | Что делает |
|--------|-----------|
| `core` | `OperatorFamily`, W, W⁺, predict, inverse, nullspace |
| `algebra` | ⊕, ∘, ⊗, ∫ ожидание |
| `topology` | монодромия, фаза Берри, exceptional points, spectral flow |
| `hessian` | ∂²λ/∂k² — аналитически и конечно-разностно |
| `families` | random, graph_laplacian, toeplitz, diagonal, avoided_crossing |
| `adapters` | 12 доменных адаптеров (Audio…Mesh) |
| `tasks` | classify_task, cdf_rank_sort, dct_matrix, filter_via_dct |
| `constructor` | from_task("sort", data) → OperatorFamily |
| `graphop` | мосты, точки сочленения, k-core за O(1) |
| `embed` | детерминированные графовые эмбеддинги |
| `order` | CDF, ранг, квантиль, дефект-спектроскопия |
| `cluster` | спектральная кластеризация, kNN, авто-базис |
| `compress` | спектральное сжатие, DCT codec |
| `transport` | оптимальный транспорт 1D (Monge map + W₂) |
| `streaming` | онлайн CDF и ORDER |
| `carleman` | GF(2)/GF(3) операторы, комплексный HF |
| `homotopy` | гомотопическое продолжение, Tikhonov W⁺, trust-region |
| `inversion` | bottleneck, fixed_point, monodromy inverse |
| `invariants` | 5 глобальных инвариантов: kurtosis, sparsity, preimage, coherence, zeta |
| `basis` | Toeplitz, DCT/Fourier, affinity, граф-генераторы |
| `arnoldi` | итерация Арнольди, Ritz, Krylov |
| `codec` | InstantSpectralCodec: M·Δk encode/decode |
| `verify` | C1-C8 + S1-S5 верификация теорем |

---

## Теоретические основы

SFT обобщает преобразование Фурье на некоммутативные операторы.
Fourier ⊂ SFT: для циркулянтных операторов W совпадает с DFT.
Для произвольных A(k) = A₀ + Σ kⱼ·Bⱼ — W даёт полный Jacobian спектра.

**Ранг W = вычислительная сложность задачи:**
- rank(W) = 1 → ORDER-режим (сортировка, CDF)
- rank(W) ≪ N → структура (графы, фильтры)
- rank(W) ≈ N → random (нет shortcut)

**Nullspace W = изоспектральное многообразие:**
dim(ker(W)) направлений в k-пространстве не меняют спектр.
Оператор их «не слышит».

---

## Установка

```bash
pip install -e .
```

Зависимости: `numpy>=1.24`, `scipy>=1.10`. Python ≥ 3.10.

---

## Лицензия

MIT
