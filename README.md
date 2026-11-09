# leech-cocode

**An exact Cocode-Turyn construction of the Leech lattice, with a syndrome-table decoder verified against two independent exact-ML oracles.**

This repository implements the Turyn glue model of the Leech lattice Λ₂₄
from first principles — it builds the extended binary Golay code G₂₄ over
GF(2¹¹) (not as a lookup table), forms the glue group of the scaled lattice
√2·Λ₂₄, and decodes with a 2¹²-entry coset table whose addresses come from
the *cocode* (the dual code C* syndromes) together with the mod-4 sum bit.
The companion note (PDF) develops the construction; the code verifies it.

All arithmetic is exact (Python-int bitmasks) in the construction and
verification stages; decoding uses float vectors in √2·Λ₂₄ coordinates.

## What the test suite verifies

`python run_tests.py` runs four stages, deterministic (fixed seeds).
Measured on commit when published: **~44 s end-to-end, all pass**.

**Stage 1 — structural (exact integer arithmetic, ~7 s)**
- G₂₄: dimension 12, weight enumerator {0:1, 8:759, 12:2576, 16:759, 24:1}, self-dual, covering radius 4 (12,951 words ≤ weight 4, avg 3.16 per coset).
- Turyn glue group |G| = 4096, ≅ (Z/2)¹² (all element orders ≤ 2).
- The integer-count glue subgroup equals **ker(σ) ∩ ker(Σ)**, with both sides enumerated *independently* (2048 = 2048).
- G is *not* a product of its per-block projections (|πᵢ| = 128, |πᵢⱼ| = 2048) — a reported negative structural result.
- Kissing number by exact three-shape enumeration: 1104 + 97152 + 98304 = **196,560**.

**Stage 2 — cross-validation (~7 s)**
- Two structurally *independent* exact-ML oracles (family enumeration; glue-coset decoding with per-block E₈) agree to **4.4×10⁻¹⁶**.
- The syndrome-table decoder is exact on the radius-1 region (29/30 sampled points inside it), 100% overall agreement at σ = 0.12.

**Stage 3 — exactness suite (N = 2000 per σ)**
- σ = 0.20: ML agreement 96.45% | radius-1 agreement **100.00%** | in-BDD 99.917% | 65 µs/pt
- σ = 0.35: ML agreement 55.65% | radius-1 agreement **100.00%** | in-BDD 100.000% | 68 µs/pt

**Stage 4 — candidate-weight sweep (W knob)**
- W = 4: 95.20% ML agreement (63 µs/pt); W = 5: 98.20% (126 µs/pt); W = 6: 98.60% (494 µs/pt).

Expected final output: `=== ALL TESTS PASSED ===`.

## Quickstart

```bash
pip install numpy
python run_tests.py        # ~44-90 s on a typical machine
```

No data files, no network, no build step. Requires Python 3.9+ and numpy.

## Honest scope

- The decoder is **provably exact whenever the candidate is within the
  BDD ball** (radius 1 in Λ₂₄ scale). Outside it, agreement with ML is
  high but not guaranteed (weight-≥5 parity error patterns are the gap);
  Stage 2/3 measure and report the true ML-agreement rate at several
  noise levels rather than asserting an unconditional bound.
- Stage-2 oracle B is independent of the family lemma; but both oracles and
  the decoder share the elementary "componentwise-closest + one ±2" lemma
  within a parity family, argued in the note.
- Distances are floats (half-integer √2-scaled) checked at tight tolerances;
  exact rational verification is a possible hardening step.
- Timings above were measured on one machine; treat them as indicative.
- The benchmark tables in the PDF note were produced with slightly
  different parameter settings (seeds/N) than `run_tests.py`'s defaults,
  so a few percentage points may differ; the numbers in this README are
  from the committed script as-is.

## Layout

| File | Contents |
|---|---|
| `golay.py` | Extended binary Golay code G₂₄ from first principles (QR code over GF(2¹¹), α = primitive 23rd root of unity), coset-leader table |
| `leech_glue.py` | √2·Λ₂₄ model, Turyn glue group (BFS closure), syndrome maps σ/Σ, E₈ decode |
| `decoder.py` | Syndrome-table Cocode-Turyn decoder + the two exact-ML oracles |
| `run_tests.py` | The four-stage verification suite (the numbers above) |
| `Leech_Cocode_Turyn_Note.pdf` | The construction/verification write-up ("The Golay Cocode as the Dual Address Space for Turyn Glue") |

## License

MIT — see `LICENSE`.