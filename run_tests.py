"""
run_tests.py — Full verification and exactness test suite for the
Cocode-Turyn dual decoder (the Phase 1 deliverable).

Stages (all must pass):
  1. Structural verification of the Golay code, the Leech model, the glue
     group and the syndrome theorems (exact arithmetic).
  2. Cross-validation: the decoder and BOTH independent exact ML oracles
     agree on random inputs.
  3. Exactness suite: N random inputs per noise level; the decoder is
     compared against exact ML; statistics inside/outside the BDD ball.
  4. Candidate-weight sweep (W = 4, 5, 6): the exactness/speed knob.

Typical runtime: ~60-90 s (pure Python decoder, numpy oracles).
"""
import time
import math
import numpy as np

from leech_glue import LeechGlue
from decoder import CocodeTurynDecoder

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.2f}s]", *a, flush=True)


def main():
    # ------------------------------------------------ stage 1: structure
    log("=== Stage 1: structural verification (exact) ===")
    dec = CocodeTurynDecoder(W=4)
    dec.L.verify_all(log=log)

    rng = np.random.default_rng(42)

    # ------------------------------------------------ stage 2: cross-validation
    log("=== Stage 2: decoder vs two independent exact oracles ===")
    # sigma=0.12: ~93% of points inside the provable radius-1 region.
    Y = np.array([dec.L.random_leech_point(rng) + rng.normal(0, 0.12 * math.sqrt(2), 24)
                  for _ in range(30)])
    d_dec = np.array([dec.decode(y)[1] for y in Y])
    d_fam = dec.oracle_families(Y)
    d_glu = np.array([dec.oracle_glue(y) for y in Y])
    assert np.allclose(d_fam, d_glu, atol=1e-9), "two exact oracles disagree!"
    assert (d_dec - d_fam).min() > -1e-9, "decoder beat ML: impossible"
    r1 = d_fam < 1.0
    assert np.isclose(d_dec[r1], d_fam[r1], atol=1e-9).all(), \
        "decoder not exact inside provable radius"
    log("oracles agree to %.1e; decoder exact in radius-1 region (%d/%d pts); "
        "overall agreement %.1f%% (sigma=0.12)"
        % (np.abs(d_fam - d_glu).max(), r1.sum(), len(Y),
           np.isclose(d_dec, d_fam, atol=1e-9).mean() * 100))

    # ------------------------------------------------ stage 3: exactness suite
    log("=== Stage 3: exactness suite (N points per sigma) ===")
    N = 2000
    for sigma24 in (0.2, 0.35):
        rng = np.random.default_rng(1234 + int(sigma24 * 100))
        Y = np.array([dec.L.random_leech_point(rng)
                      + rng.normal(0, sigma24 * math.sqrt(2), 24) for _ in range(N)])
        t0 = time.time()
        d_dec = np.array([dec.decode(y)[1] for y in Y])
        t_us = (time.time() - t0) / N * 1e6
        d_fam = dec.oracle_families(Y)
        assert (d_dec - d_fam).min() > -1e-9, "decoder beat ML: impossible"
        agree = np.isclose(d_dec, d_fam, atol=1e-9)
        bdd = d_fam < 2.0
        r1 = d_fam < 1.0                      # provable exactness radius
        log(f"sigma={sigma24}: ML agreement {agree.mean()*100:.2f}% | "
            f"radius-1 agreement {agree[r1].mean()*100 if r1.any() else 100:.2f}% | "
            f"in-BDD {agree[bdd].mean()*100 if bdd.any() else 100:.3f}% | "
            f"{t_us:.0f} us/pt")

    # ------------------------------------------------ stage 4: W sweep
    log("=== Stage 4: candidate-weight sweep (W knob) ===")
    rng = np.random.default_rng(555)
    Y = np.array([dec.L.random_leech_point(rng) + rng.normal(0, 0.2 * math.sqrt(2), 24)
                  for _ in range(500)])
    d_fam = dec.oracle_families(Y)
    for W in (4, 5, 6):
        dec.W = W
        dec.decode(Y[0])                      # warm up (build extended table)
        t0 = time.time()
        d_dec = np.array([dec.decode(y)[1] for y in Y])
        t_us = (time.time() - t0) / len(Y) * 1e6
        log(f"W={W}: ML agreement {np.isclose(d_dec, d_fam, atol=1e-9).mean()*100:.2f}%"
            f"  ({t_us:.0f} us/pt)")
    log("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
