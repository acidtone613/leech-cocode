"""
decoder.py — The Cocode-Turyn dual decoder and two independent exact ML oracles.

Scales: all vectors are in Lambda~ = sqrt(2)*Lambda_24 coordinates.
  * min dist of Lambda~ = 2*sqrt(2); BDD (unique-decoding) radius sqrt(2)
    here, i.e. radius 1 in Lambda_24 scale.
  * noise sigma is passed in Lambda_24 scale (internally scaled by sqrt(2)).

DECODER (syndrome/table form), parameter W (candidate weight bound, def. 4):
    z = round(y); rho = y - z
    p = z mod 2 ; s = syndrome(p) in C* (12-bit cocode address)
    for each w in cand[s] (all coset words of weight <= W):
        delta_i = sign(rho_i) on supp(w), 0 elsewhere
        if sum(z+delta) != 0 mod 4: one +-2 at cheapest off-support slot
        candidate x = z + delta
    repeat for the tau shell (y - tau); return the best.

  Provably exact whenever ||y - x_hat|| < 1 (Lambda~ scale) with W >= 4.

ORACLE A (families): enumerate all 8192 families {x = B + 2m, sum = r mod 4},
  B in {codewords} x {tau + codewords}; componentwise closest + one +-2.
  Exact ML. Vectorized numpy.

ORACLE B (glue cosets): enumerate all 4096 glue cosets with per-block exact
  E8 decoding. Exact ML. Independent of oracle A.
"""
import math
import numpy as np

from leech_glue import LeechGlue, T, TAU


class CocodeTurynDecoder:
    def __init__(self, W=4):
        self.L = LeechGlue()
        self.W = W
        self._cand = None            # extended candidate table (built lazily)
        self._cand_W = None
        self.codeMat = np.array([[(c >> i) & 1 for i in range(24)]
                                 for c in sorted(self.L.codeSet)], dtype=np.float64)
        self.B = np.vstack([self.codeMat, self.codeMat + TAU[None, :]])
        self.Rtarget = np.concatenate([np.zeros(4096), np.full(4096, 10.0)])
        self._glue_reps = None

    @property
    def cand(self):
        """syndrome -> list of all coset words of weight <= W (extended table
        is built on demand when W exceeds the default Golay covering radius)."""
        if self._cand is None or self._cand_W != self.W:
            if self.W <= 4:
                self._cand = self.L.cand
            else:
                import itertools
                tbl = {}
                for k in range(0, self.W + 1):
                    for supp in itertools.combinations(range(24), k):
                        v = sum(1 << i for i in supp)
                        tbl.setdefault(self.L.syndrome(v), []).append(v)
                for s in tbl:
                    tbl[s].sort(key=lambda v: bin(v).count("1"))
                self._cand = tbl
            self._cand_W = self.W
        return self._cand

    # ------------------------------------------------------------ decoder
    def _decode_shell(self, y):
        z = np.round(y)
        rho = y - z
        p = 0
        for i in range(24):
            p |= (int(z[i]) & 1) << i
        s = self.L.syndrome(p)
        sz = int(z.sum())
        slotcost = (2.0 - np.abs(rho)) ** 2
        order = np.argsort(slotcost)
        best_x, best_d = None, math.inf
        for w in self.cand[s]:
            supp = [i for i in range(24) if (w >> i) & 1]
            delta = np.zeros(24)
            for i in supp:
                delta[i] = 1.0 if rho[i] >= 0 else -1.0
            if (sz + int(delta.sum())) % 4:
                for i in order:
                    if delta[i] == 0:
                        delta[i] = 2.0 if rho[i] >= 0 else -2.0
                        break
            d = ((rho - delta) ** 2).sum()
            if d < best_d:
                best_d = d
                best_x = z + delta
        return best_x, best_d

    def decode(self, y):
        """Cocode-Turyn dual decode of y (Lambda~ coords). -> (x, dist2)."""
        x1, d1 = self._decode_shell(y)
        x2, d2 = self._decode_shell(y - TAU)
        if d1 <= d2:
            return x1, d1
        return x2 + TAU, d2

    # ------------------------------------------------------------ oracle A
    def oracle_families(self, Y):
        """Exact ML squared distance for each row of Y. Vectorized."""
        Y = np.atleast_2d(Y)
        N = Y.shape[0]
        best = np.full(N, np.inf)
        for c0 in range(0, 8192, 2048):
            Bc = self.B[c0:c0 + 2048]
            Rt = self.Rtarget[c0:c0 + 2048]
            for i0 in range(0, N, 16):
                Yc = Y[i0:i0 + 16]
                X = Bc[None, :, :] + 2.0 * np.round((Yc[:, None, :] - Bc[None, :, :]) / 2.0)
                S = X.sum(axis=2)
                need = (S - Rt[None, :]) % 4.0 != 0
                D = ((Yc[:, None, :] - X) ** 2).sum(axis=2)
                if need.any():
                    RHO = Yc[:, None, :] - X
                    costp = (2.0 - np.abs(RHO)) ** 2 - RHO ** 2   # net fix cost
                    idx = np.argmin(costp, axis=2)
                    fix = np.take_along_axis(costp, idx[..., None], axis=2)[..., 0]
                    D = D + np.where(need, fix, 0.0)
                best[i0:i0 + 16] = np.minimum(best[i0:i0 + 16], D.min(axis=1))
        return best if best.shape[0] > 1 else best[0]

    # ------------------------------------------------------------ oracle B
    def oracle_glue(self, y):
        """Exact ML by 4096 glue cosets with per-block exact E8 decode."""
        if self._glue_reps is None:
            self._glue_reps = self.L.glue_reps()
        best = math.inf
        for tg in self._glue_reps:
            d = 0.0
            for b in range(3):
                v = (y[8 * b:8 * b + 8] - tg[8 * b:8 * b + 8]) / 2.0
                e = self.L.e8_nearest(v)
                d += float(((v - e) ** 2).sum()) * 4.0
            if d < best:
                best = d
        return best
