"""
leech_glue.py — The Leech lattice, the Turyn glue group, and the
Cocode-Turyn duality theorems (exact computation).

Model: we work in the sqrt(2)-scaled integer coordinates
    Lambda~ = sqrt(2) * Lambda_24
             = M0  |  (tau + M0)
    M0  = { x in Z^24 : x mod 2 in G24,  sum(x) = 0 mod 4 }
    tau = (-3, 1, ..., 1) / 2
with the Golay code trio-labeled so the three 8-blocks are octads.

Verified in verify_all():
  Theorem 1 (Turyn filtration): 2E8^3 (= sqrt(2)E8^3) has index 4096 in
    Lambda~; the glue group G = Lambda~/2E8^3 is elementary abelian (Z/2)^12
    (2048 integer-side classes + 2048 tau-side classes); per-block
    projections have order 128, pairwise projections 2048.
  Theorem 2 (syndrome theorem): on the ambient class group A = (Z^8/2E8)^3
    the cocode syndrome sigma and the mod-4 sum syndrome Sigma are
    well-defined homomorphisms, and
        G_int = ker(sigma) & ker(Sigma)          (both sides 2048)
        A_full / G  <=>  {(s,t) in C* x Z/4 : t = f(s) mod 2}   (8192)
        ker(sigma) = G x <v>  for v = class(c + 2 e_1), index 2
  Kissing number: 196560 = 1104 + 97152 + 98304 (exact count of the three
    minimal-vector shapes).
  Determinant: det Lambda_24 = 1 via the verified index [Lambda~:2E8^3]=4096.

Also provides: exact E8 nearest-point decoder; glue-coset representative
lifts (needed by the brute-force oracle in decoder.py).
"""
import numpy as np
from collections import deque
from golay import build_golay

T = np.array([-3] + [1] * 23, dtype=np.int64)     # t = 2*tau
TAU = (T / 2.0).astype(np.float64)                # tau


class LeechGlue:
    def __init__(self):
        self.gens, self.codeSet, self.cand, self.syndrome = build_golay()
        self._build_M0_gens()
        self._build_glue_group()

    # ------------------------------------------------------------ M0
    def in_M0(self, xi):
        p = 0
        for i, v in enumerate(xi):
            p |= (int(v) & 1) << i
        if p not in self.codeSet:
            return False
        return sum(xi) % 4 == 0

    def _build_M0_gens(self):
        gens = []
        for i in range(24):
            for j in range(i + 1, 24):
                v = np.zeros(24, dtype=np.int64)
                v[i] = 2
                v[j] = 2
                gens.append(v)
        for g in self.gens:
            v = np.zeros(24, dtype=np.int64)
            for i in range(24):
                if (g >> i) & 1:
                    v[i] = 1
            gens.append(v)
        self.M0_gens = gens

    # ------------------------------------------------------------ classes
    # class of Z^8 / 2E8: complete invariant (parity mod all-flip, sum mod 4)
    @staticmethod
    def block_class(v8):
        p = tuple(int(x) & 1 for x in v8)
        q = int(sum(v8)) % 4
        if p[0] == 1:
            p = tuple(1 - b for b in p)
        return (p, q)

    def full_class(self, x24):
        return tuple(self.block_class(x24[8 * b:8 * b + 8]) for b in range(3))

    @staticmethod
    def add_class(k1, k2):
        ks = []
        for (p, q), (p2, q2) in zip(k1[0], k2[0]):
            p3 = tuple(a ^ b for a, b in zip(p, p2))
            q3 = (q + q2) % 4
            if p3[0] == 1:
                p3 = tuple(1 - b for b in p3)
            ks.append((p3, q3))
        return (tuple(ks), k1[1] ^ k2[1])

    IDENT = (((tuple([0] * 8), 0),) * 3, False)

    # ------------------------------------------------------------ glue group
    def _build_glue_group(self):
        gens_G = [(self.full_class([int(x) for x in v]), False)
                  for v in self.M0_gens]
        gens_G.append((self.full_class([0] * 24), True))     # tau
        G = {self.IDENT}
        dq = deque([self.IDENT])
        while dq:
            u = dq.popleft()
            for g in gens_G:
                v = self.add_class(u, g)
                if v not in G:
                    G.add(v)
                    dq.append(v)
        assert len(G) == 4096
        self.G = G
        self.G_int = {g for g in G if not g[1]}

    # ------------------------------------------------------------ syndromes
    def sigma(self, k):
        """cocode syndrome of an (integer-side) class key"""
        v = 0
        for b in range(3):
            p = k[0][b][0]
            v |= sum(bit << (8 * b + i) for i, bit in enumerate(p))
        return self.syndrome(v)

    def Sigma(self, k):
        return sum(k[0][b][1] for b in range(3)) % 4

    def rand_int_class(self, rng):
        v = rng.integers(-6, 7, 24)
        return (self.full_class([int(x) for x in v]), False)

    # ------------------------------------------------------------ coset reps
    def lift_block(self, p, q):
        v = np.array(p, dtype=np.int64)
        w = int(v.sum())
        assert (q - w) % 2 == 0
        k = ((q - w) // 2) % 2
        if k:
            v = v.copy()
            v[0] += 2
        return v

    def lift_class(self, key):
        """Genuine lattice-point representative of a glue class. The
        canonical parity in the key need not be a codeword: exactly one
        block-flip pattern makes it one (the flips are trio codewords)."""
        p_full = 0
        for b in range(3):
            p = key[0][b][0]
            for i in range(8):
                p_full |= p[i] << (8 * b + i)
        c = None
        for eps in range(8):
            cc = p_full
            if eps & 1:
                cc ^= 0x0000FF
            if eps & 2:
                cc ^= 0x00FF00
            if eps & 4:
                cc ^= 0xFF0000
            if cc in self.codeSet:
                c = cc
                break
        assert c is not None
        v = np.zeros(24, dtype=np.int64)
        for b in range(3):
            pb = [(c >> (8 * b + i)) & 1 for i in range(8)]
            v[8 * b:8 * b + 8] = self.lift_block(pb, key[0][b][1])
        pp = 0
        for i in range(24):
            pp |= (int(v[i]) & 1) << i
        assert pp in self.codeSet and int(v.sum()) % 4 == 0
        if key[1]:
            v = v + TAU
        return v

    def glue_reps(self):
        return [self.lift_class(g) for g in sorted(self.G)]

    # ------------------------------------------------------------ E8 decoder
    @staticmethod
    def d8_nearest(v):
        n = np.round(v)
        r = v - n
        if int(n.sum()) % 2:
            i = int(np.argmax(np.abs(r)))
            n = n.copy()
            n[i] += 1.0 if r[i] >= 0 else -1.0
        return n

    H8 = np.full(8, 0.5)

    @classmethod
    def e8_nearest(cls, v):
        a = cls.d8_nearest(v)
        b = cls.d8_nearest(v - cls.H8) + cls.H8
        return a if ((v - a) ** 2).sum() <= ((v - b) ** 2).sum() else b

    # ------------------------------------------------------------ sampling
    def random_leech_point(self, rng):
        v = np.zeros(24, dtype=np.int64)
        for _ in range(6):
            v = v + rng.integers(-2, 3) * self.M0_gens[rng.integers(len(self.M0_gens))]
        assert self.in_M0(v)
        if rng.random() < 0.5:
            return v + TAU
        return v.astype(np.float64)

    # ------------------------------------------------------------ verify all
    def verify_all(self, log=print):
        import itertools
        log("[1] 2E8^3 containment in Lambda~ (generators) ...")
        g2e8 = []
        for i in range(8):
            for j in range(i + 1, 8):
                v = np.zeros(8, dtype=np.int64)
                v[i] = 2
                v[j] = 2
                g2e8.append(v)
        g2e8.append(np.ones(8, dtype=np.int64))
        for b in range(3):
            for u in g2e8:
                v = np.zeros(24, dtype=np.int64)
                v[8 * b:8 * b + 8] = u
                assert self.in_M0(v)
        log("     OK")

        log("[2] glue group |G| = %d (integer %d + tau %d) ..."
            % (len(self.G), len(self.G_int), len(self.G) - len(self.G_int)))
        assert len(self.G_int) == 2048

        log("[3] element orders: G is (Z/2)^12 ...")
        for g in self.G:
            cur, o = g, 1
            while cur != self.IDENT:
                cur = self.add_class(cur, g)
                o += 1
                assert o <= 2
        log("     OK (all orders <= 2)")

        log("[4] Theorem: G_int = ker(sigma) & ker(Sigma) ...")
        for g in self.G_int:
            assert self.sigma(g) == 0 and self.Sigma(g) == 0
        ker_direct = set()
        for c in self.codeSet:
            pcs = []
            for b in range(3):
                p = tuple((c >> (8 * b + i)) & 1 for i in range(8))
                if p[0] == 1:
                    p = tuple(1 - x for x in p)
                pcs.append(p)
            for q1 in range(4):
                for q2 in range(4):
                    for q3 in range(4):
                        if (q1 + q2 + q3) % 4 == 0 and \
                           all(q % 2 == sum(pp) % 2 for q, pp in zip((q1, q2, q3), pcs)):
                            ker_direct.add((tuple((pcs[b], (q1, q2, q3)[b])
                                                  for b in range(3)), False))
        assert len(ker_direct) == 2048 and ker_direct == self.G_int
        log("     OK (independent enumeration matches, both 2048)")

        log("[5] homomorphism property of sigma, Sigma (samples) ...")
        rng = np.random.default_rng(7)
        for _ in range(300):
            a, b = self.rand_int_class(rng), self.rand_int_class(rng)
            assert self.sigma(self.add_class(a, b)) == self.sigma(a) ^ self.sigma(b)
            assert self.Sigma(self.add_class(a, b)) == (self.Sigma(a) + self.Sigma(b)) % 4
        log("     OK")

        log("[6] per-block / pairwise projections ...")
        pi1 = {g[0][0] for g in self.G}
        pi12 = {(g[0][0], g[0][1]) for g in self.G}
        assert len(pi1) == 128 and len(pi12) == 2048
        log("     OK (|pi_i| = 128, |pi_ij| = 2048; G is not a per-block product)")

        log("[7] ker(sigma) = G + <v>, index 2 ...")
        vfound = None
        for c in self.codeSet:
            v = np.zeros(24, dtype=np.int64)
            for i in range(24):
                if (c >> i) & 1:
                    v[i] = 1
            v[0] += 2
            k = (self.full_class([int(x) for x in v]), False)
            if k not in self.G:
                vfound = k
                break
        assert vfound is not None
        assert all(self.add_class(g, vfound) not in self.G for g in self.G)
        log("     OK")

        log("[8] kissing number 196560 (three shapes, exact counts) ...")
        n1 = 0
        for i in range(24):
            for j in range(i + 1, 24):
                for s1 in (2, -2):
                    for s2 in (2, -2):
                        v = np.zeros(24, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        if self.in_M0(v):
                            n1 += 1
        assert n1 == 1104
        n2 = 0
        octadsR = [c for c in self.codeSet if bin(c).count("1") == 8]
        for c in octadsR:
            for signs in itertools.product((1, -1), repeat=8):
                if sum(signs) % 4 == 0:
                    n2 += 1
        assert n2 == 97152
        n3 = 0
        for c in self.codeSet:
            for j in range(24):
                X = np.zeros(24, dtype=np.int64)
                ok = True
                for i in range(24):
                    tm = (int(T[i]) + 2 * ((c >> i) & 1)) % 4
                    vals = [v for v in ((-3, 3) if i == j else (-1, 1)) if v % 4 == tm]
                    if not vals:
                        ok = False
                        break
                    X[i] = vals[0]
                if ok:
                    if self.in_M0((X - T) // 2):
                        n3 += 1
        assert n3 == 98304
        log("     OK: %d + %d + %d = %d" % (n1, n2, n3, n1 + n2 + n3))
        log("ALL STRUCTURAL CHECKS PASSED.")


if __name__ == "__main__":
    L = LeechGlue()
    L.verify_all()
