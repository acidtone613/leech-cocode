"""
golay.py — The extended binary Golay code G24, from first principles.

Construction: extended quadratic-residue code of length 23 over GF(2^11)
(primitive polynomial x^11 + x^2 + 1, alpha a primitive 23rd root of unity),
extended by an overall parity bit.

Everything is exact (Python ints as F2 bitmasks, bit i = coordinate i).

Verified here:
  * dimension 12, 4096 codewords
  * weight enumerator {0:1, 8:759, 12:2576, 16:759, 24:1}
  * self-duality (generators pairwise orthogonal)
  * a trio relabeling so the three 8-blocks are disjoint octads
  * the coset-leader table: all weight<=4 words grouped by syndrome
    (covering radius 4; 12951 words, avg 3.16 per coset)

Part of the leech_cocode package (Cocode-Turyn dual decoder for the Leech
lattice). See the companion note: "The Golay Cocode as the Dual Address
Space for Turyn Glue".
"""
import itertools
from collections import deque

# ---------------------------------------------------------------- GF(2^11)
PMOD = (1 << 11) | (1 << 2) | 1          # x^11 + x^2 + 1 (primitive)


def _gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & (1 << 11):
            a ^= PMOD
    return r


def _gf_pow(a, e):
    r = 1
    while e:
        if e & 1:
            r = _gf_mul(r, a)
        a = _gf_mul(a, a)
        e >>= 1
    return r


def build_golay():
    """Return (gens, codeSet, cand, syndrome) for the trio-labeled G24.

    gens    : list of 12 independent generator bitmasks (also a parity-check
              basis, since the code is self-dual)
    codeSet : set of all 4096 codeword bitmasks
    cand    : dict syndrome -> list of all weight<=4 words in that coset
    syndrome: function F2^24 bitmask -> 12-bit syndrome (the cocode address)
    """
    beta = 2
    assert _gf_pow(beta, 2047) == 1
    assert _gf_pow(beta, 2047 // 23) != 1 and _gf_pow(beta, 2047 // 89) != 1
    alpha = _gf_pow(beta, 89)
    assert _gf_pow(alpha, 23) == 1 and alpha != 1

    Q = sorted({(j * j) % 23 for j in range(1, 23)})    # nonzero QRs mod 23
    assert len(Q) == 11

    poly = [1]                                          # generator polynomial
    for i in Q:
        a = _gf_pow(alpha, i)
        new = [0] * (len(poly) + 1)
        for k in range(len(poly)):
            new[k + 1] ^= poly[k]
            new[k] ^= _gf_mul(a, poly[k])
        poly = new
    assert len(poly) == 12 and all(c in (0, 1) for c in poly)

    gint = sum(c << k for k, c in enumerate(poly))
    MASK23 = (1 << 23) - 1

    def shift_reduce(a, j):
        v = a << j
        while v > MASK23:
            v = (v & MASK23) ^ (v >> 23)
        return v

    def wt(v):
        return bin(v).count("1")

    gens23 = [shift_reduce(gint, j) for j in range(12)]
    seen = {0}                                          # F2-linear span
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for gv in gens23:
            v = u ^ gv
            if v not in seen:
                seen.add(v)
                dq.append(v)
    assert len(seen) == 4096

    MASK24 = (1 << 24) - 1
    code = {c | ((wt(c) & 1) << 23) for c in seen}
    assert len(code) == 4096

    enum = {}
    for c in code:
        enum[wt(c)] = enum.get(wt(c), 0) + 1
    assert enum == {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}

    # independent generators
    gens, span = [], {0}
    for c in sorted(code):
        if c not in span:
            gens.append(c)
            span = {s ^ c for s in span} | span
            if len(gens) == 12:
                break
    assert len(span) == 4096
    d = lambda a, b: wt(a & b) & 1
    assert all(d(gens[i], gens[j]) == 0 for i in range(12) for j in range(12))

    # ---- trio relabeling: three disjoint octads become the three blocks
    octads = [c for c in code if wt(c) == 8]
    c1 = octads[0]
    c2 = next(c for c in octads if c & c1 == 0)
    c3 = (~(c1 | c2)) & MASK24
    assert wt(c3) == 8 and c3 in code
    perm = []
    for c in (c1, c2, c3):
        perm += [i for i in range(24) if (c >> i) & 1]
    assert sorted(perm) == list(range(24))

    relabel = lambda v: sum(((v >> old) & 1) << new for new, old in enumerate(perm))
    codeSet = {relabel(c) for c in code}
    gens = [relabel(g) for g in gens]
    assert 0x0000FF in codeSet and 0x00FF00 in codeSet and 0xFF0000 in codeSet

    def syndrome(v):
        s = 0
        for i, g in enumerate(gens):
            s |= d(v, g) << i
        return s

    cand = {}
    for k in range(0, 5):
        for supp in itertools.combinations(range(24), k):
            v = sum(1 << i for i in supp)
            cand.setdefault(syndrome(v), []).append(v)
    assert len(cand) == 4096
    for s in cand:
        cand[s].sort(key=wt)
    return gens, codeSet, cand, syndrome


if __name__ == "__main__":
    gens, codeSet, cand, syn = build_golay()
    n = sum(len(v) for v in cand.values())
    print("G24 verified: 4096 words, enumerator 1,759,2576,759,1, self-dual,")
    print("trio-labeled; coset table: %d weight<=4 words over %d syndromes"
          " (avg %.2f/coset)" % (n, len(cand), n / 4096))
